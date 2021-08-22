from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD
import numpy as np
import pandas as pd
import keras
import math
import argparse
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Run Dual-MGAN.")
    parser.add_argument('--path_out', nargs='?', default='/Data/out10.csv',
                        help='Identified Anomalies path.')
    parser.add_argument('--path_unl', nargs='?', default='/Data/unl10.csv',
                        help='Unlabeled data path.')
    parser.add_argument('--path_test', nargs='?', default='/Data/test.csv',
                        help='Test data path.')
    parser.add_argument('--save_model', nargs='?', default='discriminator.h5',
                        help='The final model.')
    parser.add_argument('--k_means', type=int, default=10,
                        help='The k in k-means.')
    parser.add_argument('--max_iter_MGAOS', type=int, default=2000,
                        help='Stop training sub_generators in MGAOS after max_iter_MGAOS.')
    parser.add_argument('--max_iter_MGAAL', type=int, default=1000,
                        help='Stop training sub_generators in MGAAL after max_iter_MGAAL.')
    parser.add_argument('--lr_sg', type=float, default=0.0001,
                        help='Learning rate of sub_generators.')
    parser.add_argument('--lr_sd', type=float, default=0.01,
                        help='Learning rate of sub_discriminators.')
    parser.add_argument('--lr_d', type=float, default=0.001,
                        help='Learning rate of the detector.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='batch_size.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--nnr_MGAOS', type=float, default=0.4,
                        help='The thresholds of Nnr in MGAOS.')
    parser.add_argument('--nnr_MGAAL', type=float, default=0.2,
                        help='The thresholds of Nnr in MGAAL.')
    return parser.parse_args()

# Sub-Generator
def create_generator(latent_size):
    gen = Sequential()
    gen.add(Dense(latent_size, input_dim=latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    gen.add(Dense(latent_size, activation='relu', kernel_initializer=keras.initializers.Identity(gain=1.0)))
    latent = Input(shape=(latent_size,))
    fake_data = gen(latent)
    return Model(latent, fake_data)

# Sub-Discriminator
def create_sub_discriminator(size):
    dis = Sequential()
    dis.add(Dense(size, input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(10, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)

# Detector
def create_discriminator(size):
    dis = Sequential()
    dis.add(Dense(size, input_dim=latent_size, activation='relu', kernel_initializer= keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)))
    dis.add(Dense(10, activation='relu', kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None)))
    dis.add(Dropout(0.2))
    dis.add(Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None)))
    data = Input(shape=(latent_size,))
    fake = dis(data)
    return Model(data, fake)


# Load data
def load_data():
    data_out = pd.read_table('{path}'.format(path = args.path_out), sep=' ', header=None)
    data_unl = pd.read_table('{path}'.format(path = args.path_unl), sep=' ', header=None)
    data_out = data_out.sample(frac=1).reset_index(drop=True)
    data_unl = data_unl.sample(frac=1).reset_index(drop=True)
    id_out = data_out.pop(0)
    id_unl = data_unl.pop(0)
    y_out = data_out.pop(1)
    y_unl = data_unl.pop(1)
    data_out_x = data_out.as_matrix()
    data_unl_x = data_unl.as_matrix()
    data_id_out = id_out.values
    data_id_unl = id_unl.values
    data_out_y = y_out.values
    data_unl_y = y_unl.values
    return data_out_x, data_unl_x, data_id_out, data_id_unl, data_out_y, data_unl_y

# Load test data
def load_test_data():
    data = pd.read_table('{path}'.format(path = args.path_test), sep=' ', header=None)
    data = data.sample(frac=1, replace=False).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1)
    data_x = data.as_matrix()
    data_id = id.values
    data_y = y.values
    return data_x, data_id, data_y

if __name__ == '__main__':
    # initilize arguments
    args = parse_args()
    names = locals()
    auc_d = 0

    # initialize dataset
    data_out_x, data_unl_x, data_id_out, data_id_unl, data_out_y, data_unl_y = load_data()
    test_x, test_id, test_y = load_test_data()
    data_x = np.concatenate((data_out_x, data_unl_x), axis=0)
    data_y = np.concatenate((data_out_y, data_unl_y), axis=0)
    data_out_size = data_out_x.shape[0]
    data_unl_size = data_unl_x.shape[0]
    data_size = data_out_size + data_unl_size
    latent_size = data_x.shape[1]
    batch_size = min(args.batch_size, data_size)
    mul = math.ceil(data_unl_size/data_out_size)-1
    print("The dimensions of the outliers:{}*{}".format(data_out_size, latent_size))
    print("The dimensions of the unlabeled data:{}*{}".format(data_unl_size, latent_size))

    # The number of the outliers in test
    top_n = 0
    for i in range(test_y.shape[0]):
        top_n = test_y[i]+top_n

    # k-means
    k_out = min(data_out_size, args.k_means)
    k_unl = min(data_unl_size, args.k_means)
    if data_out_size <= args.k_means:
        kmeans_cen_out = data_out_x
    else:
        kmeans_out = KMeans(n_clusters=k_out, random_state=0, max_iter=1000).fit(data_out_x)
        kmeans_cen_out = pd.DataFrame(kmeans_out.cluster_centers_)
        kmeans_cen_out = kmeans_cen_out.as_matrix()
    kmeans_unl = KMeans(n_clusters=k_unl, random_state=0, max_iter=1000).fit(data_unl_x)
    kmeans_cen_unl = pd.DataFrame(kmeans_unl.cluster_centers_)
    kmeans_cen_unl = kmeans_cen_unl.as_matrix()
    for i in range(k_out):
        names['data_out_x_' + str(i)] = []
        names['data_out_num_x_' + str(i)] = 0
    for idx in range(data_out_size):
        dists_out = np.sqrt(np.sum((data_out_x[idx,] - kmeans_cen_out) ** 2, axis=1))
        index = np.argsort(dists_out)
        for i in range(k_out):
            if index[0] == i:
                if names['data_out_x_' + str(i)] == []:
                    names['data_out_x_' + str(i)] = data_out_x[idx,].reshape(1,latent_size)
                else:
                    names['data_out_x_' + str(i)] = np.concatenate((names['data_out_x_' + str(i)], data_out_x[idx,].reshape(1,latent_size)), axis=0)
        names['data_out_num_x_' + str(i)] = names['data_out_num_x_' + str(i)]+1
    for i in range(k_unl):
        names['data_unl_x_' + str(i)] = []
    for idx in range(data_unl_size):
        dists_unl = np.sqrt(np.sum((data_unl_x[idx,] - kmeans_cen_unl) ** 2, axis=1))
        index = np.argsort(dists_unl)
        for i in range(k_unl):
            if index[0] == i:
                if names['data_unl_x_' + str(i)] == []:
                    names['data_unl_x_' + str(i)] = data_unl_x[idx,].reshape(1,latent_size)
                else:
                    names['data_unl_x_' + str(i)] = np.concatenate((names['data_unl_x_' + str(i)], data_unl_x[idx,].reshape(1,latent_size)), axis=0)

    # Create sub-discriminator
    for i in range(k_out):
        names['discriminator_out_' + str(i)] = create_sub_discriminator(min(data_size,1000))
        names['discriminator_out_' + str(i)].compile(optimizer=SGD(lr=args.lr_sd, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')
    for i in range(k_unl):
        names['discriminator_unl_' + str(i)] = create_sub_discriminator(min(data_size,1000))
        names['discriminator_unl_' + str(i)].compile(optimizer=SGD(lr=args.lr_sd, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

    #Create detector
    discriminator_all = create_discriminator(min(data_size,1000))
    discriminator_all.compile(optimizer=SGD(lr=args.lr_d, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

    # Create sub-generator and combine_model
    for i in range(k_out):
        names['generator_out_' + str(i)] = create_generator(latent_size)
        latent = Input(shape=(latent_size,))
        names['fake_out_' + str(i)] = names['generator_out_' + str(i)](latent)
        names['discriminator_out_' + str(i)].trainable = False
        names['fake_out_' + str(i)] = names['discriminator_out_' + str(i)](names['fake_out_' + str(i)])
        names['combine_model_out_' + str(i)] = Model(latent, names['fake_out_' + str(i)])
        names['combine_model_out_' + str(i)].compile(optimizer=SGD(lr=args.lr_sg, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')
    for i in range(k_unl):
        names['generator_unl_' + str(i)] = create_generator(latent_size)
        latent = Input(shape=(latent_size,))
        names['fake_unl_' + str(i)] = names['generator_unl_' + str(i)](latent)
        names['discriminator_unl_' + str(i)].trainable = False
        names['fake_unl_' + str(i)] = names['discriminator_unl_' + str(i)](names['fake_unl_' + str(i)])
        names['combine_model_unl_' + str(i)] = Model(latent, names['fake_unl_' + str(i)])
        names['combine_model_unl_' + str(i)].compile(optimizer=SGD(lr=args.lr_sg, decay=args.decay, momentum=args.momentum), loss='binary_crossentropy')

    # pre-training the MGAOS
    for i in range(k_out):
        names['stop_out_' + str(i)] = 0
        names['dis_out_' + str(i)] = 0
        names['generated_data_out_all_' + str(i)] = []
        names['nash_out_' + str(i)] = 0
        if names['data_out_x_' + str(i)].shape[0] == 1:
            dists_out = np.sqrt(np.sum((names['data_out_x_' + str(i)] - data_x) ** 2, axis=1))
            index = np.argsort(dists_out)
            names['dis_out_' + str(i)] = dists_out[index[4]]
        elif names['data_out_x_' + str(i)].shape[0] <= 10:
            for idx in range(names['data_out_x_' + str(i)].shape[0]):
                dists_out = np.sum(np.sqrt(np.sum((names['data_out_x_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2, axis=1)), axis=0)
                names['dis_out_' + str(i)] = names['dis_out_' + str(i)] + dists_out
            names['dis_out_' + str(i)] = names['dis_out_' + str(i)]/(names['data_out_x_' + str(i)].shape[0]*(names['data_out_x_' + str(i)].shape[0]-1))

    stop_out = 0
    for epoch in range(args.max_iter_MGAOS):
        print('Epoch_out {} of {}'.format(epoch + 1, args.max_iter_MGAOS))
        for i in range(k_out):
            if names['stop_out_' + str(i)] == 0:
                names['data_out_x_' + str(i)] = pd.DataFrame(names['data_out_x_' + str(i)])
                noise = np.random.uniform(0, 1, (int(names['data_out_x_' + str(i)].shape[0]), latent_size))
                names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                names['x_out_' + str(i)] = np.concatenate((names['data_out_x_' + str(i)], names['generated_data_out_' + str(i)]), axis=0)
                names['y_out_' + str(i)] = np.array([1] * (int(names['data_out_x_' + str(i)].shape[0])) + [0] * (int(names['data_out_x_' + str(i)].shape[0])))
                names['discriminator_out' + str(i)] = names['discriminator_out_' + str(i)].train_on_batch(names['x_out_' + str(i)], names['y_out_' + str(i)])
                trick_out = np.array([1] * (names['data_out_x_' + str(i)].shape[0]))
                names['generator_out' + str(i)] = names['combine_model_out_' + str(i)].train_on_batch(noise, trick_out)

                # The evaluation of sub-GANs
                if names['data_out_x_' + str(i)].shape[0] == 1:
                    dis = np.linalg.norm(names['generated_data_out_' + str(i)] - names['data_out_x_' + str(i)])
                    if dis <= names['dis_out_' + str(i)]:
                        if names['generated_data_out_all_' + str(i)] == []:
                            names['generated_data_out_all_' + str(i)] = names['generated_data_out_' + str(i)].reshape(1,latent_size)
                        else:
                            names['generated_data_out_all_' + str(i)] =np.concatenate((names['generated_data_out_all_' + str(i)], names['generated_data_out_' + str(i)].reshape(1,latent_size)), axis=0)
                        if names['generated_data_out_all_' + str(i)].shape[0] >= mul:
                            names['stop_out_' + str(i)] = 1
                            stop_out = stop_out + names['stop_out_' + str(i)]
                            break
                        noise = np.random.uniform(0, 1, (mul, latent_size))
                        names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                        for idx in range(mul):
                            dis = np.linalg.norm(names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)])
                            if dis <= names['dis_out_' + str(i)]:
                                names['generated_data_out_all_' + str(i)] = np.concatenate((names['generated_data_out_all_' + str(i)], names['generated_data_out_' + str(i)][idx,].reshape(1,latent_size)), axis=0)
                                if names['generated_data_out_all_' + str(i)].shape[0] >= mul:
                                    names['stop_out_' + str(i)] = 1
                                    stop_out = stop_out + names['stop_out_' + str(i)]
                                    break
                elif names['data_out_x_' + str(i)].shape[0] <= 10:
                    go_on_gen = 0
                    for idx in range(names['data_out_x_' + str(i)].shape[0]):
                        dis = (np.sum(np.sqrt(np.sum((names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2, axis=1)),axis=0))/names['data_out_x_' + str(i)].shape[0]
                        if dis <= names['dis_out_' + str(i)]:
                            go_on_gen = 1
                            if names['generated_data_out_all_' + str(i)] == []:
                                names['generated_data_out_all_' + str(i)] = names['generated_data_out_' + str(i)][idx,].reshape(1,latent_size)
                            else:
                                names['generated_data_out_all_' + str(i)] =np.concatenate((names['generated_data_out_all_' + str(i)],names['generated_data_out_' + str(i)][idx,].reshape(1,latent_size)), axis=0)
                            if names['generated_data_out_all_' + str(i)].shape[0] >= names['data_out_x_' + str(i)].shape[0]*mul:
                                names['stop_out_' + str(i)] = 1
                                stop_out = stop_out + names['stop_out_' + str(i)]
                                break
                    if go_on_gen == 1:
                        noise = np.random.uniform(0, 1, (mul * names['data_out_x_' + str(i)].shape[0], latent_size))
                        names['generated_data_out_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                        for idx in range(names['data_out_x_' + str(i)].shape[0]*mul):
                            dis = (np.sum(np.sqrt(np.sum((names['generated_data_out_' + str(i)][idx,] - names['data_out_x_' + str(i)]) ** 2, axis=1)),axis=0))/names['data_out_x_' + str(i)].shape[0]
                            if dis <= names['dis_out_' + str(i)]:
                                names['generated_data_out_all_' + str(i)] = np.concatenate((names['generated_data_out_all_' + str(i)], names['generated_data_out_' + str(i)][idx,].reshape(1,latent_size)), axis=0)
                                if names['generated_data_out_all_' + str(i)].shape[0] >= names['data_out_x_' + str(i)].shape[0]*mul:
                                    names['stop_out_' + str(i)] = 1
                                    stop_out = stop_out + names['stop_out_' + str(i)]
                                    break
                else:
                    names['generated_data_out_' + str(i)] = pd.DataFrame(names['generated_data_out_' + str(i)])
                    sample_num = min(20, names['data_out_x_' + str(i)].shape[0])
                    names['eva_nash_data_out_' + str(i)] = names['data_out_x_' + str(i)].sample(sample_num, replace=False, random_state=None, axis=0)
                    names['eva_nash_data_out_' + str(i)] = names['eva_nash_data_out_' + str(i)].as_matrix()
                    names['Nnr_MGAOS_' + str(i)] = 0
                    for idx in range(sample_num):
                        real = 0
                        dis = np.sqrt(np.sum((names['eva_nash_data_out_' + str(i)][idx,] - names['x_out_' + str(i)]) ** 2, axis=1))
                        dis = pd.DataFrame(dis)
                        names['y_out_' + str(i)] = pd.DataFrame(names['y_out_' + str(i)])
                        dis = np.concatenate((dis, names['y_out_' + str(i)]), axis=1)
                        dis = pd.DataFrame(dis, columns=['d', 'y'])
                        dis = dis.sort_values('d', ascending=True)
                        dis = dis.as_matrix()
                        for index in range(sample_num):
                            if dis[index, 1] == 0:
                                real = real + 1
                        nnr = real / sample_num
                        if nnr > args.nnr_MGAOS:
                            names['Nnr_MGAOS_' + str(i)] = names['Nnr_MGAOS_' + str(i)] + 1
                    names['Nnr_MGAOS_' + str(i)] = names['Nnr_MGAOS_' + str(i)] / sample_num
                    if names['Nnr_MGAOS_' + str(i)] > args.nnr_MGAOS:
                        names['stop_out_' + str(i)] = 1
                        stop_out = stop_out + 1
        if stop_out == k_out:
            break

    # Augment the minority class
    new_data_out_size = 0
    for i in range(k_out):
        if names['data_out_x_' + str(i)].shape[0]<=10:
            if names['generated_data_out_all_' + str(i)] ==[]:
                names['new_data_out_x_' + str(i)] = names['data_out_x_' + str(i)]
            else:
                names['new_data_out_x_' + str(i)] = np.concatenate((names['generated_data_out_all_' + str(i)], names['data_out_x_' + str(i)]), axis=0)
            if names['new_data_out_x_' + str(i)].shape[0] < names['data_out_x_' + str(i)].shape[0]*(mul+1):
                names['new_data_out_x_' + str(i)] = pd.DataFrame(names['new_data_out_x_' + str(i)])
                names['new_data_out_x_' + str(i)] = names['new_data_out_x_' + str(i)].sample(n=math.ceil(names['data_out_x_' + str(i)].shape[0] * (mul+1)), replace=True, random_state=None, axis=0)
        else:
            if names['stop_out_' + str(i)] == 1:
                noise = np.random.uniform(0, 1, (int(names['data_out_x_' + str(i)].shape[0]*mul), latent_size))
                names['generated_data_out_all_' + str(i)] = names['generator_out_' + str(i)].predict(noise, verbose=0)
                names['new_data_out_x_' + str(i)] = np.concatenate((names['generated_data_out_all_' + str(i)], names['data_out_x_' + str(i)]), axis=0)
            else:
                names['data_out_x_' + str(i)] = pd.DataFrame(names['data_out_x_' + str(i)])
                names['new_data_out_x_' + str(i)] = names['data_out_x_' + str(i)].sample(n=math.ceil(names['data_out_x_' + str(i)].shape[0] * (mul + 1)), replace=True, random_state=None,axis=0)
        new_data_out_size = new_data_out_size + names['new_data_out_x_' + str(i)].shape[0]

    # Start iteration
    for i in range(k_unl):
        names['stop_unl_' + str(i)] = 0
        if names['data_unl_x_' + str(i)].shape[0] == 1:
            names['change_' + str(i)] = 0
            dists_unl = np.sqrt(np.sum((names['data_unl_x_' + str(i)] - data_x) ** 2, axis=1))
            index = np.argsort(dists_unl)
            names['dis_unl_' + str(i)] = dists_unl[index[4]]

    for epoch in range(args.max_iter_MGAAL):
        print('Epoch {} of {}'.format(epoch + 1, args.max_iter_MGAAL))

        # Sample mini-batch date
        for i in range(k_out):
            names['new_data_out_x_' + str(i)] = pd.DataFrame(names['new_data_out_x_' + str(i)])
            names['data_out_batch_x_' + str(i)] = names['new_data_out_x_' + str(i)].sample(n=math.ceil(names['new_data_out_x_' + str(i)].shape[0] * (batch_size/2) / new_data_out_size), replace=False, random_state=None, axis=0)
        for i in range(k_unl):
            names['data_unl_x_' + str(i)] = pd.DataFrame(names['data_unl_x_' + str(i)])
            names['data_unl_batch_x_' + str(i)] = names['data_unl_x_' + str(i)].sample(n=math.ceil((names['data_unl_x_' + str(i)].shape[0] * (batch_size/2)) / data_unl_size), replace=False, random_state=None, axis=0)

        # Train sub-generators and sub-discriminators
        for i in range(k_unl):
            names['data_unl_batch_x_' + str(i)] = pd.DataFrame(names['data_unl_batch_x_' + str(i)])
            if names['stop_unl_' + str(i)] == 0:
                noise = np.random.uniform(0, 1, (int(names['data_unl_batch_x_' + str(i)].shape[0]), latent_size))
                names['generated_data_unl_' + str(i)] = names['generator_unl_' + str(i)].predict(noise, verbose=0)
                names['x_unl_' + str(i)] = np.concatenate((names['data_unl_batch_x_' + str(i)], names['generated_data_unl_' + str(i)]), axis=0)
                names['y_unl_' + str(i)] = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])) + [0] * (int(names['data_unl_batch_x_' + str(i)].shape[0])))
                names['discriminator_unl' + str(i)] = names['discriminator_unl_' + str(i)].train_on_batch(names['x_unl_' + str(i)], names['y_unl_' + str(i)])
                trick_unl = np.array([1] * (int(names['data_unl_batch_x_' + str(i)].shape[0])))
                names['generator_unl' + str(i)] = names['combine_model_unl_' + str(i)].train_on_batch(noise, trick_unl)

                # The evaluation of sub-GANs
                names['generated_data_unl_' + str(i)] = pd.DataFrame(names['generated_data_unl_' + str(i)])
                sample_num = min(20,names['data_unl_batch_x_' + str(i)].shape[0])
                names['eva_nash_data_unl_' + str(i)] = names['data_unl_batch_x_' + str(i)].sample(sample_num, replace=False, random_state=None, axis=0)
                names['eva_nash_data_unl_' + str(i)] = names['eva_nash_data_unl_' + str(i)].as_matrix()
                names['Nnr_unl_' + str(i)] = 0
                if sample_num >= 2:
                    for idx in range(sample_num):
                        real = 0
                        dis = np.sqrt(np.sum((names['eva_nash_data_unl_' + str(i)][idx,] - names['x_unl_' + str(i)]) ** 2,axis=1))
                        dis = pd.DataFrame(dis)
                        names['y_unl_' + str(i)] = pd.DataFrame(names['y_unl_' + str(i)])
                        dis = np.concatenate((dis, names['y_unl_' + str(i)]), axis=1)
                        dis = pd.DataFrame(dis, columns=['d', 'y'])
                        dis = dis.sort_values('d', ascending=True)
                        dis = dis.as_matrix()
                        for index in range(sample_num):
                            if dis[index, 1] == 0:
                                real = real + 1
                        nnr = real / sample_num
                        if nnr > args.nnr_MGAAL:
                            names['Nnr_unl_' + str(i)] = names['Nnr_unl_' + str(i)] + 1
                    names['Nnr_unl_' + str(i)] = names['Nnr_unl_' + str(i)] / sample_num
                    if names['Nnr_unl_' + str(i)] > args.nnr_MGAAL:
                        names['stop_unl_' + str(i)] = 1
                    print("The {}th subset contains {} samples, the evaluation of the sub-GAN is {}".format(i, sample_num, names['Nnr_unl_' + str(i)]))
                else:
                    dis = np.linalg.norm(names['eva_nash_data_unl_' + str(i)] - names['generated_data_unl_' + str(i)])
                    if dis <= names['dis_unl_' + str(i)]:
                        names['change_' + str(i)] = names['change_' + str(i)]+1
                        if names['change_' + str(i)] > 5:
                            names['stop_unl_' + str(i)] = 1
                    print("The {}th subset contains {} samples, the evaluation of the sub-GAN is {}".format(i, sample_num, names['change_' + str(i)]))

        # Train the detector
        for i in range(k_out):
            if i==0:
                data_out_batch = names['data_out_batch_x_' + str(i)]
            else:
                data_out_batch = np.concatenate((data_out_batch, names['data_out_batch_x_' + str(i)]), axis=0)
        for i in range(k_unl):
            if i==0:
                data_unl_batch = names['data_unl_batch_x_' + str(i)]
            else:
                data_unl_batch = np.concatenate((data_unl_batch, names['data_unl_batch_x_' + str(i)]), axis=0)
        for i in range(k_unl):
            noise_all = np.random.uniform(0, 1, (math.ceil(batch_size / (2*k_unl)), latent_size))
            names['generated_data_unl_all_' + str(i)] = names['generator_unl_' + str(i)].predict(noise_all, verbose=0)
            if i==0:
                x_unl_all = np.concatenate((data_unl_batch, names['generated_data_unl_all_' + str(i)]), axis=0)
            else:
                x_unl_all = np.concatenate((x_unl_all, names['generated_data_unl_all_' + str(i)]), axis=0)

        x_all = np.concatenate((x_unl_all, data_out_batch), axis=0)
        y_all = np.array([1] * (int(data_unl_batch.shape[0])) + [0] * (x_all.shape[0] - data_unl_batch.shape[0]))
        discriminator_all.train_on_batch(x_all, y_all)

        # The selection of optimal model(auc)
        p_value = discriminator_all.predict(data_x)
        p_value = 1 - p_value
        p_value = pd.DataFrame(p_value)
        data_y_ide = np.array([1] * (int(data_out_x.shape[0])) + [0] * (data_unl_x.shape[0]))
        data_y_ide = pd.DataFrame(data_y_ide)
        auc = roc_auc_score(data_y_ide, p_value)
        print('The performance evaluation(auc) of the detector{}'.format(auc))

        if auc_d <= auc:
            auc_d = auc
            discriminator_all.save('{path}'.format(path=args.save_model))


    discriminator_test = load_model('{path}'.format(path=args.save_model))
    p_value = 1 - discriminator_test.predict(test_x)
    p_value = [i for j in p_value for i in j]
    p_valuen = p_value[:]
    p_value2n = p_value[:]
    p_value_list = np.argsort(p_value)
    p_value_n = p_value[p_value_list[-top_n]]
    p_value_2n = p_value[p_value_list[-2 * top_n]]
    for idx1 in range(len(p_value)):
        if p_valuen[idx1] >= p_value_n:
            p_valuen[idx1] = 1
        else:
            p_valuen[idx1] = 0
    for idx2 in range(len(p_value)):
        if p_value2n[idx2] >= p_value_2n:
            p_value2n[idx2] = 1
        else:
            p_value2n[idx2] = 0
    p_value = pd.DataFrame(p_value)
    p_valuen = pd.DataFrame(p_valuen)
    test_y = pd.DataFrame(test_y)
    auc = roc_auc_score(test_y, p_value)
    ap = average_precision_score(test_y, p_value)
    p_valuen = pd.DataFrame(p_valuen)
    precision_n = precision_score(test_y, p_valuen)
    recall_n = recall_score(test_y, p_valuen)
    f1score_n = f1_score(test_y, p_valuen)
    print('AUC_final:{}'.format(auc))
    print('AP_final:{}'.format(ap))
    print('precision@n_final:{}'.format(precision_n))
    print('recall@n_final:{}'.format(recall_n))
    print('f1@n_final:{}'.format(f1score_n))