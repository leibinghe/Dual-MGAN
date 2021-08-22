# Dual-MGAN: An Efficient Approach for Semi-supervised Outlier Detection with Few Identified Anomalies

## Environment
- Python 3.5- Tensorflow (version: 1.0.1)- Keras (version: 2.0.2)

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the parse_args function).

Run Dual-MGAN:
```
python Dual-MGAN.py --path_out Data/out10.csv --path_unl Data/unl10.csv --path_test Data/test.csv --lr_d 0.001
```


## More Details:
Use `python RCC-Dual-GAN.py -h` to get more argument setting details.

```shell
-h, --help	show this help message and exit--path_out	Input the path of the identified anomalies
--path_unl 	Input the path of the unlabeled data
--path_test 	Input the path of the test data
--lr_sg		Learning rate of sub_generators
--lr_sd		Learning rate of sub_discriminators
--lr_d		Learning rate of the detector
--k_means	The k in k-means
--max_iter_MGAOS	Stop training sub_generators in MGAOS after max_iter_MGAOS
--max_iter_MGAAL	Stop training sub_generators in MGAAL after max_iter_MGAAL
--nnr_MGAOS		The thresholds of Nnr in MGAOS
--nnr_MGAAL		The thresholds of Nnr in MGAAL
```



