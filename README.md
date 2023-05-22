# Guided Retraining

### In this repository, we host the dataset and the code of our Guided Retraining paper.

## Dataset
We relied on a public dataset that can be found at: https://github.com/Trustworthy-Software/Combination-malware-detectors

We provide the exact lists of sha256 of the apps used in our experiments in the "dataset/data\_splits\_files" folder.
The folder contains text files that include the lists of hashes and the labels for the dataset splits used in our experiments.
We recommend to clone the repository since you might be unable to view some lists of hashes using the GitHub interface due to their big size.
Please use these lists to download the apps from the public reporsitory [AndroZoo](https://androzoo.uni.lu/)




As we mentioned in the paper, we relied on the publicly available [re-implementation](https://github.com/Trustworthy-Software/Reproduction-of-Android-Malware-detection-approaches) of state-of-the-art approaches from their replication study. Please refer to this re-implementation to generate the matrices of each approach for the different splits provided in "dataset/data\_splits\_files" folder.
When the training, validation and test matrices are generated, place them in the "dataset/matrices" folder.

Note that the order of the rows in your matrices should match the order of sha256 in the provided lists. This will ensure that the matrices are compatible with the labels we provide.

## BC training

To train the BC classifiers, use "train_bc.py" script.
Example:

```sh
python train_bc.py --path_save "outputs" --split_id 1
```
The outputs are the predictions and probabilities for the training, valid and test datasets.

## Spliting the datasets into easy and difficut subsets
For DREBIN, RevelDroid and MaMaDroid approaches, use "split\_data\_easy\_difficult.py" script.

For MalScan, use "split\_data\_easy\_difficult\_malscan.py".

The two scripts generate the indices of the samples for the the easy and difficult subsets.
```sh
python split\_data\_easy\_difficult.py --path_files "outputs" --path_save "outputs" --split_id 1 --approach "drebin"
```
The outputs are:

- indices_diff_tr_drebin: The indices of the difficult samples in the training dataset
- indices_diff_va_drebin: The indices of the difficult samples in the validation dataset
- indices_diff_te_drebin: The indices of the difficult samples in the test dataset
- indices_diff_tr_drebin_tp: The indices of the difficult TP in the training dataset
- indices_diff_tr_drebin_tn: The indices of the difficult TN in the training dataset
- indices_diff_tr_drebin_fp: The indices of the difficult FP in the training dataset
- indices_diff_tr_drebin_fn: The indices of the difficult FN in the training dataset

## Guided Retraining
To train Model1, Model2, Model3 and Model4, use "supcon_4P.py" script.

Example:
For Model1:
```sh
python supcon_4P.py --approach drebin --keyword_approach dr --threshold 5 --mem True --path_indices "outputs/indices" --split_id 1 --part 1
```



When the four Models finish training, you can proceed with Model5 training using "supcon_4P_conca.py" script.

Example:
```sh
python supcon_4P_conca.py --approach mama_f --keyword_approach mf --threshold 5 --mem False --path_indices "outputs/indices" --split_id 1
```

For the classification:
```sh
python linear_4P.py --approach malscan_co --keyword_approach mco --threshold 5 --mem True --path_indices "outputs/indices" --split_id 1
```
The Models and the classifier are located in the save directory.


## RBC training
To train the RBC classifiers, use "train_rbc.py" script.


Example:

```sh
python train_rbc.py --path_save "outputs" --path_indices "outputs/indices" --split_id 1
```
The outputs are the predictions and probabilities for the training, valid and test difficult subsets.

## RClassic training on the whole training dataset
To train the RClassic classifier on the whole training dataset, use "supcon_all.py" and "linear_all.py" scripts.
Example:
For the embeddings generation:
```sh
python supcon_all.py --approach mama_p --keyword_approach mp --threshold 5 --mem True --split_id 1
```
For the classification:



```sh
python linear_all.py --approach reveal --keyword_approach rev --threshold 5 --mem True --split_id 1
```
The model and the classifier are located in the save directory.

## RClassic training on the difficult subset
To train the RClassic classifier on the difficult training subset, use "supcon_diff.py" and "linear_diff.py" scripts.
Example:
For the embeddings generation:
```sh
python supcon_diff.py --approach malscan_a --keyword_approach ma --threshold 5 --mem False --path_indices "outputs/indices" --split_id 1
```
For the classification:
```sh
python linear_diff.py --approach drebin --keyword_approach dr --threshold 5 --mem True --path_indices "outputs/indices" --split_id 1
```


The model and the classifier are located in the save directory.
