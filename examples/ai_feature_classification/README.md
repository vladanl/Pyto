
# Supervised feature-based classification by machine and deep learning #

Optimization, training and inference for supervised classification of structures visualized by cryo-electron tomography by Logistic regression, SVM and Neural nets. In all cases, input is a set of features of the structures, as opposed to images.

The data provided are features of protein bridges that tether synaptic vesicle to plasma membrane. However, data obtained from other biological systems can be used, provided that the features are appropriately scalled.

Developed for Orozco-Burunda et al, Direct cryo-ET detection of native SNARE and Munc13 protein bridges using AI classification and preprocessing, bioRxiv https://doi.org/10.1101/2024.12.18.629213

## Usage ##

The workflow is contained in the following notebooks:

* supervised_ml.ipynb: Logistic regression and SVM (uses scikit-learn)
* nn_feature_classifier.ipynb: Neural nets (uses Pytorch)

An example of input data (a subset of tether data) is given in input_data/tethers/ .

Output is written in output_data/ .
