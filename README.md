# HyperPredict: Estimating Hyperparameter Effects for Instance-Specific Regularization in Deformable Image Registration.

This is the official Pytorch implementation of "HyperPredict: Estimating Hyperparameter Effects for Instance-Specific Regularization in Deformable Image Registration." 

Authors: Aisha Lawal Shuaibu and Ivor J. A. Simpson.

## Summary
We propose a novel method for evaluating the influence of hyperparameters and subsequently selecting an optimal value for given pair of images based on a flexible selection criteria.

Keywords: Deformable Image Registration, Hyperparameter Selection, Regularization
.
## Prerequisites
- `Python`
- `Pytorch`
- `NumPy`
- `NiBabel`

For Visualization in test files:

- `Matplotlib`
- `Seaborn`
- `Neurite`

## Structure of the data
The OASIS data set has been split to train, validation and test. To replicate the results in the paper (or for inference/further studies) using OASIS dataset.
1. Download the preprocessed OASIS data from [neurite-oasis](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)
2. Unzip it and put it in the folder "data/oasis"
3. To use the same split and image pairs as the paper, run the script `split_data.py` located in directory "data/oasis". This will create 3 sub-directories; data/oasis/training, data/oasis/validation, and data/oasis/testing, containing the train, validation and test dataset. 

Note that, if you wish to use our already ran niftyreg and clapirn registration results to train your own HyperPredict model, then swaping/changing the location of any image file in the created sub-directories may lead to an error. 

If you wish to go ahead with the change, we have provided two scripts `run_clairn.py` and `run_niftyreg.py`. 
1. Split the unzipped data to your preferred choice
2. Save the image files in data/oasis/training, data/oasis/validation, and data/oasis/testing. 
3. Run `run_clairn.py` and `run_niftyreg.py` from the terminal to generate registration results of the image pair for clapirn and niftyreg respectively.

## Inference
To test Hyperpredict<sub>clap</sub> run `test_hyperpredict_clapirn.py` with encoding type argument set to mean_encoding
```
test_hyperpredict_clapirn.py --encoding_type mean_encoding
```

To test Hyperpredict<sub>clap</sub> run `test_hyperpredict_niftyreg.py` with encoding type argument set to mean_encoding
```
test_hyperpredict_niftyreg.py --encoding_type mean_encoding
```
This will by default test with pretrained models in models/checkpoints/..

## Train your own model
#currently working on documentation, updates coming soon!

Append specific arguments to the command, depending on use-case. The list of arguments are:

## Acknowledgment
The pretrained models are obtained from from [cLapIRN](https://github.com/cwmok/Conditional_LapIRN/tree/main) and [Symnet](https://github.com/cwmok/Fast-Symmetric-Diffeomorphic-Image-Registration-with-Convolutional-Neural-Networks/tree/master). We obtain our data from [neurite-oasis](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)


##### Contact 
For any questions or potential collaborations contact any of the following people:

Aisha Lawal Shuaibu (corresponding Author): a.shuaibu@sussex.ac.uk

Ivor J. A. Simpson: i.simpson@sussex.ac.uk


