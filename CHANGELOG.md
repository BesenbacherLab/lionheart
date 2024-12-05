# Change log

## 1.1.xx

This release adds multiple CLI commands that 

1) allow reproducing results from the article and seeing the effect of adding your own datasets:

 - Adds `lionheart cross_validate` command. Perform nested leave-one-dataset-out cross-validation on your own dataset(s) and/or the included features.
 - Adds `lionheart evaluate_univariates` command. Evaluate each feature (cell-type) separately on your own dataset(s) and/or the included features.
 
2) expands what you can do with your own data:

 - Adds `lionheart extract_roc` command. Calculate the ROC curve (for deciding probability thresholds) on your own data and/or the included features. Can be performed for either a custom model or an included model. Allows using probability thresholds suited to your own data when using `lionheart predict_sample`.
 
Also:

 - Adds `matplotlib` as dependency.

## 1.0.2

 - Fixes bug when training model on a single dataset.
 - Adds tests for a subset of the CLI tools.

## 1.0.1

 - Fixed model name.