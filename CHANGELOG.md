# Change log

## 1.1.xx

This release adds multiple CLI commands that allow reproducing results from the article and seeing the effect of adding your own datasets:

 - Adds `lionheart cross_validate` command. Perform nested leave-one-dataset-out cross-validation on your own dataset(s) and or the included features.
 - Adds `lionheart evaluate_univariates` command. Evaluate each feature (cell-type) separately on your own dataset(s) and or the included features.
 - Adds `matplotlib` as dependency.

## 1.0.2

 - Fixes bug when training model on a single dataset.
 - Adds tests for a subset of the CLI tools.

## 1.0.1

 - Fixed model name.