# Change log

## 1.1.3

 - Adds project URLs to package to list them on the `pypi` site.

**Future note**: An *upcoming* version will contain completely recomputed resource files with changed bin-coordinates to reduce RAM usage of the `mosdepth` coverage extraction. At the same time, we will be updating the exclusion bin index files to fix a small discrepency between the shared features and the features extracted with the current `lionheart` version. Stay tuned for updates in the coming month(s).

## 1.1.2

 - Fixes writing of README in `lionheart predict_sample`. Thanks to @LauraAndersen for detecting the problem.
 - Improvements to installation guide in repository README.
 - Workflow example improvements.

## 1.1.1

 - Improves CLI documentation for some commands (in `--help` pages).

## 1.1.0

This release adds multiple CLI commands that:

1) allow reproducing results from the article and seeing the effect of adding your own datasets:

 - Adds `lionheart cross_validate` command. Perform nested leave-one-dataset-out cross-validation on your dataset(s) and/or the included features.
 - Adds `lionheart validate` command. Validate a model on the included external dataset or a custom dataset.
 - Adds `lionheart evaluate_univariates` command. Evaluate each feature (cell-type) separately on your dataset(s) and/or the included features.
 
2) expands what you can do with your own data:

 - Adds `lionheart customize_thresholds` command. Calculate the ROC curve and probability densities (for deciding probability thresholds) on your data and/or the included features for a custom model or an included model. Allows using probability thresholds suited to your own data when using `lionheart predict_sample` and `lionheart validate`.
 - Adds `--custom_threshold_dirs` argument in `lionheart predict_sample`. Allows passing the ROC curves and probability densities extracted with `lionheart customize_thresholds`.
 
Also:

 - Adds `matplotlib` as dependency.
 - Bumps `generalize` dependency requirement to `0.2.1`.
 - Bumps `utipy` dependency requirement to `1.0.3`.

## 1.0.2

 - Fixes bug when training model on a single dataset.
 - Adds tests for a subset of the CLI tools.

## 1.0.1

 - Fixed model name.