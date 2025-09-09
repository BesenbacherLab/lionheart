# Changelog

## 2.0.2

 - Replaces the use of "truncate" / "truncation" with "clip" / "clipping". The functionality stays the same, but this is more precise terminology. By clipping, we mean "replacing a value beyond a threshold with the value at that threshold".
 - Now requires `generalize` v0.3.0 or above.
 - Fixes unit/regression tests for v2.

## 2.0.0

**NOTE!** This update contains **major changes**. To use it, please reinstall the conda environment, the lionheart package, and the custom mosdepth version, and download the newest resources from zenodo (https://zenodo.org/records/15747531).

### Highlights

 - The number of cell types was increased from 408 unique (487 cell type features) to 700 unique (898 cell type features).
   - Note: The previous features are not compatible with the new model.
 - Features from an additional dataset from Budhraja et al. 2023 was used for training the shared model.
 - Adds `resource_creation` directory with scripts and workflows for creating the resources for LIONHEART. Most users will never need to use these but they are added for transparency and reproducibility and to enable users to extend LIONHEART with their own cell types, etc.
 - Significant decreases in the RAM and time usage of the feature extraction tool. Across our cohorts, the max. memory peak usage was ~20gb but many samples used less.
 

### `lionheart extract_features`

 - For the two mosdepth calls, we see large reductions in RAM (from ~50gb to < 6gb) and time usage. The two calls are now run in parallel saving even more time. We recommend specifying at least 10 cores (`--n_jobs 10`).
   - Note: The feature extraction still requires more RAM than this. We recommend ~25gb.
 - Adds check of chromosome names in BAM file header. Requires the "chrXX" naming convention and the presence of all autosomes (chr1-chr22).
 - Optimizations of internal `normalize_megabins()`.
 - The coverage statistics in `coverage_stats.json` are now calculated from the raw coverage values prior to corrections.
 - Handles rounding errors in loaded arrays by rounding to array-wise meaningful decimal points.


### `lionheart cross_validate`

 - Adds `--merge_datasets` for specifying datasets that should be considered as one in the leave-one-dataset-out cross-validation.
 - Adds `--reps` argument for specifying number of repetitions.
 - Adds `LH_ADVANCED` environment variable for **advanced options**
   - Adds `--feature_type` for running on the benchmark features.
   - Adds `--feature_categories` for specifying cell types categories to include / exclude from the model after scaling.
   - Adds `--loco` for running leave-one-cancer-type-out cross-validation.
   - Adds `--loco_train_only_classes` for specifying train-only classes in `--loco` mode.
   - When `k_inner` is not `None` or it's in `--loco` mode, refitting uses `generalize::make_simplest_model_refit_strategy()` to find the simplest model wrt. the LASSO C and PCA explained variance hyperparameters.


### `lionheart train_model`

 - Adds `--merge_datasets` for specifying datasets that should be considered as one in the leave-one-dataset-out cross-validation during hyperparameter tuning.


### Dependencies

Please reinstall the conda environment and the custom mosdepth version and redownload the resources.

 - Version bump to `joblib==1.4.2`.
 - Adds `pigz` as dependency for faster (de)compression.
 - Adds `gawk` and `mawk` as dependencies.
 - Adds `pyarrow` as dependency.
 - Adds `samtools` as dependency to allow check of BAM files before running.


### Minor changes

 - Disables matplotlib font manager logger where relevant. This reduces irrelevant logging messages.
 - Adds `lionheart --version` command to CLI.


## 1.1.5

 - Fixes package specification in pyproject.toml

**Future note**: An *upcoming* version will contain completely recomputed resource files with changed bin-coordinates to reduce RAM usage of the `mosdepth` coverage extraction. At the same time, we will be updating the exclusion bin index files to fix a small discrepency between the shared features and the features extracted with the current `lionheart` version. Stay tuned for updates in the coming month(s).


## 1.1.4

 - Adds project URLs to package to list them on the `pypi` site.


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