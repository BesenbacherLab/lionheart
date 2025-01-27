# LIONHEART Cancer Detector <a href='https://github.com/besenbacherlab/lionheart'><img src='https://raw.githubusercontent.com/besenbacherlab/lionheart/main/lionheart_242x280_250dpi.png' align="right" height="160" /></a>

LIONHEART is a method for detecting cancer from whole genome sequenced plasma cell-free DNA.

This software lets you run feature extraction and predict the cancer status of your samples. Further, you can train a model on your own data. 

Developed for hg38. See the `remap` directory for the applied remapping pipeline.

Preprint: https://www.medrxiv.org/content/10.1101/2024.11.26.24317971v1

The code was developed and implemented by [@ludvigolsen](https://github.com/LudvigOlsen).

If you experience an issue, please [report it](https://github.com/BesenbacherLab/lionheart/issues).


## Installation

This section describes the installation of `lionheart` and the custom version of `mosdepth` (exp. time: <10m). The code has only been tested on linux but should also work on Mac and Windows.

Install the main package:

```
# Create and activate conda environment
$ conda config --set channel_priority true
$ conda env create -f https://raw.githubusercontent.com/BesenbacherLab/lionheart/refs/heads/main/environment.yml
$ conda activate lionheart

# Install package from PyPI
$ pip install lionheart

# OR install from GitHub
$ pip install git+https://github.com/BesenbacherLab/lionheart.git

```

### Custom mosdepth 

We use a modified version of `mosdepth` available at https://github.com/LudvigOlsen/mosdepth/

To install this, it requires an installation of `nim` so we can use `nimble install`. Note that we use `nim 1.6.14`.

```
# Download nim installer and run
$ curl https://nim-lang.org/choosenim/init.sh -sSf | sh

# Add to PATH
# Change the path to fit with your system
# Tip: Consider adding it to the terminal configuration file (e.g., ~/.bashrc)
$ export PATH=/home/<username>/.nimble/bin:$PATH

# Install and use nim 1.6.4 
# NOTE: This step should be done even when nim is already installed
$ choosenim 1.6.14
```

Now that nim is installed, we can install the custom mosdepth. To not override an existing mosdepth installation, we install it into a separate directory:

```
# Make a directory for installing the nim packages into
$ mkdir mosdepth_installation

# Install modified mosdepth
$ NIMBLE_DIR=mosdepth_installation nimble install -y https://github.com/LudvigOlsen/mosdepth

# Get path to mosdepth binary to use in the software
$ find mosdepth_installation/pkgs/ -name "mosdepth*"
>> mosdepth_installation/pkgs/mosdepth-0.x.x/mosdepth

```

## Get Resources

Download and unzip the required resources.
```
$ wget https://zenodo.org/records/14215762/files/inference_resources_v002.tar.gz
$ tar -xvzf inference_resources_v002.tar.gz 
```

## Main commands

This section describes the commands in `lionheart` and lists their *main* output files:

| Command                          | Description                                                         | Main Output                                                                         |
| :------------------------------- | :------------------------------------------------------------------ | :---------------------------------------------------------------------------------- |
| `lionheart extract_features`     | Extract features from a BAM file.                                   | `feature_dataset.npy` and correction profiles                                       |
| `lionheart predict_sample`       | Predict cancer status of a sample.                                  | `prediction.csv`                                                                    |
| `lionheart collect`              | Collect predictions and/or features across samples.                 | `predictions.csv`, `feature_dataset.npy`, and correction profiles *for all samples* |
| `lionheart customize_thresholds` | Extract ROC curve and more for using custom probability thresholds. | `ROC_curves.json` and `probability_densities.csv`                                   |
| `lionheart cross_validate`       | Cross-validate the model on new data and/or the included features.  | `evaluation_summary.csv`,  `splits_summary.csv`                                     |
| `lionheart train_model`          | Train a model on your own data and/or the included features.        | `model.joblib` and training data results                                            |
| `lionheart validate`             | Validate a model on a validation dataset.                           | `evaluation_scores.csv` and `predictions.csv`                                       |
| `lionheart evaluate_univariates` | Evaluate the cancer detection potential of each feature separately. | `univariate_evaluations.csv`                                                        |


## Examples

### Run via command-line interface

This example shows how to run lionheart from the command-line.

Note: If you don't have a BAM file at hand, you can download an example BAM file from: https://zenodo.org/records/13909979 
It is a downsampled version of a public BAM file from Snyder et al. (2016; 10.1016/j.cell.2015.11.050) that has been remapped to hg38. On our system, the feature extraction for this sample takes ~1h15m using 12 cores (`n_jobs`).

```
# Start by skimming the help page
$ lionheart -h

# Extract feature from a given BAM file
# `mosdepth_path` is the path to the customized `mosdepth` installation
# E.g., "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g., "/home/<username>/anaconda3/envs/lionheart/lib/"
$ lionheart extract_features --bam_file {bam_file} --resources_dir {resources_dir} --out_dir {out_dir} --mosdepth_path {mosdepth_path} --ld_library_path {ld_library_path} --n_jobs {cores}

# `sample_dir` is the `out_dir` of `extract_features`
$ lionheart predict_sample --sample_dir {sample_dir} --resources_dir {resources_dir} --out_dir {out_dir} --thresholds max_j spec_0.95 spec_0.99 sens_0.95 sens_0.99 0.5 --identifier {sample_id}
```

After running these commands for a set of samples, you can use `lionheart collect` to collect features and predictions across the samples. You can then use `lionheart train_model` to train a model on your own data (and optionally the included features).


### Via `gwf` workflow

We provide a simple workflow for submitting jobs to slurm via the `gwf` package. Make a copy of the `workflow` directory, open `workflow.py`, change the paths and list the samples to run `lionheart` on.

The first time running a workflow it's required to first set the `gwf` backend to slurm or one of the other ![backends](https://gwf.app/reference/backends/):

```
# Start by downloading the repository
$ wget -O lionheart-main.zip https://github.com/BesenbacherLab/lionheart/archive/refs/heads/main.zip
$ unzip lionheart-main.zip

# Copy workflow directory to a location
$ cp -r lionheart-main/workflow <location>/workflow

# Navigate to your copy of the the workflow directory
$ cd <location>/workflow

# Activate conda environment
$ conda activate lionheart

# Set `gwf` backend to slurm (or another preferred backend)
$ gwf config set backend slurm
```

Open the `workflow.py` file and change the various paths. When you're ready to submit the jobs, run:

```
$ gwf run
```

`gwf` allows seeing a status of the submitted jobs:

```
$ gwf status
$ gwf status -f summary
```

### Reproduction of results

This section shows how to reproduce the main results (cross-validation and external validation) from the paper. It uses the included features so the reproduction can be run without access to the raw sequencing data.

Note that different compilations of scikit-learn on different operating systems may lead to slightly different results. On linux, the results should match the reported results.

#### Cross-validation analysis

We start by performing the nested leave-one-dataset-out cross-validation analysis from Figure 3A (not including the benchmarks).

Note that the default settings are the ones used in the paper.

```
# Perform the cross-validation
# {cv_out_dir} should specify where you want the output files
$ lionheart cross_validate --out_dir {cv_out_dir} --resources_dir {resources_dir} --use_included_features --num_jobs 10
```

The output directory should now include multiple files. The main results are in `evaluation_summary.csv` and `splits_summary.csv`. Note that the results are given for multiple probability thresholds. The threshold reported in the paper is the "Max. J Threshold". You can extract the relevant lines of the summaries with:

```
$ awk 'NR==1 || /Average/ && /J Threshold/' {cv_out_dir}/evaluation_summary.csv
$ awk 'NR==1 || /Average/ && /J Threshold/' {cv_out_dir}/splits_summary.csv
```

#### External validation analysis

To reproduce the external validation, we first train a model on all the included training datasets and then validate it on the included validation dataset:

```
# Train a model on the included datasets
# {new_model_dir} should specify where you want the model files
$ lionheart train_model --out_dir {new_model_dir} --resources_dir {resources_dir} --use_included_features

# Validate the model on the included validation dataset
# {val_out_dir} should specify where you want the output files
$ lionheart validate --out_dir {val_out_dir} --resources_dir {resources_dir} --model_dir {new_model_dir} --use_included_validation --thresholds 'max_j'
```

The model training creates the `model.joblib` file along with predictions and evaluations from the *training data* (e.g., `predictions.csv`, `evaluation_scores.csv`, and `ROC_curves.json`).

The validation creates `evaluation_scores.csv` and `predictions.csv` from applying the model on the validation dataset. You will find the reported AUC score in `evaluation_scores.csv`:

```
$ cat {val_out_dir}/evaluation_scores.csv
```

#### Univariate analyses

Finally, we reproduce the univariate modeling evaluations in Figure 2D and 2E:

```
# Evaluate the classification potential of each cell type separately
# {univariates_dir} should specify where you want the evaluation files
$ lionheart evaluate_univariates --out_dir {univariates_dir} --resources_dir {resources_dir} --use_included_features --num_jobs 10
```

This creates the `univariate_evaluations.csv` file with evaluation metrics per cell-type. There are coefficients and p-values (bonferroni-corrected) from univariate logistic regression models and evaluation metrics from per-cell-type leave-one-dataset-out cross-validation.