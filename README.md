# LIONHEART Cancer Detector <a href='https://github.com/besenbacherlab/lionheart'><img src='https://raw.githubusercontent.com/besenbacherlab/lionheart/main/lionheart_242x280_250dpi.png' align="right" height="160" /></a>

LIONHEART is a method for detecting cancer from whole genome sequenced plasma cell-free DNA.

This software lets you run feature extraction and predict the cancer status of your samples. Further, you can train a model on your own data. 

Developed for hg38. See the `remap` directory for the applied remapping pipeline.

The code was developed and implemented by [@ludvigolsen](https://github.com/LudvigOlsen).

## Installation

Install the main package:

```
# Create and activate conda environment
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
# Tip: Consider adding it to the terminal configuration file (e.g. ~/.bashrc)
$ export PATH=/home/<username>/.nimble/bin:$PATH

# Install and use nim 1.6.4 
# NOTE: This step should be done even when nim is already installed
$ choosenim 1.6.14
```

Now that nim is installed, we can install the custom mosdepth with:

```
# Install modified mosdepth
$ nimble install -y https://github.com/LudvigOlsen/mosdepth
```

## Get Resources

Download and unzip the required resources.
```
$ wget https://zenodo.org/records/14215762/files/inference_resources_v002.tar.gz
$ tar -xvzf inference_resources_v002.tar.gz 
```

NOTE: Current version has an old model, so only the feature extraction works.

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
# E.g. "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g. "/home/<username>/anaconda3/envs/lionheart/lib/"
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
