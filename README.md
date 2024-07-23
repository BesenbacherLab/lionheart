# LIONHEART Cancer Detector

Run the feature extraction and learned model from [paper ref] on your own data.

Developed for hg38.

The code was developed and implemented by [@ludvigolsen](https://github.com/LudvigOlsen). Concept and method 

## Installation

Install the main package:

```
# Download repository
# TODO: Public version should not use `gh`
# but this might be necessary for private repo (or download manually)
$ gh repo clone LudvigOlsen/lionheart

# Create and activate conda environment
$ conda env create -f lionheart/environment.yml
$ conda activate lionheart

# Install package
$ pip install lionheart

```

### Custom mosdepth 

We use a modified version of `mosdepth` available at https://github.com/LudvigOlsen/mosdepth/

To install this, it requires an installation of `nim` so we can use `nimble install`. Note that we use `nim 1.6.14`.

```
# Download nim installer and run
$ curl https://nim-lang.org/choosenim/init.sh -sSf | sh

# Add to ~/.bashrc
export PATH=/home/<username>/.nimble/bin:$PATH

$ source ~/.bashrc

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
$ wget https://zenodo.org/records/11067054/files/inference_resources__v001.tar.gz
$ tar -xvzf inference_resources__v001.tar.gz 
```

NOTE: Current version has an old model, so only the feature extraction works.

## Examples

### Via `gwf` workflow

We provide a simple workflow for submitting jobs to slurm via the `gwf` package. Make a copy of the `workflow` directory (to a different location), open `workflow.py`, change the paths and list the samples to run for.

The first time running a workflow it's required to first set the `gwf` backend to slurm or one of the other ![backends](https://gwf.app/reference/backends/):

```
# Navigate to your copy of the the workflow directory
$ cd workflow/

# Activate conda environment
$ conda activate lionheart

# Set `gwf` backend to slurm (or another preferred backend)
$ gwf config set backend slurm
```

When you're ready to submit the jobs, run:

```
$ gwf run
```

With many samples, this can take a while. `gwf` allows seeing a status of the submitted jobs as well

### CLI

```
# Start by skimming the help page
$ lionheart -h

# Or read the usage guide
$ lionheart guide_me

# `mosdepth_path` is the path to the customized `mosdepth` installation
# E.g. "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g. "/home/<username>/anaconda3/envs/lionheart/lib/"
$ lionheart extract_features --bam_file {bam_file} --resources_dir {resources_dir} --out_dir {out_dir} --mosdepth_path {mosdepth_path} --ld_library_path {ld_library_path} --n_jobs {cores}

# `sample_dir` is the out_dir of `extract_features`
$ lionheart predict_sample --sample_dir {sample_dir} --resources_dir {resources_dir} --out_dir {out_dir} --thresholds max_j spec_0.95 spec_0.99 sens_0.95 sens_0.99 0.5 --identifier {sample_id}

```
