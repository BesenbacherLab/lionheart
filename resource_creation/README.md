# LIONHEART Resource Generation (For Reproducibility)

This directory contains scripts for reproducing the main resources used in LIONHEART.

**NOTE**: These scripts are NOT necessary for the use of LIONHEART.

## Additional package dependencies

Running these scripts require a few more dependencies. Since installing these will alter the versions of some of the other dependencies, you may prefer to make a separate conda environment for running them. 

 - `lionheart` itself must be installed (`pip install lionheart`)
 - `bedtools` (`conda install bioconda::bedtools`)
 - `py2bit` (`conda install bioconda::py2bit`)
 - `pyarrow` (`pip install pyarrow`)


## Required files

| Argument                          | Description                                                                                     | Source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| :-------------------------------- | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bam_file`                      | Small BAM file (>=1 fragment per autosome). Keep it small as only the bin coordinates are used. | We used the [example BAM](https://zenodo.org/records/13909979)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `--chrom_sizes_file`              | Chromosome sizes.                                                                               | [`hg38.chrom.sizes`](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chromsizes)                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `--exclusion_bed_files`           | Mappability blacklists.                                                                         | the *complementary* of [`k100.umap.bed`](https://zenodo.org/records/800645/files/hg38.umap.tar.gz) and [`hg38-blacklist.v2.bed`](https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/lists/hg38-blacklist.v2.bed.gz)                                                                                                                                                                                                                                                                                                                                  |
| `--reference_file`                | DESCRIPTION                                                                                     | [`hg38.2bit`](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg382bit)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `--tracks_dir`                    | ATAC- and DNase-seq files (may require LiftOver to hg38).                                       | ENCODE [DNase](https://www.encodeproject.org/search/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=DNase-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assembly=GRCh38&files.file_type=bed+narrowPeak), ENCODE [ATAC](https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&perturbed=false&assembly=GRCh38), [ATACdb](http://www.licpathway.net/ATACdb/), and [TCGA](https://gdc.cancer.gov/about- |
| 24 data/publications/ATACseq-AWG) |


The ATAC- and DNase-seq tracks (`--tracks_dir`) were manually categorized into features and tissue/organ systems based on the associated meta data files.

## Examples


### Run via command-line interface

These are the *main* scripts that we run. For the full workflow, check the `resource_creation/workflow/workflow.py´ file.

We start by extracting the 10bp bin indices to use and their GC contents in a 100bp context. A few extra files are generated as well. This takes around 18min and <70Gb of RAM on our system.

```bash
# Use a small BAM file (>=1 fragment per autosome)
# `mosdepth_path` is the path to the customized `mosdepth` installation
# E.g., "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g., "/home/<username>/anaconda3/envs/lionheart/lib/"

$ python bin_genome.py --bam_file {bam_file} --out_dir {out_dir} --chrom_sizes_file {hg38.chrom.sizes} --exclusion_bed_files {k100.umap.exclusion_intervals.bed} {hg38-blacklist.v2.bed} --reference_file {hg38.2bit} --mosdepth_path {mosdepth_path} --ld_library_path {ld_library_path} --bin_size 10 --gc_bin_size 100 --num_gc_bins 100
```

We then bin the chromatin accessibility tracks into sparse arrays (separately for DNase and ATAC):

```bash
# num_jobs: Number of available cores - as high as possible!
$ python bin_chromatin_tracks.py --coordinates_file {coordinates_file} --tracks_dir {[DNase / ATAC]_track_dir} --out_dir {out_dir} --meta_data_file {.../chromatin_tracks.meta_data.[DNase / ATAC].tsv} --chrom_sizes_file {hg38.chrom.sizes} --num_jobs {24}
```

For each sample, we run outlier candidate detection:

```bash
$ LD_LIBRARY_PATH={ld_library_path} detect_outlier_candidates.sh {bam_file} {mosdepth_path} 1e-4 {bin_size} {coordinates_file} {bam_out_dir}
```

And then collect the outlier candidates per dataset and then across datasets:

```bash
# Per dataset
# candidate_dirs: Space-separated paths to the per-sample candidates
$ python collect_outliers.py --candidate_dirs {candidate_dirs} --out_dir {out_dir} --thresholds 1e-4 {1 / 263_051_621} --out_ofs 0.25 0.1 --n_chunks {cores}

# Across datasets
# outlier_dirs: Space-separated paths to the per-dataset candidates
$ python collect_outliers_across_datasets.py --outlier_dirs {outlier_dirs} --out_dir {out_dir} --outlier_method union --zero_method intersection
```


### Via `gwf` workflow

We provide a workflow for submitting jobs to slurm via the `gwf` package. Make a copy of the `resource_creation/workflow` directory, open `workflow.py`, change the paths and list the samples to use.

The first time running a workflow it's required to first set the `gwf` backend to slurm or one of the other ![backends](https://gwf.app/reference/backends/):

```
# Start by downloading the repository
$ wget -O lionheart-main.zip https://github.com/BesenbacherLab/lionheart/archive/refs/heads/main.zip
$ unzip lionheart-main.zip

# Copy workflow directory to a location
$ cp -r lionheart-main/resource_creation/workflow <location>/resource_creation_workflow

# Navigate to your copy of the the workflow directory
$ cd <location>/resource_creation_workflow

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
