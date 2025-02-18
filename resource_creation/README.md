# LIONHEART Resource Generation (For Reproducibility)

This directory contains scripts for reproducing the main resources used in LIONHEART.

**NOTE**: These scripts are NOT necessary for the use of LIONHEART.

## Additional package dependencies

Running these scripts require a few more dependencies. Since installing these will alter the versions of some of the other dependencies, you may prefer to make a separate conda environment for running them. 

 - `lionheart` itself must be installed (`pip install lionheart`)
 - `bedtools` (`conda install bioconda::bedtools`)
 - `py2bit` (`conda install bioconda::py2bit`)


## Required files

TODO: Finish this section!!

| Argument                | Description                                                                                     | Source                                                                                                                                                                        |
| :---------------------- | :---------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bam_file`            | Small BAM file (>=1 fragment per autosome). Keep it small as only the bin coordinates are used. | We used the example BAM file from (https://zenodo.org/records/13909979). Downsampled hg38 version of a public BAM file from Snyder et al. (2016; 10.1016/j.cell.2015.11.050). |
| `--chrom_sizes_file`    | DESCRIPTION                                                                                     | `hg38.chrom.sizes` from                                                                                                                                                       |
| `--exclusion_bed_files` | DESCRIPTION                                                                                     | `k100.umap.exclusion_intervals.bed` from () and `hg38-blacklist.v2.bed` from ()                                                                                               |
| `--reference_file`      | DESCRIPTION                                                                                     | `hg38.2bit` from ()                                                                                                                                                           |


## Examples


### Run via command-line interface


```
# Use a small BAM file (>=1 fragment per autosome)
# `mosdepth_path` is the path to the customized `mosdepth` installation
# E.g., "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g., "/home/<username>/anaconda3/envs/lionheart/lib/"

$ python bin_genome.py --bam_file {bam_file} --out_dir  {out_dir} --chrom_sizes_file {hg38.chrom.sizes} --exclusion_bed_files {k100.umap.exclusion_intervals.bed} {hg38-blacklist.v2.bed} --reference_file {hg38.2bit} --mosdepth_path {mosdepth_path} --ld_library_path {ld_library_path} --bin_size 10 --gc_bin_size 100 --num_gc_bins 100

$ python create_cell_type_masks.py ...
```


### Via `gwf` workflow

TODO: Make this