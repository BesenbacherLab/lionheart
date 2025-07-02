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

TODO: Finish this section!!

| Argument                | Description                                                                                     | Source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :---------------------- | :---------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--bam_file`            | Small BAM file (>=1 fragment per autosome). Keep it small as only the bin coordinates are used. | We used the [example BAM](https://zenodo.org/records/13909979)                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `--chrom_sizes_file`    | Chromosome sizes.                                                                               | [`hg38.chrom.sizes`](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chromsizes)                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `--exclusion_bed_files` | Mappability blacklists.                                                                         | the *complementary* of [`k100.umap.bed`](https://zenodo.org/records/800645/files/hg38.umap.tar.gz) and [`hg38-blacklist.v2.bed`](https://github.com/Boyle-Lab/Blacklist/raw/refs/heads/master/lists/hg38-blacklist.v2.bed.gz)                                                                                                                                                                                                                                                                                    |
| `--reference_file`      | DESCRIPTION                                                                                     | [`hg38.2bit`](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg382bit)                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `--tracks_dir`          | ATAC- and DNase-seq files (may require LiftOver to hg38).                                       | ENCODE [DNase](https://www.encodeproject.org/search/?type=Experiment&control_type!=*&status=released&perturbed=false&assay_title=DNase-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&assembly=GRCh38&files.file_type=bed+narrowPeak) and [ATAC](https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&replicates.library.biosample.donor.organism.scientific_name=Homo+sapiens&perturbed=false&assembly=GRCh38). [ATACdb](http://www.licpathway.net/ATACdb/) |

## Examples


### Run via command-line interface

We start by extracting the 10bp bin indices to use and their GC contents in a 100bp context. A few extra files are generated as well. This takes around 18min and <70Gb of RAM on our system.


```
# Use a small BAM file (>=1 fragment per autosome)
# `mosdepth_path` is the path to the customized `mosdepth` installation
# E.g., "/home/<username>/mosdepth/mosdepth"
# `ld_library_path` is the path to the `lib` folder in the conda environment
# E.g., "/home/<username>/anaconda3/envs/lionheart/lib/"

$ python bin_genome.py --bam_file {bam_file} --out_dir {out_dir} --chrom_sizes_file {hg38.chrom.sizes} --exclusion_bed_files {k100.umap.exclusion_intervals.bed} {hg38-blacklist.v2.bed} --reference_file {hg38.2bit} --mosdepth_path {mosdepth_path} --ld_library_path {ld_library_path} --bin_size 10 --gc_bin_size 100 --num_gc_bins 100

# 
# num_jobs: Number of available cores - as high as possible!
$ python bin_chromatin_tracks.py --coordinates_file {coordinates_file} --tracks_dir {[DNase / ATAC]_track_dir} --out_dir {out_dir} --meta_data_file {.../chromatin_tracks.meta_data.[DNase / ATAC].tsv} --chrom_sizes_file {hg38.chrom.sizes} --num_jobs {24}
```


### Via `gwf` workflow

TODO: Make this