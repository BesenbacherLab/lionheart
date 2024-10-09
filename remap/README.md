# Remap to hg38

In case you have hg19 data, we include the settings we used to map to hg38 from bam/fastq files.

## Installation

Create the following environment (NOTE: you can replace `mamba` with `conda` if you need!)
```
$ mamba create --name remap python=3.9
$ mamba activate remap

# https://github.com/micknudsen/guessadapt
mamba install -y -c micknudsen guessadapt

# https://gwf.app
$ mamba install -y -c gwforg gwf

$ mamba install -y -c bioconda pysam cutadapt seqtk picard samtools

pip install utipy
```

**NOTE**: that it's best to to have a separate environment for the remapping, as some dependencies for the cancer detection (`htslib=1.15.1`) seem to not work with `picard`.

## Reference files

If you don't already have them, download the following files to a directory of your choosing:

```
$ wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
$ wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai
$ wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bwa_index.tar.gz
```

Unzip with:

```
$ gunzip GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
$ tar -xvzf GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.bwa_index.tar.gz 
```

## Run remapping 

Move to this directory (with this readme and workflow.py file).
```
# Set `gwf` backend to slurm (or another preferred backend)
$ gwf config set backend slurm 
```

Now open the `workflow.py` file and add the various file paths. It's python so you can create a function for creating the the `bam_files` dict (sample_id -> bam_path), read it from a csv, or similar.

Once the paths are set, while still standing in this directory, run:

```
$ mamba activate remap
$ gwf run
```

Get a summary of the job statuses with:
```
gwf status -f summary
```

Or read logfiles (<target_name>.stderr or <target_name>.stdout) from the `.gwf/logs/` folder. E.g.

```
less .gwf/logs/<target_name>.stderr
```

