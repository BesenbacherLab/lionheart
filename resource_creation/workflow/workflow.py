"""
See the README.md file.
"""

import pathlib
from gwf import Workflow
import utipy as ut

from resource_creation.workflow.target_creators import (
    bin_chromatin_tracks,
    bin_genome,
    collect_and_annotate_features_with_category_info,
    collect_outliers_across_datasets,
    collect_outliers_for_dataset,
    copy_index_map,
    find_outlier_candidates,
)

# Create `gwf` workflow
gwf = Workflow(
    defaults={
        "cores": 1,
        # "account": "",  # Setting a project/account name may help job prioritization
    }
)

##############################
# Specify paths and settings #
##############################

scripts_dir = "path/to/lionheart/resource_creation/"
new_resources_dir = "path/to/new_resources/directory/"
samples = {
    "<sample_id__1>": "path/to/bam/file",
    "<sample_id__2>": "path/to/bam/file",
}


# Paths to mosdepth and conda env lib
# Likely something like the following (note: anaconda3 could be miniforge3 or similar)
mosdepth_path = "/home/<username>/mosdepth/mosdepth"
ld_library_path = "/home/<username>/anaconda3/envs/lionheart/lib/"

# Path to needed files
# We used the BAM file downloaded from: https://zenodo.org/records/13909979
mini_bam = "path/to/bam_file.bam"
reference_file = "path/to/hg38.2bit"
chrom_sizes_file = "path/to/hg38.chrom.sizes"
exclusion_files = [
    "path/to/hg38-blacklist.v2.bed",
    "path/to/k100.umap.exclusion_intervals.bed",
]

tracks_dirs = {"ATAC": "path/to/ATAC_tracks", "DNase": "path/to/DNase_tracks"}

meta_data_files = {
    "ATAC": "path/to/ATAC_meta_data_file.tsv",
    "DNase": "path/to/DNase_meta_data_file.tsv",
}

# Outlier detection uses all BAM files from all available datasets
dataset_to_bam_files = {
    "dataset_1": ["path/to/sample_1.bam", "path/to/sample_2.bam", "..."],
    "dataset_2": ["path/to/sample_1.bam", "path/to/sample_2.bam", "..."],
}

# Available CPU cores
# NOTE: Maximum possible
num_cores = 12

# Whether to collect features and predictions across samples
collect = True

# Coverage bin size
bin_size = 10

##################
# Create targets #
##################

# Convert paths to `pathlib.Path` objects
new_resources_dir = pathlib.Path(new_resources_dir)
ut.mk_dir(new_resources_dir, "New resources directory")

# Bin the genome into 10bp bins and extract correction bin edges
genome_binning_out_files = bin_genome(
    gwf=gwf,
    scripts_dir=scripts_dir,
    bam_file=mini_bam,
    out_dir=new_resources_dir,
    mosdepth_path=mosdepth_path,
    ld_library_path=ld_library_path,
    reference_file=reference_file,
    chrom_sizes_file=chrom_sizes_file,
    exclusion_files=exclusion_files,
    cores=min(4, num_cores),
)

track_memory = 100
all_index_maps = []
for track_type, track_dir in tracks_dirs.items():
    chromatin_out_files = bin_chromatin_tracks(
        gwf=gwf,
        scripts_dir=scripts_dir,
        out_dir=new_resources_dir / "chromatin_masks" / track_type,
        coordinates_file=genome_binning_out_files["coordinates"],
        tracks_dir=track_dir,
        meta_data_file=meta_data_files[track_type],
        chrom_sizes_file=chrom_sizes_file,
        rows_per_chrom_pre_exclusion_file=genome_binning_out_files[
            "rows_per_chrom_pre_exclusion"
        ],
        track_type=track_type,
        memory=f"{track_memory}g",
        cores=num_cores,
    )

    # Copy the `idx --> cell_type` map to outer directory
    all_index_maps.append(
        copy_index_map(
            gwf=gwf,
            chromatin_out_files=chromatin_out_files,
            track_type=track_type,
            new_resources_dir=new_resources_dir,
        )
    )

collect_and_annotate_features_with_category_info(
    gwf=gwf,
    scripts_dir=scripts_dir,
    out_file=new_resources_dir / "feature_names_and_grouping.csv",
    idx_to_cell_type_files=all_index_maps,
    cell_type_to_category_files=list(meta_data_files.values()),
)

#####################
# Outlier Detection #
#####################

dataset_to_dataset_outlier_paths = {}
dataset_to_outlier_candidate_paths = {}
for dataset, bam_files in dataset_to_bam_files.items():
    outliers_dir = new_resources_dir / "outliers"
    dataset_to_outlier_candidate_paths[dataset] = find_outlier_candidates(
        gwf=gwf,
        scripts_dir=scripts_dir,
        out_dir=outliers_dir / "candidates" / dataset,
        dataset_name=dataset,
        bam_files=bam_files,
        mosdepth_path=mosdepth_path,
        ld_library_path=ld_library_path,
        coordinates_file=genome_binning_out_files["coordinates"],
        cores=min(4, num_cores),
    )

    dataset_to_dataset_outlier_paths[dataset] = collect_outliers_for_dataset(
        gwf=gwf,
        scripts_dir=scripts_dir,
        out_dir=outliers_dir / "per_dataset" / dataset,
        dataset_name=dataset,
        candidate_files=dataset_to_outlier_candidate_paths[dataset],
    )

# Collect across datasets
collect_outliers_across_datasets(
    gwf=gwf,
    scripts_dir=scripts_dir,
    out_dir=outliers_dir,
    dataset_to_outlier_paths=dataset_to_dataset_outlier_paths,
)
