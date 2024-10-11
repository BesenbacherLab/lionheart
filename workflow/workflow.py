"""
See the README.md file.
"""

import pathlib
from gwf import Workflow
from target_creators import extract_features, predict_sample, collect_samples

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

resources_dir = "path/to/resources/directory/"
samples = {
    "<sample_id__1>": "path/to/bam/file",
    "<sample_id__2>": "path/to/bam/file",
}
out_dir = "path/to/output/directory/"

# Paths to mosdepth and conda env lib
# Likely something like the following (note: anaconda3 could be miniforge3 or similar)
mosdepth_path = "/home/<username>/mosdepth/mosdepth"
ld_library_path = "/home/<username>/anaconda3/envs/lionheart/lib/"

# Available CPU cores
num_cores = 10

# Whether to collect features and predictions across samples
collect = True

##################
# Create targets #
##################

# Convert paths to `pathlib.Path` objects
resources_dir = pathlib.Path(resources_dir)
assert resources_dir.exists()

out_dir = pathlib.Path(out_dir)
# Does not need to exist already

# Initialize list to save sample directories
sample_dirs = []

# Extract features and predict the cancer status per sample
for sample_id, bam_file in samples.items():
    sample_dir = out_dir / sample_id
    sample_dirs.append(sample_dir)

    extract_features(
        gwf=gwf,
        sample_id=sample_id,
        bam_file=bam_file,
        resources_dir=resources_dir,
        out_dir=sample_dir,
        mosdepth_path=mosdepth_path,
        ld_library_path=ld_library_path,
        cores=num_cores,
    )

    predict_sample(
        gwf=gwf,
        sample_id=sample_id,
        sample_dir=sample_dir,
        resources_dir=resources_dir,
        out_dir=sample_dir,
    )

# Collect features and predictions for all samples
if collect:
    collect_samples(
        gwf=gwf,
        sample_dirs=sample_dirs,
        prediction_dirs=sample_dirs,
        out_dir=out_dir / "collected",
    )
