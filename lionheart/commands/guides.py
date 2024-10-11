from lionheart.utils.cli_utils import (
    Guide,
    REPO_URL,
    LION_ASCII,
    LIONHEART_ASCII,
    LIONHEART_STRING,
)
from lionheart.utils.global_vars import INCLUDED_MODELS


future_ops = """
5) cross-validate the model with both your data and 
       the included features using leave-one-dataset-out 
       cross-validation to see how adding your dataset 
       to the training affects generalization to the 
       other datasets
"""


def get_usage_guide():
    """
    Create overview/guide/tutorial for the package to show in the CLI.
    """

    guide = Guide()

    guide.add_vertical_space()
    guide.add_description(LION_ASCII)
    guide.add_description(LIONHEART_ASCII)
    guide.add_title("USAGE GUIDE")
    guide.add_description(
        f"""
This guide will lay out the available steps and processes in {LIONHEART_STRING}.

You will: 
    1) extract features from your own data
    2) predict their cancer status
    3) collect features and predictions across samples
    4) validate the included model on your features 
        - this will give you the probability thresholds 
          that fit your data (for future data)
    5) train a new model using both your own features 
       and the included features
    6) validate the new model on the validation dataset 
       from Zhu et al. (2023)
"""
    )
    guide.add_header("Requirements")
    guide.add_description(
        f"""
First, if haven't followed the installation steps in the GitHub 
repository `README` or downloaded the resources folder, head 
over to the GitHub repository and follow the steps: 
{REPO_URL}

Second, you will need a set of BAM files (hg38) with whole-genome 
sequenced plasma cell-free DNA. For steps 4-6 you will need to have 
both some samples <i>with</i> cancer and some samples <i>without</i> cancer.

Depending on the sequencing depth, feature extraction
may take up to an hour per sample (usually less).
Patience may be needed :-)
"""
    )
    guide.add_header("Alternative: `gwf` workflow")
    guide.add_description(
        f"""
For parallelizing the jobs on a cluster, we refer to the provided 
`gwf` workflow (similar to `snakemake` but `Python`-based) 
which contains each of these steps as well: 
{REPO_URL} 
(See the README example on how to run it)
"""
    )

    guide.add_header("1) Extract features")
    guide.add_description("<insert description>")
    guide.add_example(
        code="""--bam_file path/to/subject_1/<file_name>.bam
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/features/
--mosdepth_path /home/<username>/mosdepth/mosdepth
--ld_library_path /home/<username>/anaconda3/envs/<env_name>/lib/
--n_jobs 10""",
        use_prog="lionheart extract_features",
    )

    guide.add_header("2) Predict cancer status")
    guide.add_description("<insert description>")
    guide.add_example(
        code="""--sample_dir path/to/subject_1/features/
--resources_dir path/to/resource/directory
--out_dir path/to/subject_1/predictions""",
        use_prog="lionheart predict_sample",
    )

    guide.add_header("3) Collect across samples")
    guide.add_description("<insert description>")
    guide.add_example(
        code="""--feature_dirs path/to/subject_1/features path/to/subject_2/features
--prediction_dirs path/to/subject_1/predictions path/to/subject_2/predictions
--out_dir path/to/collected/dataset
""",
        use_prog="lionheart collect",
    )

    guide.add_header("4) Validate included model on your features")
    guide.add_description("<insert description> - e.g. create meta_data.csv")
    guide.add_example(
        code=f"""--dataset_paths path/to/collected/dataset/feature_dataset.npy 
--meta_data_paths path/to/collected/dataset/meta_data.csv 
--out_dir path/to/output/directory
--resources_dir path/to/resource/directory
--model_name {INCLUDED_MODELS[0]}
""",
        use_prog="lionheart validate",
    )

    guide.add_header("5) Cross-validate model to see cross-dataset generalization")
    guide.add_description("<insert description>")
    guide.add_example(
        code="""--dataset_paths path/to/collected/dataset/feature_dataset.npy 
--meta_data_paths path/to/collected/dataset/meta_data.csv 
--out_dir path/to/output/directory
--use_included_features
--resources_dir path/to/resource/directory
<...incomplete example...>
""",
        use_prog="lionheart cross_validate",
    )

    guide.add_header("6) Train a model on your features")
    guide.add_description("<insert description> - AND INCLUDED FEATURES")
    guide.add_example(
        code="""--dataset_paths path/to/collected/dataset/feature_dataset.npy
--meta_data_paths path/to/collected/dataset/meta_data.csv
--out_dir path/to/new_model
--use_included_features
--resources_dir path/to/resource/directory""",
        use_prog="lionheart train_model",
    )

    guide.add_header("7) Validate new model on validation dataset")
    guide.add_description("<insert description>")
    guide.add_example(
        code="""--out_dir path/to/model_validation
--resources_dir path/to/resource/directory
--model_dir path/to/new_model
--use_included_validation
""",
        use_prog="lionheart validate",
    )

    guide.add_vertical_space(1)
    guide.add_description("End of USAGE GUIDE.")
    return guide.construct_guide()


def setup_parser(parser):
    parser.set_defaults(func=main)


def main(args):
    print(get_usage_guide())


# class Guide:
#     def __init__(self) -> None:
#         self.elements = []

#     def add_header(self, header: str):
#         self.elements.append(f"<h1>{header}</h1>")
#         self.add_vertical_space()

#     def add_description(self, desc: str):
#         self.elements.append(desc)
#         self.add_vertical_space()

#     def add_example(self, code: str, pre_comment: str = "", use_prog: bool = True):
#         self.elements.append(
#             Examples.format_example(
#                 description=pre_comment, example=code, use_prog=use_prog
#             )
#         )
#         self.add_vertical_space(n=2)

#     def add_vertical_space(self, n=1):
#         for i in range(n):
#             self.elements.append("")
