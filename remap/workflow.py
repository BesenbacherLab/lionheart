# Workflow for (re)mapping data (e.g., (hg19 BAM or FASTQ) -> hg38 BAM).
# Adapted from Genome-Mapping-0.1.4 by Michael Knudsen @ MOMA.
# Generalized by Ludvig Renbo Olsen @ MOMA.

import gzip
import re
import os
import gwf
import pysam
import pathlib
from collections.abc import Sequence
from utipy import mk_dir
from gwf import AnonymousTarget


######################
# Specify paths etc. #
######################

# Set these (see remap README for file download)
GENOME_FASTA = "<path_to>/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
output_data_dir = "path/to/output/directory"
ACCOUNT = ""  # Project account for slurm prioritization - may not apply to your system
SEQ_CENTER = ""  # Sequencing center for read group
PLATFORM = "ILLUMINA"

# Specify dict of {sample_id -> bam file path} and/or {sample_id -> fastq paths}
# You can write python code to generate these dicts

# "<sample_id_1>": "path/to/x.bam"
bam_files = {}

# Pass the two FASTQ files as a tuple with r1 first, r2 second
# "<sample_id_1>": ("path/to/x_r1.fq.gz", "path/to/x_r2.fq.gz")
fastq_files = {}

# Optional per-sample read-group metadata override.
# Use this for files whose headers do not follow the Illumina format expected below
# (for example some SRA FASTQ or BAM files). Values map:
#   sample_id -> {"instrument": "...", "flowcell": "...", "lane": "..."}
# If one or more fields are genuinely unknown, set them explicitly to "unknown"
# instead of inventing values. These fields are written into the BAM read group
# `ID` and `PU`, so they may matter for provenance or tools that distinguish data
# by read group or platform unit.
# "<sample_id_1>": {
#     "instrument": "unknown",
#     "flowcell": "unknown",
#     "lane": "unknown",
# }
read_group_overrides = {}

# Whether to run some additional metric collections
# These are not required for cancer detection
COLLECT_INSERT_SIZES = False
COLLECT_WGS_METRICS = False

############
# Workflow #
############

if not SEQ_CENTER:
    raise ValueError("Please specify `SEQ_CENTER`.")

if bam_files and not isinstance(bam_files, dict):
    raise TypeError("When not empty, `bam_files` must be a dictionary.")
if fastq_files and not isinstance(fastq_files, dict):
    raise TypeError("When not empty, `fastq_files` must be a dictionary.")
if read_group_overrides and not isinstance(read_group_overrides, dict):
    raise TypeError("When not empty, `read_group_overrides` must be a dictionary.")


SHELL_SAFE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/@%+=:,-]+$")
SHELL_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._@%+=:,-]+$")


def resolve_path(path) -> pathlib.Path:
    return pathlib.Path(path).expanduser().resolve()


def has_read_permission(filepath):
    return os.access(str(filepath), os.R_OK)


def validate_shell_safe_path(filepath: pathlib.Path, label: str) -> pathlib.Path:
    filepath = pathlib.Path(filepath)
    if not SHELL_SAFE_PATH_PATTERN.fullmatch(str(filepath)):
        raise ValueError(
            f"`{label}` contains characters that are unsafe in unquoted shell commands: "
            f"{filepath}"
        )
    return filepath


def validate_sample_id(sample_id) -> str:
    if not isinstance(sample_id, str):
        raise TypeError(
            f"Sample IDs must be strings. Got {type(sample_id).__name__}: {sample_id!r}"
        )
    if not sample_id:
        raise ValueError("Sample IDs cannot be empty.")

    sample_path = pathlib.PurePath(sample_id)
    if (
        sample_path.is_absolute()
        or sample_id in {".", ".."}
        or len(sample_path.parts) != 1
    ):
        raise ValueError(
            f"Sample IDs must be plain names, not paths. Got: {sample_id!r}"
        )
    if not SHELL_SAFE_NAME_PATTERN.fullmatch(sample_id):
        raise ValueError(
            f"Sample ID contains characters that are unsafe in output paths or shell "
            f"commands: {sample_id!r}"
        )

    return sample_id


def validate_readable_file(filepath, label: str) -> pathlib.Path:
    filepath = resolve_path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"`{label}` does not exist: {filepath}")
    if not filepath.is_file():
        raise FileNotFoundError(f"`{label}` is not a file: {filepath}")
    if not has_read_permission(filepath):
        raise PermissionError(f"Lacks read permissions for `{label}`: {filepath}")
    return validate_shell_safe_path(filepath, label=label)


def validate_bam_path(filepath, sample_id: str) -> pathlib.Path:
    filepath = validate_readable_file(
        filepath, label=f"BAM file for sample {sample_id}"
    )
    if filepath.suffix.lower() != ".bam":
        raise ValueError(
            f"BAM file for sample {sample_id} must end with `.bam`, got: {filepath}"
        )
    return filepath


def validate_fastq_paths(paths, sample_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    if isinstance(paths, (str, bytes, os.PathLike)) or not isinstance(paths, Sequence):
        raise TypeError(
            f"FASTQ files for sample {sample_id} must be a 2-item sequence of paths."
        )
    if len(paths) != 2:
        raise ValueError(
            f"FASTQ files for sample {sample_id} must contain exactly 2 paths, got "
            f"{len(paths)}."
        )

    fq1_path = validate_readable_file(
        paths[0], label=f"FASTQ R1 file for sample {sample_id}"
    )
    fq2_path = validate_readable_file(
        paths[1], label=f"FASTQ R2 file for sample {sample_id}"
    )
    return fq1_path, fq2_path


def validate_bam_files(bam_files: dict) -> dict[str, pathlib.Path]:
    validated_bam_files = {}
    for sample_id, bam_path in bam_files.items():
        sample_id = validate_sample_id(sample_id)
        validated_bam_files[sample_id] = validate_bam_path(
            bam_path, sample_id=sample_id
        )
    return validated_bam_files


def validate_fastq_files(
    fastq_files: dict,
) -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    validated_fastq_files = {}
    for sample_id, fastq_paths in fastq_files.items():
        sample_id = validate_sample_id(sample_id)
        validated_fastq_files[sample_id] = validate_fastq_paths(
            fastq_paths, sample_id=sample_id
        )
    return validated_fastq_files


def validate_read_group_value(value, field_name: str, sample_id: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"Read-group field `{field_name}` for sample {sample_id} must be a string."
        )
    if not value:
        raise ValueError(
            f"Read-group field `{field_name}` for sample {sample_id} cannot be empty."
        )
    if not SHELL_SAFE_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            f"Read-group field `{field_name}` for sample {sample_id} contains "
            f"characters that are unsafe in shell commands: {value!r}"
        )
    return value


def validate_read_group_override(override: dict, sample_id: str) -> dict[str, str]:
    if not isinstance(override, dict):
        raise TypeError(f"`read_group_overrides[{sample_id!r}]` must be a dictionary.")

    required_fields = {"instrument", "flowcell", "lane"}
    provided_fields = set(override)
    missing_fields = required_fields - provided_fields
    extra_fields = provided_fields - required_fields
    if missing_fields or extra_fields:
        problems = []
        if missing_fields:
            problems.append(f"missing {sorted(missing_fields)}")
        if extra_fields:
            problems.append(f"unexpected {sorted(extra_fields)}")
        raise ValueError(
            f"`read_group_overrides[{sample_id!r}]` must contain exactly the keys "
            f"`instrument`, `flowcell`, and `lane` ({', '.join(problems)})."
        )

    return {
        field_name: validate_read_group_value(
            override[field_name], field_name=field_name, sample_id=sample_id
        )
        for field_name in ("instrument", "flowcell", "lane")
    }


def validate_read_group_overrides(
    read_group_overrides: dict,
) -> dict[str, dict[str, str]]:
    validated_overrides = {}
    for sample_id, override in read_group_overrides.items():
        sample_id = validate_sample_id(sample_id)
        validated_overrides[sample_id] = validate_read_group_override(
            override, sample_id=sample_id
        )
    return validated_overrides


def validate_output_data_dir(output_data_dir) -> pathlib.Path:
    return validate_shell_safe_path(
        resolve_path(output_data_dir), label="output_data_dir"
    )


output_data_dir = validate_output_data_dir(output_data_dir)
GENOME_FASTA = validate_readable_file(GENOME_FASTA, label="GENOME_FASTA")
bam_files = validate_bam_files(bam_files)
fastq_files = validate_fastq_files(fastq_files)
read_group_overrides = validate_read_group_overrides(read_group_overrides)

# Check no overlap between BAM and FASTQ
files = {"BAM": bam_files, "FASTQ": fastq_files}
if set(bam_files.keys()).intersection(fastq_files):
    raise ValueError("`bam_files` and `fastq_files` can't have overlapping sample IDs.")

undefined_override_samples = (
    set(read_group_overrides) - set(bam_files) - set(fastq_files)
)
if undefined_override_samples:
    raise ValueError(
        "`read_group_overrides` contains sample IDs that are not present in "
        "`bam_files` or `fastq_files`: "
        f"{sorted(undefined_override_samples)}"
    )


def legalize_target_name(target_name):
    # First check if target name is legal.
    if re.match(r"^[a-zA-Z_][a-zA-Z0-9._]*$", target_name):
        return target_name

    # Target name is not legal. Replace illegal characters by underscores.
    result = ""

    # First character is special.
    if re.match(r"[a-zA-Z_]", target_name[0]):
        result += target_name[0]
    else:
        result += "_"

    # Check remaining characters.
    for i in range(1, len(target_name)):
        if re.match(r"[a-zA-Z0-9._]", target_name[i]):
            result += target_name[i]
        else:
            result += "_"

    return result


def parse_illumina_read_header(header: str, source: str) -> tuple[str, str, str]:
    header = header.strip()
    if not header:
        raise ValueError(
            f"`{source}` is empty, so read-group information cannot be inferred."
        )
    if not header.startswith("@"):
        raise ValueError(
            f"`{source}` does not start with a FASTQ header line (`@...`). "
            f"First line: {header!r}"
        )

    parts = header[1:].split(":")
    if len(parts) < 4:
        raise ValueError(
            f"`{source}` has an unsupported read header format. Expected at least 4 "
            f"colon-separated fields after `@`, but got {len(parts)} in: {header!r}"
        )

    instrument, _, flowcell, lane = parts[:4]
    return instrument, flowcell, lane


def fastq_info(filename):
    with gzip.open(filename, "rt") as handle:
        first_line = handle.readline()

    return parse_illumina_read_header(first_line, source=str(filename))


def get_read_info(filename):
    bamfile = pysam.AlignmentFile(str(filename), "rb")
    for alignment in bamfile.fetch(until_eof=True):
        query_name = alignment.query_name
        parts = query_name.split(":")
        if len(parts) < 4:
            raise ValueError(
                f"`{filename}` has an unsupported BAM read name format. Expected at least "
                f"4 colon-separated fields in query name, but got {len(parts)} in: "
                f"{query_name!r}"
            )
        instrument, _, flowcell, lane = parts[:4]
        return instrument, flowcell, lane

    raise ValueError(
        f"`{filename}` contains no reads, so read-group information cannot be inferred."
    )


def get_read_group_info(
    sample_id: str, source, infer_read_group_info
) -> tuple[str, str, str]:
    override = read_group_overrides.get(sample_id)
    if override is not None:
        return override["instrument"], override["flowcell"], override["lane"]

    try:
        return infer_read_group_info(source)
    except ValueError as exc:
        raise ValueError(
            f"Could not infer read-group metadata for sample {sample_id} from "
            f"{source!s}. {exc} If this file does not use the expected Illumina "
            "header format, set `read_group_overrides` for this sample. If one or "
            'more fields are genuinely unknown, set them explicitly to `"unknown"`.'
        ) from exc


def trim_and_map(
    r1_file,
    r2_file,
    sample,
    library,
    seq_center,
    platform,
    instrument,
    flowcell,
    lane,
    output_folder: pathlib.Path,
    tmp_folder: pathlib.Path,
):
    read_group = f"@RG\\tCN:{seq_center}\\tPL:{platform}\\tID:{instrument}.{lane}\\tSM:{sample}\\tLB:{library}\\tPU:{flowcell}.{lane}"

    inputs = [r1_file, r2_file]
    outputs = {
        "bam_file": str(
            output_folder / f"{sample}_{library}_{flowcell}_{lane}.aligned.sorted.bam"
        ),
        "cutadapt_report": str(
            output_folder / f"{sample}_{library}_{flowcell}_{lane}.cutadapt.metrics.txt"
        ),
    }

    options = dict(cores="24", memory="32g", walltime="12:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    spec = f"""

    set -e

    SCRATCH_FOLDER={tmp_folder}/scratch/${{SLURM_JOBID}}
    mkdir -p ${{SCRATCH_FOLDER}}

    tmp_dir=$(mktemp -d --tmpdir=${{SCRATCH_FOLDER}})

    scratch_bam_file=${{SCRATCH_FOLDER}}/$(basename {outputs["bam_file"]})
    scratch_cutadapt_report=${{SCRATCH_FOLDER}}/$(basename {outputs["cutadapt_report"]})

    adapter=$(guessadapt -n 1000000 {r1_file} | head -n1 | cut -f1)

    seqtk mergepe <(zcat {r1_file}) <(zcat {r2_file}) \
    | \
    cutadapt --cores={options["cores"]} \
             --interleaved \
             --minimum-length=20 \
             --error-rate=0.1 \
             --quality-cutoff=20 \
             --overlap=1 \
             -a ${{adapter}} \
             -A ${{adapter}} \
             - 2> ${{scratch_cutadapt_report}} \
    | \
    bwa mem -p -Y -K 100000000 -t {options["cores"]} -R "{read_group}" {GENOME_FASTA} - \
    | \
    samtools sort -o ${{scratch_bam_file}} -

    mv ${{scratch_bam_file}} {outputs["bam_file"]}
    mv ${{scratch_cutadapt_report}} {outputs["cutadapt_report"]}

    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def mark_duplicates(
    bam_files, sample, output_folder: pathlib.Path, tmp_folder: pathlib.Path
):
    inputs = bam_files
    outputs = {
        "bam_file": str(output_folder / f"{sample}.aligned.sorted.markdup.bam"),
        "metrics_file": str(output_folder / f"{sample}.markdup_metrics.txt"),
    }

    options = dict(cores="1", memory="32g", walltime="02:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    input_string = " ".join([f"--INPUT {bam_file}" for bam_file in bam_files])

    spec = f"""

    set -e

    SCRATCH_FOLDER={tmp_folder}/scratch/${{SLURM_JOBID}}
    mkdir -p ${{SCRATCH_FOLDER}}

    tmp_dir=$(mktemp -d --tmpdir=${{SCRATCH_FOLDER}})

    scratch_markdup_bam_file=${{SCRATCH_FOLDER}}/$(basename {outputs["bam_file"]})
    scratch_markdup_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs["metrics_file"]})

    picard -Xmx{int(options["memory"][:-1]) - 8}g -Djava.io.tmpdir=${{tmp_dir}} \
        MarkDuplicates \
        {input_string} \
        --OUTPUT ${{scratch_markdup_bam_file}} \
        --METRICS_FILE ${{scratch_markdup_metrics_file}} \
        --OPTICAL_DUPLICATE_PIXEL_DISTANCE 2500 \
        --VALIDATION_STRINGENCY SILENT \
        --ASSUME_SORTED true

    mv ${{scratch_markdup_bam_file}} {outputs["bam_file"]}
    mv ${{scratch_markdup_metrics_file}} {outputs["metrics_file"]}

    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def index(bam_file):
    inputs = bam_file
    outputs = {"bam_index": str(bam_file) + ".bai"}

    options = dict(cores="1", memory="8g", walltime="12:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    spec = f"""
    set -e
    samtools index {inputs}
    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def collect_insert_size_metrics(
    bam_file, sample, output_folder: pathlib.Path, tmp_folder: pathlib.Path
):
    options = dict(cores="1", memory="8g", walltime="12:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    inputs = [bam_file]
    outputs = {
        "metrics": str(output_folder / f"{sample}.insert_size_metrics.txt"),
        "plot": str(output_folder / f"{sample}.insert_size_metrics.pdf"),
    }

    spec = f"""

    set -e

    
    SCRATCH_FOLDER={tmp_folder}/scratch/${{SLURM_JOBID}}
    mkdir -p ${{SCRATCH_FOLDER}}

    tmp_dir=$(mktemp -d --tmpdir=${{SCRATCH_FOLDER}})

    scratch_insert_size_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs["metrics"]})
    scratch_insert_size_metrics_plot=${{SCRATCH_FOLDER}}/$(basename {outputs["plot"]})

    picard -Xmx{options["memory"]} -Djava.io.tmpdir=${{tmp_dir}} \
           CollectInsertSizeMetrics \
           --VALIDATION_STRINGENCY SILENT \
           --INPUT {bam_file} \
           --OUTPUT ${{scratch_insert_size_metrics_file}} \
           --Histogram_FILE ${{scratch_insert_size_metrics_plot}} \
           --REFERENCE_SEQUENCE {GENOME_FASTA}

    mv ${{scratch_insert_size_metrics_file}} {outputs["metrics"]}
    mv ${{scratch_insert_size_metrics_plot}} {outputs["plot"]}

    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


def collect_wgs_metrics(
    bam_file, sample, output_folder: pathlib.Path, tmp_folder: pathlib.Path
):
    options = dict(cores="1", memory="8g", walltime="12:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    inputs = [bam_file]
    outputs = {"metrics": str(output_folder / f"{sample}.wgs_metrics.txt")}

    spec = f"""

    set -e

    SCRATCH_FOLDER={tmp_folder}/scratch/${{SLURM_JOBID}}
    mkdir -p ${{SCRATCH_FOLDER}}

    tmp_dir=$(mktemp -d --tmpdir=${{SCRATCH_FOLDER}})

    scratch_wgs_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs["metrics"]})

    picard -Xmx{options["memory"]} -Djava.io.tmpdir=${{tmp_dir}} \
           CollectWgsMetrics \
           --INPUT {bam_file} \
           --OUTPUT ${{scratch_wgs_metrics_file}} \
           --REFERENCE_SEQUENCE {GENOME_FASTA}

    mv ${{scratch_wgs_metrics_file}} {outputs["metrics"]}

    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


# Create gwf workflow with defaults
main_defaults = {
    "cores": 1,
    "memory": "32g",
    "walltime": "11:00:00",
}
if ACCOUNT:
    main_defaults["account"] = ACCOUNT

gwf = gwf.Workflow(defaults=main_defaults)


for file_type, file_collection in files.items():
    for sample_id, path_s in file_collection.items():
        sample_output_dir = output_data_dir / sample_id
        tmp_dir = sample_output_dir / "tmp"
        # Create output directory
        mk_dir(sample_output_dir, f"output directory for {sample_id}", messenger=None)
        mk_dir(tmp_dir, f"tmp directory for {sample_id}", messenger=None)

        if file_type == "BAM":
            bam_path = path_s
            # Name without .bam extension
            bam_prefix: str = bam_path.stem

            tmp_prefix = str(tmp_dir / bam_prefix)
            (
                instrument,
                flowcell,
                lane,
            ) = get_read_group_info(
                sample_id=sample_id,
                source=bam_path,
                infer_read_group_info=get_read_info,
            )  # Currently issues a warning if the BAM file is not indexed (harmless)

            fq0_path = str(sample_output_dir / (bam_prefix + "_R0.fq.gz"))
            fq1_path = str(sample_output_dir / (bam_prefix + "_R1.fq.gz"))
            fq2_path = str(sample_output_dir / (bam_prefix + "_R2.fq.gz"))

            (
                gwf.target(
                    legalize_target_name(f"bam2fastq_{sample_id}"),
                    inputs=[str(bam_path)],
                    outputs=[fq1_path, fq2_path],
                )
                << f"""\
            samtools collate -O {bam_path} {tmp_prefix} | samtools fastq -1 {fq1_path} -2 {fq2_path} -s {fq0_path}
            """
            )
        elif file_type == "FASTQ":
            fq1_path = str(path_s[0])
            fq2_path = str(path_s[1])
            instrument, flowcell, lane = get_read_group_info(
                sample_id=sample_id,
                source=fq1_path,
                infer_read_group_info=fastq_info,
            )
        else:
            raise ValueError(f"Unknown file type: {file_type}.")

        trim_and_map_target = gwf.target_from_template(
            legalize_target_name(f"TrimAndMap_{sample_id}"),
            trim_and_map(
                r1_file=fq1_path,
                r2_file=fq2_path,
                sample=sample_id,
                library="x",
                instrument=instrument,
                flowcell=flowcell,
                lane=lane,
                platform=PLATFORM,
                seq_center=SEQ_CENTER,
                output_folder=sample_output_dir,
                tmp_folder=tmp_dir,
            ),
        )

        mark_duplicates_target = gwf.target_from_template(
            legalize_target_name(f"MarkDuplicates_{sample_id}"),
            mark_duplicates(
                bam_files=[trim_and_map_target.outputs["bam_file"]],
                sample=sample_id,
                output_folder=sample_output_dir,
                tmp_folder=tmp_dir,
            ),
        )

        index_target = gwf.target_from_template(
            legalize_target_name(f"Index_{sample_id}"),
            index(bam_file=mark_duplicates_target.outputs["bam_file"]),
        )

        if COLLECT_INSERT_SIZES:
            collect_insert_size_metrics_target = gwf.target_from_template(
                legalize_target_name(f"CollectInsertSizeMetrics_{sample_id}"),
                collect_insert_size_metrics(
                    bam_file=mark_duplicates_target.outputs["bam_file"],
                    sample=sample_id,
                    output_folder=sample_output_dir,
                    tmp_folder=tmp_dir,
                ),
            )

        if COLLECT_WGS_METRICS:
            collect_wgs_metrics_target = gwf.target_from_template(
                legalize_target_name(f"CollectWgsMetrics_{sample_id}"),
                collect_wgs_metrics(
                    bam_file=mark_duplicates_target.outputs["bam_file"],
                    sample=sample_id,
                    output_folder=sample_output_dir,
                    tmp_folder=tmp_dir,
                ),
            )
