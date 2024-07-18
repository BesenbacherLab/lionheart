# Workflow for (re)mapping data (e.g. (hg19 BAM or FASTQ) -> hg38 BAM).
# Adapted from Genome-Mapping-0.1.4 by Michael Knudsen @ MOMA.
# Generalized by Ludvig Renbo Olsen @ MOMA.

import gzip
import re
import gwf
import pysam
import pathlib
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
# You can write code to generate these dicts

# "<sample_id_1>": "path/to/x.bam"
bam_files = {}

# Pass the two FASTQ files as a tuple with r1 first, r2 second
# "<sample_id_1>": ("path/to/x_r1.fq.gz", "path/to/x_r2.fq.gz")
fastq_files = {}

# Whether to run some additional metric collections
# These are not required for cancer detection
COLLECT_INSERT_SIZES = False
COLLECT_WGS_METRICS = False

############
# Workflow #
############

if not SEQ_CENTER:
    raise ValueError("Please specify `SEQ_CENTER`.")

# Check no overlap between BAM and FASTQ
files = {"BAM": bam_files, "FASTQ": fastq_files}
if set(bam_files.keys()).intersection(fastq_files):
    raise ValueError("`bam_files` and `fastq_files` can't have overlapping sample IDs.")


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


def fastq_info(filename):
    with gzip.open(filename, "rt") as handle:
        first_line = handle.readline()

    # This is standard for FASTQ files from Illumina sequencers.
    instrument, _, flowcell, lane = first_line[1:].split(":")[:4]

    return instrument, flowcell, lane


def get_read_info(filename):
    bamfile = pysam.AlignmentFile(filename, "rb")
    for alignment in bamfile.fetch(until_eof=True):
        instrument, _, flowcell, lane = alignment.query_name.split(":")[:4]
        break

    return instrument, flowcell, lane


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

    scratch_bam_file=${{SCRATCH_FOLDER}}/$(basename {outputs['bam_file']})
    scratch_cutadapt_report=${{SCRATCH_FOLDER}}/$(basename {outputs['cutadapt_report']})

    adapter=$(guessadapt -n 1000000 {r1_file} | head -n1 | cut -f1)

    seqtk mergepe <(zcat {r1_file}) <(zcat {r2_file}) \
    | \
    cutadapt --cores={options['cores']} \
             --interleaved \
             --minimum-length=20 \
             --error-rate=0.1 \
             --quality-cutoff=20 \
             --overlap=1 \
             -a ${{adapter}} \
             -A ${{adapter}} \
             - 2> ${{scratch_cutadapt_report}} \
    | \
    bwa mem -p -Y -K 100000000 -t {options['cores']} -R "{read_group}" {GENOME_FASTA} - \
    | \
    samtools sort -o ${{scratch_bam_file}} -

    mv ${{scratch_bam_file}} {outputs['bam_file']}
    mv ${{scratch_cutadapt_report}} {outputs['cutadapt_report']}

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

    options = dict(cores="1", memory="32g", walltime="01:00:00")
    if ACCOUNT:
        options["account"] = ACCOUNT

    input_string = " ".join([f"--INPUT {bam_file}" for bam_file in bam_files])

    spec = f"""

    set -e

    SCRATCH_FOLDER={tmp_folder}/scratch/${{SLURM_JOBID}}
    mkdir -p ${{SCRATCH_FOLDER}}

    tmp_dir=$(mktemp -d --tmpdir=${{SCRATCH_FOLDER}})

    scratch_markdup_bam_file=${{SCRATCH_FOLDER}}/$(basename {outputs['bam_file']})
    scratch_markdup_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs['metrics_file']})

    picard -Xmx{int(options['memory'][:-1]) - 8}g -Djava.io.tmpdir=${{tmp_dir}} \
        MarkDuplicates \
        {input_string} \
        --OUTPUT ${{scratch_markdup_bam_file}} \
        --METRICS_FILE ${{scratch_markdup_metrics_file}} \
        --OPTICAL_DUPLICATE_PIXEL_DISTANCE 2500 \
        --VALIDATION_STRINGENCY SILENT \
        --ASSUME_SORTED true

    mv ${{scratch_markdup_bam_file}} {outputs['bam_file']}
    mv ${{scratch_markdup_metrics_file}} {outputs['metrics_file']}

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

    scratch_insert_size_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs['metrics']})
    scratch_insert_size_metrics_plot=${{SCRATCH_FOLDER}}/$(basename {outputs['plot']})

    picard -Xmx{options['memory']} -Djava.io.tmpdir=${{tmp_dir}} \
           CollectInsertSizeMetrics \
           --VALIDATION_STRINGENCY SILENT \
           --INPUT {bam_file} \
           --OUTPUT ${{scratch_insert_size_metrics_file}} \
           --Histogram_FILE ${{scratch_insert_size_metrics_plot}} \
           --REFERENCE_SEQUENCE {GENOME_FASTA}

    mv ${{scratch_insert_size_metrics_file}} {outputs['metrics']}
    mv ${{scratch_insert_size_metrics_plot}} {outputs['plot']}

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

    scratch_wgs_metrics_file=${{SCRATCH_FOLDER}}/$(basename {outputs['metrics']})

    picard -Xmx{options['memory']} -Djava.io.tmpdir=${{tmp_dir}} \
           CollectWgsMetrics \
           --INPUT {bam_file} \
           --OUTPUT ${{scratch_wgs_metrics_file}} \
           --REFERENCE_SEQUENCE {GENOME_FASTA}

    mv ${{scratch_wgs_metrics_file}} {outputs['metrics']}

    """

    return AnonymousTarget(inputs=inputs, outputs=outputs, options=options, spec=spec)


# Create gwf workflow with defaults
main_defaults = {
    "cores": 1,
    "memory": "32g",
    "walltime": "01:00:00",
}
if ACCOUNT:
    main_defaults["account"] = ACCOUNT

gwf = gwf.Workflow(defaults=main_defaults)


for file_type, file_collection in files.items():
    for sample_id, path_s in file_collection.items():
        sample_output_dir = pathlib.Path(output_data_dir) / sample_id
        tmp_dir = sample_output_dir / "tmp"
        # Create output directory
        mk_dir(sample_output_dir, f"output directory for {sample_id}", messenger=None)
        mk_dir(tmp_dir, f"tmp directory for {sample_id}", messenger=None)

        if file_type == "BAM":
            bam_path = path_s
            # Name without .bam extension
            bam_prefix: str = pathlib.Path(bam_path).name[:-4]

            tmp_prefix = str(tmp_dir / bam_prefix)
            instrument, flowcell, lane = get_read_info(
                bam_path
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
            fq1_path = path_s[0]
            fq2_path = path_s[1]
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
