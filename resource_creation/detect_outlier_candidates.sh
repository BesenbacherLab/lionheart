#!/bin/bash
# Usage: ./detect_outlier_candidates.sh input_file.bam mosdepth_path threshold bin_size keep_file out_dir
#   input_file.bam: BAM file to run mosdepth on
#   mosdepth_path: Path to mosdepth executable (set LD_LIBRARY_PATH if needed)
#   threshold: probability threshold (e.g., 0.001)
#   bin_size: window size for mosdepth (--by option)
#   keep_file: BED file with intervals to keep (tab-separated; at least chrom, start, end)
#   out_dir: output directory for zeros.txt, candidates.txt, etc.
#
# This script runs mosdepth, filters its output using bedtools intersect with the keep file,
# computes zero-inflated Poisson probabilities, and outputs:
#   zeros.txt: lines (per chromosome) with count == 0 with chrom and per-chrom index
#   candidates.txt: chrom, per-chrom index, count, and ZIP tail probability for rows with count > mean and > computed threshold
#
# Temporary files are stored in a temporary directory inside out_dir and removed on exit
# NOTE: make script executable: chmod +x detect_outlier_candidates.sh

set -euo pipefail

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 input_file.bam mosdepth_path threshold bin_size keep_file out_dir"
    exit 1
fi

input_file="$1"
mosdepth_path="$2"
threshold="$3"
bin_size="$4"
keep_file="$5"
out_dir="$6"

# Create output directory if it doesn't exist
mkdir -p "$out_dir"

# Create a temporary directory inside out_dir and remove it on exit
tmpdir=$(mktemp -d -p "$out_dir" tmp.XXXXXXXX)
trap "rm -rf '$tmpdir'" EXIT

# Detect the directory where this script resides
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run mosdepth, outputting to the temporary directory
echo "Running mosdepth on BAM file..."
"$mosdepth_path" --by "$bin_size" --threads 4 --no-per-base --mapq 20 --min-frag-len 20 --max-frag-len 600 --fragment-mode "$tmpdir/coverage" "$input_file"
coverage_file="$tmpdir/coverage.regions.bed.gz"

# Filter the mosdepth output using keep file indices (chromosome and new index)
filtered_file="$tmpdir/filtered_input.bed"
echo "Filtering mosdepth output using keep file indices..."
unpigz -c "$coverage_file" | gawk -F'\t' -v keep_file="$keep_file" 'BEGIN {
    # Build an array from the keep file keyed by "chrom:new_index"
    while ((getline line < keep_file) > 0) {
        split(line, a, "\t")
        key = a[1] ":" a[4]
        keep[key] = 1
    }
    close(keep_file)
    prev = ""
    i = 0  # Never used
}
# Process only autosomes (chr1 to chr22) and assign a per-chromosome index
$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {
    if ($1 != prev) {
        i = 0
        prev = $1
    } else {
        i++
    }
    key = $1 ":" i
    if (key in keep)
        # Append the per-chromosome index as a new column
        print $0 "\t" i
}
' > "$filtered_file"

# Compute histogram of rounded coverage counts and save to histogram.txt
echo "Computing histogram of coverage counts..."
gawk -F'\t' '{
    count_int = int($4 + 0.5)
    hist[count_int]++
}
END {
    for (c in hist)
        print c "\t" hist[c]
}' "$filtered_file" > "$out_dir/histogram.txt"

# First pass on the filtered file: compute mean, total rows, nonzero count,
# maximum count, and p_nonzero
echo "Calculate coverage statistics..."
read mean total nonzeros max_count p_nonzero < <(gawk -F'\t' '{
    count_val = $4
    count_int = int(count_val + 0.5)
    # Use floating point for mean calculation
    # As it is more precise
    sum += count_val
    total++
    if (count_int != 0)
        nz++
    if (count_int > max)
        max = count_int
} END {
    if (total > 0)
        print sum/total, total, nz, max, nz/total
}' "$filtered_file")

echo -e "  Mean: $mean\tTotal rows: $total\tNonzero rows: $nonzeros\tMax count: $max_count\tNonzero probability: $p_nonzero"

# Precompute tail probabilities using Python
cdf_lookup_file="$tmpdir/cdf_lookup.tsv"
python3 "$script_dir/calculate_tail_cdf.py" --mean "$mean" --p_nonzero "$p_nonzero" --max_count "$max_count" --out_file "$cdf_lookup_file"

# Determine count_threshold using only the lookup table
echo "Calculate coverage threshold..."
count_threshold=$(gawk -M -v PREC=100 -F'\t' -v threshold="$threshold" -v mean="$mean" -v max_count="$max_count" '
FNR==NR {
    if (NR == 1) next
    tailprobs[$1] = $2
    next
}
END {
    count = int(mean) + 1
    while (count <= max_count && tailprobs[count] > threshold) {
        count++
    }
    print count
}' "$cdf_lookup_file")
echo "  Count threshold = $count_threshold"

# Save statistics to a TSV file
stats_file="$out_dir/stats.tsv"
echo -e "mean\ttotal\tnonzeros\tmax_count\tp_nonzero\tcount_threshold" > "$stats_file"
echo -e "$mean\t$total\t$nonzeros\t$max_count\t$p_nonzero\t$count_threshold" >> "$stats_file"

# Second pass on the filtered file:
# For each chromosome, reuse the per-chromosome index computed in the filtering step.
# Only consider outlier candidates where count > count_threshold.
# Output:
#   zeros.txt: chrom and per-chrom index for rows with count == 0
#   candidates.txt: chrom, per-chrom index, count, and tail probability
echo "Extracting outlier indices per chromosome..."
candidates_file="$out_dir/candidates.txt"
zeros_file="$out_dir/zeros.txt"
gawk -M -v PREC=80 -F'\t' -v count_threshold="$count_threshold" -v out_dir="$out_dir" '
BEGIN {
    # No need for a separate index; we reuse the per-chromosome index (last field)
}
# Read the lookup table from the Python-generated file
FNR==NR {
    if (NR == 1) next
    tailprobs[$1] = $2
    next
}
# Process the filtered file
{
    chrom = $1
    # The filtered file now has an extra field (the last field) containing the per-chrom index
    chrom_idx = $NF
    count_val = $4
    count_int = int(count_val + 0.5)
    if (count_int == 0) {
         print chrom "\t" chrom_idx > out_dir"/zeros.txt"
    } else if (count_int > count_threshold) {
         tail_prob = (count_int in tailprobs) ? tailprobs[count_int] : 0
         print chrom "\t" chrom_idx "\t" count_val "\t" tail_prob > out_dir"/candidates.txt"
    }
}
' "$cdf_lookup_file" "$filtered_file"

read num_candidates _ < <(wc -l "$candidates_file")
read num_zeros _ < <(wc -l "$zeros_file")
echo -e "  # candidates: $num_candidates\t# zeros: $num_zeros"

echo "Processing complete. Results are in $out_dir."
