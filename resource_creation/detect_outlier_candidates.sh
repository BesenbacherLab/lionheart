#!/bin/bash
# Usage: ./detect_outlier_candidates.sh input_file.bam mosdepth_path threshold keep_file out_dir
#   input_file.bam: BAM file to run mosdepth on
#   mosdepth_path: Path to mosdepth executable (set LD_LIBRARY_PATH if needed)
#   threshold: probability threshold (e.g., 0.001)
#   keep_file: BED file with intervals to keep (tab-separated; at least chrom, start, end)
#   out_dir: output directory for zeros.txt and candidates.txt
#
# This script runs mosdepth, filters its output using bedtools intersect with the keep file,
# computes zero-inflated Poisson probabilities, and outputs:
#   zeros.txt: 0-indexed row indices (from the filtered rows) with count == 0.
#   candidates.txt: row index, count, and ZIP probability (tab-separated) for rows with count > threshold.
# Temporary files are stored in a temporary directory inside out_dir and removed on exit. 
# 
# NOTE: make script executable: chmod +x detect_outlier_candidates.sh

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 input_file.bam mosdepth_path threshold keep_file out_dir"
    exit 1
fi

input_file="$1"
mosdepth_path="$2"
threshold="$3"
keep_file="$4"
out_dir="$5"

# Create output directory if it doesn't exist
mkdir -p "$out_dir"

# Create temporary directory inside out_dir and clean it up on exit
tmpdir=$(mktemp -d -p "$out_dir" tmp.XXXXXXXX)
trap "rm -rf '$tmpdir'" EXIT

# Run mosdepth, outputting to the temporary directory
echo "Running mosdepth on BAM file..."
"$mosdepth_path" --by 10 --threads 4 --no-per-base --mapq 20 --min-frag-len 100 --max-frag-len 220 --fragment-mode "$tmpdir/coverage" "$input_file"
coverage_file="$tmpdir/coverage.regions.bed.gz"

# Filter the mosdepth output using keep file indices (chromosome and new index).
filtered_file="$tmpdir/filtered_input.bed"
echo "Filtering mosdepth output using keep file indices..."
zcat "$coverage_file" | gawk -F'\t' -v keep_file="$keep_file" 'BEGIN {
    # Build an array of indices from the keep file, keyed by "chrom:new_index".
    while ((getline line < keep_file) > 0) {
        split(line, a, "\t");
        key = a[1] ":" a[4];
        keep[key] = 1;
    }
    close(keep_file);
    prev = "";
    i = 0;
}
# Process only autosomes (chr1 to chr22) and compute the new index per chromosome.
$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {
    if ($1 != prev) {
        i = 1;
        prev = $1;
    } else {
        i++;
    }
    key = $1 ":" i;
    if (key in keep)
        print;
}
' > "$filtered_file"

# First pass on the filtered file: compute mean, total rows, nonzero count,
# maximum count, and p_nonzero.
echo "Calculate coverage statistics..."
read mean total nonzeros max_count p_nonzero < <(gawk -F'\t' '{
    count_val = $4;
    sum += count_val;
    total++;
    if (count_val != 0)
        nz++;
    if (count_val > max)
        max = count_val;
} END {
    if (total > 0)
        print sum/total, total, nz, max, nz/total;
}' "$filtered_file")

echo -e "  Mean: $mean\tTotal rows: $total\tNonzero rows: $nonzeros\tMax count: $max_count\tNonzero probability: $p_nonzero"

# Determine count_threshold: find the smallest count > int(mean)
# for which ZIP_prob = p_nonzero * (exp(-mean)*mean^count/gamma(count+1)) <= threshold.
echo "Calculate coverage threshold..."
count_threshold=$(gawk -F'\t' -v mean="$mean" -v p_nonzero="$p_nonzero" -v threshold="$threshold" '
function gamma(x, i, result) {
    result = 1;
    for (i = 1; i < x; i++) {
        result *= i;
    }
    return result;
}
BEGIN {
    count = int(mean) + 1;
    while ( p_nonzero * ( exp(-mean) * (mean^count) / gamma(count+1) ) > threshold ) {
            count++;
    }
    print count;
}' "$filtered_file")
echo "  Count threshold = $count_threshold"

# Second pass on the filtered file:
#   Cache full ZIP probability for counts from count_threshold to max_count,
#   and output tab-separated rows:
#    - zeros.txt: 0-indexed row indices (from the filtered rows) with count == 0.
#    - candidates.txt: row index, count, and ZIP probability for rows with count > count_threshold.
echo "Extracting outlier indices..."
candidates_file="$out_dir/candidates.txt"
zeros_file="$out_dir/zeros.txt"
gawk -F'\t' -v count_threshold="$count_threshold" -v mean="$mean" -v p_nonzero="$p_nonzero" -v max_count="$max_count" -v out_dir="$out_dir" '
function gamma(x, i, result) {
    result = 1;
    for (i = 1; i < x; i++) {
        result *= i;
    }
    return result;
}
BEGIN {
    idx = 0;
    # Precompute full ZIP probability for counts from count_threshold to max_count.
    for (i = count_threshold; i <= max_count; i++) {
         zip_cache[i] = p_nonzero * ( exp(-mean) * (mean^i) / gamma(i+1) );
    }
}
{
    count_val = $4;
    if (count_val == 0) {
        print idx > out_dir"/zeros.txt";
    } else if (count_val > count_threshold) {
         zip_prob = zip_cache[count_val];
         print idx "\t" count_val "\t" zip_prob > out_dir"/candidates.txt";
    }
    idx++;
}
' "$filtered_file"

read num_candidates _ < <(wc -l "$candidates_file")
read num_zeros _ < <(wc -l "$zeros_file")
echo -e "  # candidates: $num_candidates\t# zeros: $num_zeros"

echo "Processing complete. Results are in $out_dir."
