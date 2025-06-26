#!/usr/bin/mawk -f
# This script saves the (non-duplicate) file indices and values of non-zero overlaps
# given a data frame of (chromosome, bin index, overlap)
# Consequtive index-duplicates are summed
# The non-zero indices and values are saved in chromosome-wise files
# allowing the construction of sparse arrays without mapping 
# the non-zero values onto the original data frame (expensive)
# NOTE: make script executable: chmod +x sparsify_overlaps.awk
BEGIN {
    # These variables are set by `-v outdir=... -v chr_col=...` etc on the command line
    # We make sure they have defaults in case they're not provided
    if (outdir == "") outdir = "."
    if (chr_col == 0) chr_col = 1
    if (bin_col == 0) bin_col = 2
    if (overlap_col == 0) overlap_col = 3

    # Field separators for reading/writing
    FS = "\t"
    OFS = "\t"
}

# The main logic
{
    # Extract fields based on user-specified column indices
    currChr       = $(chr_col)
    currBin       = $(bin_col) + 0
    currOverlap   = $(overlap_col) + 0

    # If this is the first line (NR == 1), initialize
    if (NR == 1) {
        prevChrom  = currChr
        groupBin   = currBin
        groupSum   = currOverlap
        next
    }

    # If chromosome changed, finalize old group
    if (currChr != prevChrom) {
        if (groupSum != 0) {
            print groupBin, groupSum >> (outdir "/" prevChrom ".sparsed.txt")
        }
        close(outdir "/" prevChrom ".sparsed.txt")

        # Reset for new chromosome
        groupBin   = currBin
        groupSum   = currOverlap
        prevChrom  = currChr
    } else {
        # Same chromosome
        if (currBin == groupBin) {
            # duplicate bin => add overlap
            groupSum += currOverlap
        } else {
            # New bin => finalize old bin group if nonzero
            if (groupSum != 0) {
                print groupBin, groupSum >> (outdir "/" currChr ".sparsed.txt")
            }
            groupBin  = currBin
            groupSum  = currOverlap
        }
    }
}

END {
    # Finalize last group
    if (groupSum != 0) {
        print groupBin, groupSum >> (outdir "/" prevChrom ".sparsed.txt")
    }
    close(outdir "/" prevChrom ".sparsed.txt")
}
