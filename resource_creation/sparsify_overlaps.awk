#!/usr/bin/mawk -f
# This script saves the (non-duplicate) file indices and values of non-zero overlaps
# given a data frame of (chromosome, bin index, overlap)
# Consequtive duplicates are summed
# The new index is restarted per chromosome
# The non-zero index and values are saved in chromosome-wise files
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
        groupChrom = currChr
        groupBin   = currBin
        groupSum   = currOverlap
        groupIndex = 0
        next
    }

    # If chromosome changed, finalize old group
    if (currChr != prevChrom) {
        if (groupSum != 0) {
            print groupIndex, groupSum >> (outdir "/" prevChrom ".sparsed.txt")
        }
        close(outdir "/" prevChrom ".sparsed.txt")

        # reset for new chromosome
        groupIndex = 0
        groupChrom = currChr
        groupBin   = currBin
        groupSum   = currOverlap
        prevChrom  = currChr
    } else {
        # same chromosome
        if (currBin == groupBin) {
            # duplicate bin => add overlap
            groupSum += currOverlap
        } else {
            # new bin => finalize old bin group if nonzero
            if (groupSum != 0) {
                print groupIndex, groupSum >> (outdir "/" currChr ".sparsed.txt")
            }
            groupIndex++
            groupBin  = currBin
            groupSum  = currOverlap
        }
    }
}

END {
    # finalize last group
    if (groupSum != 0) {
        print groupIndex, groupSum >> (outdir "/" prevChrom ".sparsed.txt")
    }
    close(outdir "/" prevChrom ".sparsed.txt")
}
