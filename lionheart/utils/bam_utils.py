import pathlib
import subprocess
from typing import Callable, Optional, Union
from utipy import Messenger, check_messenger
from lionheart.utils.subprocess import call_subprocess, check_paths_for_subprocess


def check_autosomes_in_bam(
    bam_path: Union[str, pathlib.Path],
    messenger: Optional[Callable] = Messenger(verbose=True, indent=0, msg_fn=print),
) -> None:
    """
    Given a BAM file, check that all chromosomes have the
    `chr` prefix and that all autosomes are present.
    """
    check_paths_for_subprocess(bam_path)
    call = " ".join(
        [
            "samtools",
            "view",
            "-H",
            str(bam_path),
            "|",
            "awk",
            "-F'\t'",
            "'" + '$1=="@SQ"{ name=substr($2,4); '
            "if(name ~ /^chr/){ "
            "  idx = substr(name,4) + 0; "
            "  if(idx>=1 && idx<=22){ autos[idx]=1 } "
            "} "
            "} "
            "END{ "
            "for(i=1;i<=22;i++){ "
            "  if(!(i in autos)){ "
            '    print "ERROR: missing chromosome chr" i > "/dev/stderr"; '
            "    err=1 "
            "  } "
            "} "
            "if(err) exit 1 "
            "}'",
            "||",
            "exit",
            "1",
        ]
    )
    try:
        call_subprocess(call, "BAM file did not contain the right chromosome names")
    except subprocess.CalledProcessError:
        messenger = check_messenger(messenger)
        messenger(
            "BAM file did not contain the right chromosome names. "
            "Must use 'chr*' naming and contain all autosomes (chr1 -> chr22)."
        )
        raise
    except BaseException:
        raise
