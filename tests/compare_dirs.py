#!/usr/bin/env python3
"""Compare fort.18/24/25/26 of two finished run directories (column-scaled
max error, as compare_runs.py).  Usage: compare_dirs.py REF_DIR CAND_DIR [...]
Pairs of directories may be given repeatedly: REF1 CAND1 REF2 CAND2 ..."""
import sys
import pathlib
from compare_runs import compare_diag_files

args = sys.argv[1:]
for ref, cand in zip(args[0::2], args[1::2]):
    res = compare_diag_files(pathlib.Path(ref), pathlib.Path(cand))
    row = []
    for name, v in res.items():
        row.append(f"{name}: " + ("missing" if v is None else f"{v[0]:.2e} ({v[1]} rows of {v[2]}/{v[3]})"))
    print(pathlib.Path(cand).name, " | ".join(row))
