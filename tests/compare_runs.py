#!/usr/bin/env python3
"""CPU-vs-GPU (and cross-compiler) test harness for the GPU port.

Runs the same IMPACT-T input with two executables (a reference and a
candidate build -- e.g. gfortran vs nvfortran, or CPU vs GPU) and compares
the results. See doc/GPU_PORT_DESIGN.md, section "0.3 Test harness".

Two kinds of check:

- **Run-level** (implemented here): run both executables to completion and
  compare fort.18/24/25/26 (energy/x/y/z envelope statistics) column by
  column, at every row both runs have in common.
- **Field-level** (stub, for A2): IMPACT-T does not yet have an env-var
  hook to dump E/B at the particles for one step. Once A2 adds one (to
  `pt2ptFields` and/or the GPU kernel), point `--field-dump-env` at its
  name and `--field-dump-file` at the format it writes; `compare_fields`
  below reads two whitespace-separated files of (ex,ey,ez,bx,by,bz) rows,
  one per particle, and reports the max relative error.

Usage:
    compare_runs.py --ref ./ImpactTexe --cand ./ImpactTexe-gpu \\
        --input-dir ./impact_t_files --workdir /tmp/cmp \\
        [--mpi --nproc 1] [--rtol 1e-6]
"""
import argparse
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np

DIAG_FILES = ["fort.18", "fort.24", "fort.25", "fort.26"]


def run_case(exe, input_dir, workdir, mpi, nproc, mpirun="mpirun"):
    workdir = pathlib.Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    for f in pathlib.Path(input_dir).iterdir():
        if f.is_file():
            shutil.copy(f, workdir / f.name)

    if mpi:
        cmd = [mpirun, "-n", str(nproc), str(exe)]
    else:
        cmd = [str(exe)]

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=workdir, capture_output=True, text=True, timeout=3600
    )
    wall = time.time() - t0
    (workdir / "stdout.log").write_text(proc.stdout)
    (workdir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe} exited {proc.returncode} (see {workdir}/stderr.log)"
        )
    return workdir, wall, proc.stdout


def parse_timers(stdout):
    """Parse the ' <name>   <seconds> seconds' lines from showtime_Timer."""
    timers = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == "seconds":
            try:
                timers[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return timers


def load_diag(path):
    text = path.read_text().replace("D+", "E+").replace("D-", "E-")
    return np.loadtxt(text.splitlines())


def compare_diag_files(ref_dir, cand_dir):
    """Max relative error per file, over the rows both runs share.

    Each column is scaled by its own max magnitude (over both runs), not by
    the individual row's value: several columns here (px0, py0, ...) are
    beam-centroid quantities that hover near zero for a symmetric beam, so
    a per-row relative error blows up on noise even when the runs agree to
    the column's full working precision.
    """
    results = {}
    for name in DIAG_FILES:
        ref_f = ref_dir / name
        cand_f = cand_dir / name
        if not ref_f.exists() or not cand_f.exists():
            results[name] = None
            continue
        a = load_diag(ref_f)
        b = load_diag(cand_f)
        n = min(len(a), len(b))
        if n == 0:
            results[name] = None
            continue
        a_full, b_full = a, b
        a, b = a[:n], b[:n]
        scale = np.maximum(np.abs(a).max(axis=0), np.abs(b).max(axis=0))
        scale = np.maximum(scale, 1e-30)
        rel = np.abs(a - b) / scale
        results[name] = (float(np.max(rel)), n, len(a_full), len(b_full))
    return results


def compare_fields(ref_file, cand_file):
    """Max relative error between two (ex,ey,ez,bx,by,bz) dump files."""
    a = np.loadtxt(ref_file)
    b = np.loadtxt(cand_file)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    denom = np.maximum(np.abs(a), 1e-30)
    rel = np.abs(a - b) / denom
    return float(np.max(rel))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff-fields", nargs=2, metavar=("REF_FILE", "CAND_FILE"),
                     help="skip the run-level check; just report the max "
                          "relative error between two (ex,ey,ez,bx,by,bz) "
                          "dump files (e.g. from test_pt2ptfields' "
                          "IMPACTT_FIELD_DUMP hook) and exit")
    ap.add_argument("--rtol", type=float, default=1e-6,
                     help="max relative error to call it a pass (default 1e-6; "
                          "cross-compiler runs reorder sums, so this is looser "
                          "than the ~1e-12 the plan expects for GPU-vs-CPU "
                          "with the same compiler)")
    ap.add_argument("--ref", help="reference executable")
    ap.add_argument("--cand", help="candidate executable")
    ap.add_argument("--input-dir", help="dir with ImpactT.in + field data")
    ap.add_argument("--workdir", help="scratch dir for both runs")
    ap.add_argument("--mpi", action="store_true", help="launch with mpirun")
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--ref-mpirun", default="mpirun",
                     help="mpirun to launch --ref with (they may be built "
                          "against different MPI installs)")
    ap.add_argument("--cand-mpirun", default="mpirun",
                     help="mpirun to launch --cand with")
    args = ap.parse_args()

    if args.diff_fields:
        ref_file, cand_file = args.diff_fields
        max_rel = compare_fields(ref_file, cand_file)
        status = "OK" if max_rel <= args.rtol else "FAIL"
        print(f"fields: max relative error {max_rel:.3e} [{status}]")
        sys.exit(0 if max_rel <= args.rtol else 1)

    if not (args.ref and args.cand and args.input_dir and args.workdir):
        ap.error("--ref, --cand, --input-dir and --workdir are required "
                  "unless --diff-fields is given")

    workdir = pathlib.Path(args.workdir)
    ref_dir, ref_wall, ref_out = run_case(
        args.ref, args.input_dir, workdir / "ref", args.mpi, args.nproc,
        mpirun=args.ref_mpirun)
    cand_dir, cand_wall, cand_out = run_case(
        args.cand, args.input_dir, workdir / "cand", args.mpi, args.nproc,
        mpirun=args.cand_mpirun)

    print(f"ref  ({args.ref}): {ref_wall:.1f} s wall")
    print(f"cand ({args.cand}): {cand_wall:.1f} s wall")

    ref_timers = parse_timers(ref_out)
    cand_timers = parse_timers(cand_out)
    for name in ("integ",):
        rt = ref_timers.get(name)
        ct = cand_timers.get(name)
        if rt is not None or ct is not None:
            print(f"timer {name}: ref={rt} cand={ct}")

    results = compare_diag_files(ref_dir, cand_dir)
    ok = True
    for name, r in results.items():
        if r is None:
            print(f"{name}: MISSING")
            ok = False
            continue
        max_rel, n_common, n_ref, n_cand = r
        status = "OK" if max_rel <= args.rtol else "FAIL"
        if max_rel > args.rtol:
            ok = False
        extra = f" (ref had {n_ref} rows, cand {n_cand}, compared {n_common})" \
            if n_ref != n_cand else ""
        print(f"{name}: max relative error {max_rel:.3e} [{status}]{extra}")

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
