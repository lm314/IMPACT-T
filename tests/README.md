# A1 test: pt2ptFields

`test_pt2ptfields.f90` checks the point-to-point space-charge field routine
(`pt2ptFields` in `src/Appl/BeamBunch.f90`) against an independent
reference implementation of the same pair formulas, at 1 and 4 MPI ranks.
It is not part of the CMake build (it links against the compiled
`ImpactTexe-mpi` object files directly), so build `build-mpi` first
(`-DUSE_MPI=ON -DUSE_FFTW=ON`), then:

```
cd ../build-mpi
find CMakeFiles/ImpactTexe-mpi.dir -name "*.o" | \
  grep -vE "Contrl/(main|AccSimulator|Input|Output)\.f90\.o" > /tmp/objlist.txt
cat /tmp/objlist.txt | xargs mpif90 -cpp -DUSE_MPI -fallow-argument-mismatch -I. \
  -o ../tests/test_pt2ptfields ../tests/test_pt2ptfields.f90 \
  <fftw3 and MPI shared libraries, as printed by
   CMakeFiles/ImpactTexe-mpi.dir/link.txt>

cd ../tests
mpirun -n 1 ./test_pt2ptfields
mpirun -n 4 ./test_pt2ptfields
```

(Build the object list to a file and pipe it through `xargs` rather than
`$(find ...)` inline on the command line -- under zsh, an unquoted
command substitution is passed as a single word, not one word per line,
so the linker sees one mangled "filename" instead of ~30 object files.)

For `USE_GPU=ON` (A2 onward), build `build-mpi-gpu` the same way (with
`-DUSE_GPU=ON`, nvfortran) and link `test_pt2ptfields`/`bench_pt2ptfields`
against *that* build's object list instead, on a node with a GPU, so the
`!$acc` directives in `pt2ptFields` actually offload.

Checks (see IMPACTT_GPU_PORT_PLAN.md, step A1):

- **A**: image on, matches a direct+self-image reference to ~1e-12.
- **B**: image off, matches a direct-only reference to ~1e-13 (this is
  the state after `zimage`, and the state whenever `Flagimg=0`).
- **C**: sanity check that the image flag actually changes the result.
- **D**: a single particle at rest at height z matches the analytic
  self-image field, e/(4 pi eps0 (2z)^2), to machine precision.

For A2's GPU-vs-CPU field-level check, set `IMPACTT_FIELD_DUMP=<path>`
to also dump Check A's (image on) computed (ex,ey,ez,bx,by,bz), one row
per particle in global order (ranks' local fields are gathered to rank
0 with `MPI_GATHERV`, so the file is the same regardless of nproc).
Build the same `test_pt2ptfields.f90` once against a `USE_GPU=ON`
object set and run it on a GPU node, and once against a CPU-only
(`USE_GPU=OFF`) object set, then diff the two dumps:

```
python3 compare_runs.py --diff-fields gpu_fields.txt cpu_fields.txt --rtol 1e-10
```

The same dump/diff mechanism is how A3 checks that 1/2/4/8 GPUs give
identical results: dump at each GPU count and diff pairwise against the
1-GPU file.

# A2 performance check: bench_pt2ptfields

`bench_pt2ptfields.f90` measures pair evaluations/second for
`pt2ptFields` at 1M particles, single rank, image charge on (2*N^2 pairs
per call -- the direct term over all pairs but the particle's own, plus
the image term over all pairs including the particle's own). Link it the
same way as `test_pt2ptfields` above, against a `USE_GPU=ON` object set
to measure one H100 or `USE_GPU=OFF` to measure one CPU core:

```
mpirun -n 1 ./bench_pt2ptfields
```

Prints the measured rate and, assuming it holds at scale, the projected
per-step and whole-run (2500-step) time at the plan's 6.24M-particle
target case.

# A2 GPU profiling: bench_pt2ptfields_prof

`bench_pt2ptfields_prof.f90` is a smaller-N (100k particles), one-rep
variant of `bench_pt2ptfields.f90`, for capturing an `ncu` (Nsight
Compute) profile of the `pt2ptFields` kernel without a full-metric-set
capture taking forever. Link it against a `USE_GPU=ON` object set (same
way as the other tests), then profile the timed call (the 2nd kernel
launch; the 1st is an untimed warm-up) with e.g.:

```
ncu --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight --section SpeedOfLight_RooflineChart \
    --section Occupancy --section WarpStateStats \
    --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
    -f -o ncu_report mpirun -n 1 ./bench_pt2ptfields_prof
ncu -i ncu_report.ncu-rep --page details
```

See IMPACTT_GPU_PORT_PLAN.md's A2 Progress notes for what this found.

# A3 multi-GPU: bench_pt2ptfields_mgpu

`bench_pt2ptfields_mgpu.f90` measures scaling across several GPUs, one
MPI rank per GPU. Unlike `bench_pt2ptfields.f90` (single rank only), it
splits NTOT=1M particles into `nptlc=NTOT/nproc` local targets per
rank; every rank still gathers all NTOT particles as sources
(`MPI_ALLGATHERV`), exactly as production does, so the total pair work
(`2*NTOT^2`) is fixed and just divided across ranks/GPUs. Link it the
same way as the other GPU tests, against a `USE_GPU=ON` object set, and
launch with `srun` (not `mpirun`) so each rank gets its own GPU:

```
srun --mpi=pmix --ntasks=P --gpus-per-task=1 ./bench_pt2ptfields_mgpu
```

Reports `elapsed_max` (the slowest rank's wall time -- what the
parallel run is actually bounded by) and the aggregate rate. Run
`test_pt2ptfields` the same way (with `IMPACTT_FIELD_DUMP` at each P)
for the consistency check.

**Run each P as its own job, sequentially, not concurrently**: Slurm
can schedule independent jobs for different P onto the same shared
node at the same time if resources allow, and since every rank does
its own full ~80MB host-to-GPU copy of the (unchanging-in-size) source
array every call, concurrent jobs contend for host-GPU bandwidth and
corrupt the timing -- this produced a clearly wrong "super-linear"
scaling result once (P=1 contaminated by an overlapping P=2 job) before
being caught by checking job start/end times. Chain jobs with
`--dependency=afterany:<jobid>` so each starts only after the previous
one finishes; this still never holds more GPUs than the current step
needs.

See IMPACTT_GPU_PORT_PLAN.md's A3 Progress notes for the results.

# Phase 0 test harness: compare_runs.py

`compare_runs.py` runs the same IMPACT-T input with two executables and
compares fort.18/24/25/26 (run-level check) and the `showtime_Timer`
timers printed to stdout (needs the MPI build, even at 1 rank -- the
serial build's timers always read 0). Used for:

- the `nvfortran` CPU-only build check against `gfortran` (0.2): looser
  tolerance (`--rtol`, default 1e-6), since different compilers reorder
  floating-point sums;
- CPU-vs-GPU checks from A2 onward: tight tolerance (`--rtol 1e-12`),
  since the plan expects near-bit-identical results there.

```
python3 compare_runs.py --ref /path/to/ImpactTexe-mpi \
    --cand /path/to/ImpactTexe-mpi-gpu \
    --input-dir /path/to/impact_t_files \
    --workdir /tmp/cmp --mpi --nproc 1 --rtol 1e-12
```

It also has a `--diff-fields REF CAND` mode (A2) that skips the run-level
check and just reports the max relative error between two
(ex,ey,ez,bx,by,bz) dump files, e.g. `test_pt2ptfields`'s
`IMPACTT_FIELD_DUMP` output -- see the A2 section above.

# B1 GPU Poisson solve: test_poisson_gpu

`test_poisson_gpu.f90` checks `openBC3D_gpu` (`src/Func/PoissonGPU.f90`)
against `openBC3Dnew` (FFTW, 1 rank) and checks rank independence. Link it
against a `USE_GPU=ON` object set (same way as the A2/A3 tests, with
`-cuda -cudalib=cufft`), and run on a GPU node:

```
test_poisson_gpu NX NY NZ [NPY NPZ [DUMPFILE]]
```

- 1 rank: compares phi and phi_img for zmin = cathode / near / far (and
  `flagImage=0` giving exactly 0), and times the GPU solve (image on).
  `SKIPCPU=1` skips `openBC3Dnew` (use it at 512x512x64: ~hours on a core).
  For grids up to 40000 points it also sums the image potential directly.
- Several ranks (`srun --ntasks=NPY*NPZ --gpus-per-task=1`): GPU only;
  `DUMPFILE` gets the gathered phi/phi_img as raw binary, so dumps from
  different (NPY,NPZ) can be compared with `cmp`.

phi must match to 1e-12. phi_img matches only to the conditioning of the
integrated Green's function (an 8-corner difference of large values,
noise ~ eps*(distance/cell)^3, identical on CPU and GPU), so the test
requires 1e-11 (cathode), 2e-10 (near), and at "far" that the GPU is no
further from the brute-force sum than the CPU is.

At run time, `IMPACTT_POISSON=cpu` makes the GPU executable use the FFTW
solver, for CPU-vs-GPU full-run comparisons. Full-run inputs are generated
by `b1_cases/make_cases.py` (in the plan folder).

# B2 particles on the GPU: test_particle_gpu

`test_particle_gpu.f90` compares each GPU particle kernel
(`src/Appl/ParticleGPU.f90`, `FieldmapGPU.f90`) with the CPU routine it
ports, on the same input (1 rank, 1 GPU). Link it like `test_poisson_gpu`
against a `USE_GPU=ON` object set, and run it in a directory holding
`rfdata1001`-`1003` and `1T1.T7` (`examples/Sample4_2Dfield`):

- `kick_none` (which exercises `extfld_dev` for element types 105 and 112,
  overlapping, 5 times across an RF period) against `scatter20t_BeamBunch`;
- `kickpt2pt_gpu` against `kickpt2ptImg_BeamBunch`, image on and off;
- drifts and emission against `drifthalf`/`drifthalforg`/`driftemission` and
  the first-order emission loop of `AccSimulator`;
- `singlerange_gpu` against `singlerange`; `lost_gpu` against
  `lostREC_BeamBunch` with 0 and 37 lost particles;
- deposition (`flagpos` 0 and 1) and `gradEB_dev` against reference loops
  transcribed from `deposit1_Depositor` and the 1-rank, `Flagbc = 1` branches
  of `gradEB_FieldQuant`.

Kicks are compared as the change in momentum, scaled by the largest change.
The device kernels are not bit-identical to the host routines: the
compiler contracts multiply-adds into FMAs on the device, which moves the
last bit (about 1e-16 relative) in a minority of values. The lost test and
`gradEB_dev` are bit-identical.

`compare_dirs.py REF CAND [REF CAND ...]` compares `fort.18/24/25/26` of
finished run directories (needs numpy). Full-run inputs are generated by
`b2_cases/make_cases.py` (in the plan folder); run the default mode (all GPU)
and `IMPACTT_PARTICLES=cpu` (B1: CPU particles, GPU solve) of the same
executable and compare.
