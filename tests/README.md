# Tests for the GPU port

These programs check the GPU port piece by piece against the CPU routines it
replaces. They are not part of the CMake build. Each one is linked against
the object files of an existing build with `build_test.sh`. What the port
does is described in `doc/GPU_PORT.md`, and the design sections the comments
cite (A1, B2.4, D1.12, ...) are in `doc/GPU_PORT_DESIGN.md`.

| Test | Checks | Build | Ranks |
|---|---|---|---|
| `test_pt2ptfields.f90` | point-to-point fields (`pt2ptFields`) against an independent reference, image on and off, analytic self-image, `r0img` cutoff | CPU or GPU | 1 and 4 |
| `test_pt2pt_block.f90` | the blocked CPU pair loop of `pt2ptFields` against the unblocked loop, bit for bit, for block sizes 1 to larger than the source count, image on and off | CPU | 1 |
| `test_bendnbody_cpu.f90` | the N-body kick in the bend frame (`kickpt2ptBd_BeamBunch`) against the analytic two-particle field | CPU | 1 |
| `bench_pt2ptfields.f90` | pair evaluations per second of `pt2ptFields` (1M particles) | CPU or GPU | 1 |
| `bench_pt2ptfields_prof.f90` | the same at 100k, one call, for an Nsight Compute profile | GPU | 1 |
| `bench_pt2ptfields_mgpu.f90` | N-body scaling over several GPUs, one rank per GPU | GPU | 1–8 |
| `test_poisson_gpu.f90` | the GPU Poisson solve `openBC3D_gpu` against `openBC3Dnew` (FFTW) and a direct sum; independence of the rank count | GPU | 1, several |
| `test_particle_gpu.f90` | each GPU particle kernel (drifts, emission, deposit, gradients, kicks, external fields of types 105 and 112, losses) against the CPU routine it ports | GPU | 1 |
| `test_bend_gpu.f90` | the field kinds of types 1, 4 and 111, the bend-frame kernels and diagnostics, and the N-body kick in the bend frame, against the CPU routines | GPU | 1 |
| `test_dtl_gpu.f90` | the type-101 field (kind 6 of `FieldmapGPU.f90`) against `getfldt_DTL`, nine cells, 5 times, 1e5 points each | GPU | 1 |
| `compare_runs.py` | runs one input with two executables and compares `fort.18/24/25/26` and the timers | any | any |
| `compare_dirs.py` | compares `fort.18/24/25/26` of finished run directories | any | any |
| `make_test_inputs.py` | writes synthetic input files for `test_bend_gpu` and `test_particle_gpu` | — | — |

## Building a test

```
cmake -S src -B build-mpi -DUSE_MPI=ON -DUSE_FFTW=ON            # CPU (gfortran)
cmake -S src -B build-gpu -DUSE_MPI=ON -DUSE_FFTW=ON -DUSE_GPU=ON  # GPU (nvfortran)
cmake --build build-mpi
tests/build_test.sh build-mpi tests/test_pt2ptfields.f90         # writes tests/test_pt2ptfields
```

`build_test.sh` compiles the test with the build's compiler, flags and module
files and links it with all of the build's objects except the main program. A
test linked against a `USE_GPU=ON` build runs its kernels on the GPU (the
`!$acc` and CUDA Fortran code is inert in a CPU build). Run GPU tests with
`srun` (`--gpus-per-task=1`), not `mpirun`, so that each rank gets its own GPU.

Kicks and fields on the GPU are not bit-identical to the CPU: the device
compiler contracts multiply-adds into FMAs and the device sums in a different
order, which moves the last bits. The tolerances below allow for that.

## test_pt2pt_block

On the CPU, `pt2ptFields` passes over the sources in blocks of
`IMPACTT_PT2PT_BLOCK` sources (default 4096), so that a block stays in cache
while every local target reads it. Each target still adds the sources in order
into one running sum, so the fields do not depend on the block size. The test
compares the routine for block sizes 1, 7, 1000, 4096, 3000, 3001 and 100000
(3001 sources, 1200 targets at a rank offset, unemitted sources, one pair
inside the cutoff) with a copy of the unblocked loop and expects no differing
bits:

```
mpirun -n 1 tests/test_pt2pt_block
```

A build with `-march` options that enable FMA can contract the two copies
differently; the test is meant for the default flags.

## test_pt2ptfields

`pt2ptFields` (`src/Appl/BeamBunch.f90`) against an independent
implementation of the same pair formulas, 2000 synthetic particles:

```
mpirun -n 1 tests/test_pt2ptfields
mpirun -n 4 tests/test_pt2ptfields
```

- **A**: image on, against a direct + self-image reference: about 4e-12.
- **B**: image off (after `zimage`, or `Flagimg = 0`), against a direct-only reference: about 4e-13.
- **C**: the image term changes the result (sanity).
- **D**: a single particle at rest at height z against the analytic self-image field e/(4πε₀(2z)²): exact.
- **E**: the image floor `r0img` (separate from `r0`): the self-image follows 1/(2z)² down to 2z = `r0img` and falls linearly (2z/`r0img`³) below it, peaking at 2z = `r0img`; `r0img = r0` caps the image at `r0`.

Both rank counts give the same numbers. `IMPACTT_FIELD_DUMP=<file>` also
writes check A's fields (one row per particle, in global order whatever the
rank count). Dumps from a CPU and a GPU build, or from different GPU counts,
are compared with

```
python3 tests/compare_runs.py --diff-fields gpu_fields.txt cpu_fields.txt --rtol 1e-10
```

## test_bendnbody_cpu

Two particles side by side at γ = 57 in the bend frame, no dipole field, one
of them at z < 0 (every particle is a source in a bend): the transverse kick
equals e/(4πε₀r²γ)·(cΔt)/(mc²/e) to about 2e-11 (tolerance 1e-10, set by the
code's constants).

## bench_pt2ptfields, bench_pt2ptfields_prof

`bench_pt2ptfields` times `pt2ptFields` at 1M particles with the image on
(2N² pair evaluations per call) and prints the rate. On one H100 the GPU build
reaches about 2.0e11 pair evaluations per second. `bench_pt2ptfields_prof` is
a 100k, one-call variant for Nsight Compute:

```
ncu --launch-skip 1 --launch-count 1 --section SpeedOfLight \
    --section Occupancy --section ComputeWorkloadAnalysis \
    -f -o ncu_report mpirun -n 1 tests/bench_pt2ptfields_prof
```

The kernel is compute-bound in FP64. `-gpu=maxregcount:64` (set in
`src/CMakeLists.txt`) gave the best rate in a sweep of the register cap.

## bench_pt2ptfields_mgpu

1M particles split over P ranks, one GPU each; every rank gathers all
particles as sources (`MPI_ALLGATHERV`), as in a run, so the pair work is
fixed and divided over the GPUs:

```
srun --mpi=pmix --ntasks=P --gpus-per-task=1 tests/bench_pt2ptfields_mgpu
```

It reports the slowest rank's time and the aggregate rate. Run each P as its
own job, one after the other: jobs that share a node at the same time compete
for host–GPU bandwidth and spoil the timing. The fields are bit-identical for
every P (check with `test_pt2ptfields` and `IMPACTT_FIELD_DUMP`).

## test_poisson_gpu

`openBC3D_gpu` (`src/Func/PoissonGPU.f90`, cuFFT) against `openBC3Dnew`
(FFTW):

```
tests/test_poisson_gpu NX NY NZ [NPY NPZ [DUMPFILE]]
```

- One rank: φ and the image potential φ_img for a bunch at the cathode, near
  it and far from it, and `flagImage = 0` (φ_img exactly 0); the GPU solve is
  timed. `SKIPCPU=1` skips the FFTW solve (hours on one core at 512×512×64).
  Up to 40000 grid points the image potential is also summed directly.
- Several ranks (`srun --ntasks=NPY*NPZ --gpus-per-task=1`): GPU only;
  `DUMPFILE` gets φ and φ_img as raw binary, and dumps from different
  (NPY, NPZ) are compared with `cmp` (they are identical).

φ must match to 1e-12. φ_img matches only to the conditioning of the
integrated Green's function, an 8-corner difference of large values, whose
noise is about ε·(distance/cell)³ and the same on the CPU and the GPU. The
test therefore requires 1e-11 (cathode) and 2e-10 (near), and "far" passes
if the GPU is no further from the direct sum than the CPU is.

## test_particle_gpu

Each kernel of `src/Appl/ParticleGPU.f90` and `FieldmapGPU.f90` against the
CPU routine it ports, on the same particles (1 rank, 1 GPU). Run it in a
directory holding `rfdata1001`–`1003` (two RF cells and a solenoid, type 105
discrete on-axis data) and `1T1.T7` (the type-112 map of
`examples/Sample4_2Dfield`); `python3 tests/make_test_inputs.py particle DIR`
writes synthetic ones and copies the map.

- `kick_none_gpu` (external fields only, through `extfld_dev`, types 105 and
  112 overlapping, at 5 times across an RF period) against `scatter20t_BeamBunch`.
- `kickpt2pt_gpu` against `kickpt2ptImg_BeamBunch`, image on and off.
- The drifts and the emission against `drifthalf`, `drifthalforg`,
  `driftemission` and the emission loop of `AccSimulator`.
- `singlerange_gpu` against `singlerange`; `lost_gpu` against
  `lostREC_BeamBunch` with 0 and 37 lost particles.
- The deposit (`flagpos` 0 and 1) and `gradEB_dev` against loops transcribed
  from `deposit1_Depositor` and the one-rank, `Flagbc = 1` branches of
  `gradEB_FieldQuant`.

Kicks are compared as the change of momentum, scaled by the largest change:
within about 1e-16. The loss test and `gradEB_dev` are bit-identical.

## test_bend_gpu

The field kinds 3–5 of `FieldmapGPU.f90` (types 1, 111 and 4) and the
bend-frame kernels of `ParticleGPU.f90`, against the CPU routines (1 rank,
1 GPU). It needs a quadrupole profile `rfdata111`, a dipole Enge file
`rfdata15` and a type-111 map `1T7.T7` (z header 0..0.108 m) in the run
directory. `python3 tests/make_test_inputs.py bend RUN_DIR` writes synthetic ones
(made-up smooth fields with every component nonzero; the test compares GPU
and CPU on the same data). The test points come from a fixed linear
congruential generator.

- `fld_quad` (kind 3) against `getfld_Quadrupole`: file profile, Enge and hard
  edge, skew (`anglez` 0.1), an RF quadrupole at 5 times, both edges. File
  profile and hard edge are bit-identical, Enge within 2e-16 (device `exp`).
- `fld_dipole` (kind 5) against `getfldt_Dipole`, straight and tilted pole faces,
  points on the four face lines: within 9e-16 of the largest component.
- `fld_emcart` (kind 4) against `getfldt_EMfldCart`, 5 RF phases, both ends of
  the z test, and 100 points outside the map (zero field, and the out-of-map
  counter reads 100): bit-identical.
- `kickref_Bd_BeamBunch` (the reference-particle part of `kick2tBd`/`kick2tBd0`,
  shared by the CPU routines and the GPU loop) and `k_kick_bend0` against
  `kick2tBd0_BeamBunch`, 1e5 particles, reference at 0.1 rad: the reference
  particle bit-identical, final momenta within 3e-16 of |p| (the kicks are
  1e-5 of momenta of about 45, so the comparison is scaled by the momentum).
- `rottoT_gpu`, `rotbackT_gpu`, `convEntr_gpu`, `convExit_gpu`,
  `driftbackhalf_gpu`, `drifthalfBd_gpu` against the BeamBunch routines, 1e6
  particles, and a rank with no particles: within 2.1e-16; `poscent` (a
  1e6-term sum) within 1.3e-14.
- `diagB_pass1_gpu`, `diagB_pass2_gpu` against the host loops of
  `diagnostic1avgB_Output`: the 29 sums within 7.3e-14 of the sum of the
  absolute terms, maxima equal.
- `bendpt2pt_gpu` against `kickpt2ptBd_BeamBunch`, 2e4 particles including
  z < 0, reference at 0.1 rad, with and without a CSR wake: final momenta
  within 3.1e-16 of |p|.

## compare_runs.py, compare_dirs.py

`compare_runs.py` runs one input with two executables and compares
`fort.18/24/25/26` and the timers printed at the end (use MPI builds, even at
1 rank: the serial build's timers read 0):

```
python3 tests/compare_runs.py --ref build-mpi/ImpactTexe-mpi \
    --cand build-gpu/ImpactTexe-mpi-gpu --input-dir examples/... \
    --workdir /tmp/cmp --mpi --nproc 1 --rtol 1e-6
```

`compare_dirs.py REF CAND [REF CAND ...]` compares finished run directories
(needs numpy). The GPU executable runs three ways on the same input, which
makes CPU–GPU comparisons with one executable possible: the default (all on
the GPU), `IMPACTT_PARTICLES=cpu` (particles on the CPU, Poisson solve on the
GPU) and `IMPACTT_POISSON=cpu` (also the FFTW Poisson solve).
