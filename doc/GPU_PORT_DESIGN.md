---
title: "IMPACT-T on NVIDIA GPUs: design"
subtitle: "Branch gpu-port-mvp"
date: "November 2026"
section-numbering: ""
---

# How to read this document

This is the design of the GPU port, one part per stage of the work: A1–A4 (point-to-point space charge), B1 (the Poisson solve), B2 (particles, fields and kicks), the binary particle I/O, C1 (diagnostics) and D1 (quadrupoles, 3D maps, bends, CSR, N-body in bends). The comments in the code cite these section numbers (for example "B2.4" or "D1.12"). Each part was written before it was implemented; where the implementation differs, or a check gave a result worth recording, a note says so. The overview, how to build and run, the validation and the benchmarks are in `doc/GPU_PORT.md`.

Two conventions run through all of it:

- **Port the arithmetic verbatim.** The GPU kernels use the CPU routines' expressions in the same order with the same literals, including single-precision ones such as `asin(1.0)` or `299792458.e0`, so that GPU and CPU agree to the last bits and every difference can be traced. Per-step constants (RF phases, rotation angles) are computed on the host with the CPU's expressions and passed in. The only systematic difference is that the device compiler contracts multiply-adds into FMAs, which moves the last bit of a fraction of the values.
- **Keep the CPU code as the reference.** Every GPU part is selectable at run time against the CPU code in the same executable (`IMPACTT_PARTICLES=cpu`, `IMPACTT_POISSON=cpu`), so comparisons involve no compiler difference.

The adaptive time step (`Flagmap` > 1) is on the separate branch `adaptive-step`; the parts of B2 that handle it (B2.3's `pold_d`/`fold_d`, B2.5 item 8) are described here but are only in that branch.

# Background: the CPU code

## Point-to-point space charge

- A `-15` element turned point-to-point space charge on for the whole run. Its V3 is the cutoff `r0`: pair distances below `r0` count as `r0`. `r0` was in units of `Scxlt` = c·Dt (the header Dt), as the pair loop compares it with separations in those units. Since `master` `3e19b7f` V3 is in metres and divided by `Scxlt` when it is read; this branch does the same for V3 and V5 (`8c9926b`), and the kernels still work in `Scxlt` units.
- The time loop called `kickpt2ptImg_BeamBunch`. Each step it gathered every rank's particles into `ptstot(9,npttot)` with `MPI_ALLGATHER`, and for each local emitted particle (z > 0) summed the relativistic velocity fields over all other emitted particles and then over their mirror images (at −z, with reversed βz and the opposite sign): 2N² pair evaluations per step.
- The image term ignored `Flagimg` and `zimage`; a particle did not feel its own image (the mesh solver includes it); `ptstot` was a stack array (450 MB at 6.24M particles); the gather assumed every rank holds `nptlc` particles.

## The mesh solver

- `update3Otnew_FieldQuant` → `openBC3Dnew`: forward FFT of ρ on the doubled grid (open boundaries); the integrated Green's function (`greenf1tIntnew2`), rebuilt and transformed every step; product and inverse FFT; with image charge a second Green's function shifted by `zshift` (`greenf1tIntshift2`), a second product and inverse.
- The 3D FFT is 1D FFTs (FFTPACK or FFTW) plus MPI transposes.
- Then `gradEB_FieldQuant` (E and B by finite differences), `chgdens_Depositor` (deposition) and `kick2t_BeamBunch` (interpolation, external fields, kick).
- On one core, the field solve grows about 8× per doubling of the grid, Green's-function rebuilds are about 45 % of it, and the per-particle work is almost all in `kick2t`.

# Phase 0: build and test harness

- A CMake option `USE_GPU` (requires `nvfortran`) adds `-acc=gpu -gpu=cc90 -cuda`, links cuFFT, defines `USE_GPU`, and names the executable `ImpactTexe-mpi-gpu`. CPU builds do not change.
- `nvfortran` with `USE_GPU=OFF` against gfortran, same input: `fort.18/24/25/26` agree to about 1e-6 (summation order). `USE_GPU=ON` with no GPU code executed is bit-identical to `OFF`.
- `tests/compare_runs.py` runs one input with two executables and compares `fort.18/24/25/26` with each column scaled by its own range (per-row relative errors blow up on centroid columns of a symmetric beam), and the timers.

# Part A: point-to-point space charge

## A1. Separate the physics from the loop (CPU)

`kickpt2ptImg_BeamBunch` is restructured into three steps:

1. **Sources.** One array per particle with x, y, z, βx, βy, βz, γ and an emitted flag (γ and β computed once per particle, not once per pair), gathered into a heap array with `MPI_ALLGATHERV` with real per-rank counts.
2. **Fields.** `pt2ptFields(targets, sources, image, r0) → ex, ey, ez, bx, by, bz` holds the pair loop: the direct term with the particle's own contribution excluded by global index, and the image term fused into the same pass over the sources, with no exclusion, so each particle feels its own image.
3. **External fields and the kick**, unchanged.

The image term follows `Flagimg` and stops when the bunch centroid passes `zimage`, the test the mesh branch uses (`flagimgnow` at the call site).

Checks (`tests/test_pt2ptfields.f90`, 2k particles, 1 and 4 ranks, identical): image on against an independent direct + self-image sum, 4.4e-12; image off, 2.9e-13; a single particle at rest at height z against e/(4πε₀(2z)²), exact to rounding. A mesh-only run is unchanged.

Point-to-point space charge with image charge needs one-to-one charge: with fewer macroparticles than electrons, each macroparticle's self-image is too strong by the ratio. And the image cutoff (`r0`, or `r0img`) must be near 10 nm: at 0 or a few nm it leaves the self-image of a just-emitted electron nearly uncapped, and the beam returns to the cathode while the run continues. Check the surviving count (`fort.28`, last column) of any N-body run with emission.

## A2. `pt2ptFields` on the GPU

- **OpenACC**: `!$acc parallel loop gang vector` over targets, `!$acc loop seq` over sources, the sums in private scalars; one `!$acc data` region per call. The same routine is compiled for the CPU (the directives are inert without `-acc=gpu`).
- **Pair math**: `rinv = 1/sqrt(r²)` once per pair and `rinv³`, instead of `sqrt`, `**3` and six divisions per term; `γ²/(γ+1)` and `γ/(γ+1)` precomputed per source (two extra columns of the source array). This made the CPU 1.25× faster too.
- **Profile** (Nsight Compute, 100k particles): compute-bound (SM 53 %, memory 31 %, DRAM 0 %), FP64 the busiest pipe, occupancy limited by registers. A sweep of `-gpu=maxregcount` gave +3.2 % at 64 and spilling below 56; 64 is set in `CMakeLists.txt`. Shared-memory tiling of the sources has nothing to relieve (every thread reads the same source at once, from cache), and FP32 pair arithmetic would lose µm-scale differences of positions at the metre scale.
- **Results**: GPU and CPU fields bit-identical (same compiler, same operation order, sequential inner loop); 1.98e11 pair evaluations per second on one H100 at 1M particles with the image on, 1.56e11 with it off (one source read serves two pairs with the image on; the FP64 work per pair is the same). The source count (1M or 6.24M) and the particle distribution do not change the rate.
- **CPU loop in blocks** (`275120f`): the CPU build (no OpenACC) passes over the sources in blocks of `IMPACTT_PT2PT_BLOCK` (default 4096) and runs every target over each block, keeping each target's running sums in the output arrays between blocks. The sources are added in the same order as before, so the fields are bit-identical. Unblocked, every target streams the whole source array from memory, which limits a full node at 1M sources and more (6.24M on 128 Zen 4 cores: 6.8e9 pairs/s; blocked 1.23e10, the in-cache rate). Block sizes from 256 to 16,384 give the same rate on Zen 4 and on an Apple M1 Max; 262,144 (21 MB) loses a third. The GPU loop needs no blocking: all threads of a warp read the same source at the same time, and the L1/L2 hit rates are 98–99 % with no DRAM traffic.

## A3. Several GPUs

One MPI rank per GPU (`srun --gpus-per-task=1`), each computing the fields of its own targets after gathering all sources; no other communication. Fields are bit-identical for 1, 2, 4 and 8 GPUs. 2 GPUs scale at 95 % at 1M particles. In the full code at 6.24M particles (four steps of the benchmark beam, GPU_PORT.md, Benchmarks) a step takes 235, 120, 60.5 and 30.6 s on 1, 2, 4 and 8 H100s, 96 % efficiency at 8. (Earlier 4- and 8-GPU measurements were made on a node with two thermally throttled GPUs and are superseded.) The sources go through the host for the gather on every call, about 0.5 s a step at 6.24M; a device-side gather (NCCL or CUDA-aware MPI) is the next step if smaller counts or more GPUs matter.

## A4. Switching between point-to-point and the mesh along the beamline

- `-15` switches to point-to-point when the bunch centroid passes its V4 (default 0, so an existing `-15` acts from the start, as before), with its own `r0` (V3). A new element `-19` switches back to the mesh solver at its V3.
- Up to 100 switches, stored as `(z, r0, direction)` in `swz`/`swr0`/`swflag`, sorted by z after the lattice is read, and crossed with the test the `-4` elements use (`distance`, `dzz`). Rank 0 prints a line at each switch.
- Both solvers rebuild their state every step, so switching needs no state transfer.
- **A bug this exposed**: the point-to-point kick was called with particle counts copied once before the time loop. Once a run can switch into point-to-point after mesh steps (whose load balancing moves particles between ranks), it read and wrote outside the particle array. It now passes the current `Nplocal`/`Np`.

Checks (20k particles): a switched run reproduces the pure-solver run bit for bit up to the switch, in both directions and with three switches; 4 ranks run cleanly.

# B1. The Poisson solve on one GPU

## B1.1 What `openBC3Dnew` computes (the contract to keep)

- **Inputs:** `rho(innx,inny,innz)`, this rank's interior block; `hx, hy, hz` in metres, with γ already in `hz` (beam frame); `zshift = -(zmin+zmax)·γ·Scxlt`, already γ-scaled; `flagImage`.
- **Outputs:** `rho` is overwritten with the potential φ′; `rhoImg` receives the image potential φ′_img, or 0 when `flagImage = 0`. Both are returned before the `×hx·hy·hz` that `update3Otnew_FieldQuant` applies afterwards.
- **The math.** Nodes are at z_k = zmin·γ + (k−1)·hz. G is the integrated Green's function (the integral of 1/(4πr) over one cell, divided by hx·hy·hz). φ′(i,j,k) = Σ ρ(i′,j′,k′)·G((i−i′)hx, (j−j′)hy, (k−k′)hz). φ′_img(i,j,k) = −Σ ρ(i′,j′,k′)·G((i−i′)hx, (j−j′)hy, z_k + z_k′): the potential of −ρ at the mirror position. It depends on k + k′, a correlation in z, which IMPACT-T computes as a convolution with a z-shifted G followed by a reversal of the output in z.
- **FFT sign convention** does not need to be reproduced: ρ and G are transformed with the same sign and inverted with the opposite one. IMPACT-T's inverse is normalized and cuFFT's is not, so scale once by 1/(8·Nx·Ny·Nz).

These were checked against a short numpy prototype and a brute-force sum over direct and mirror charges on an 8×6×10 grid (both 2e-14 for φ′, 2e-13 for φ′_img; leaving out the z reversal gives a 57 % error).

## B1.2 Where to cut in and the interface

- Replace all of `openBC3Dnew`, not the 1D FFT backend (the 1D FFT + transpose structure would move data between host and device dozens of times per step).
- `PoissonGPUclass` (`src/Func/PoissonGPU.f90`, compiled only with `USE_GPU`) provides `openBC3D_gpu(innx,inny,innz,rho,hx,hy,hz,myidz,myidy,npz,npy,comm2d,pztable,pytable,inxglb,inyglb,inzglb,rhoImg,flagImage,zshift)`, with the meanings of `openBC3Dnew`'s arguments of the same names.
- In `update3Otnew_FieldQuant`, under `#ifdef USE_GPU`, `use_gpu_poisson()` selects it; it reads `IMPACTT_POISSON` once (`cpu` selects FFTW). `openBC3Dnew` stays compiled in the GPU build. Both callers of `update3Otnew_FieldQuant` go through the switch.
- Timer `t_poisgpu` around the call; `t_field` still covers the whole solve.

## B1.3 GPU programming model

- CUDA Fortran device arrays (`real(8), device, allocatable`) and `!$cuf kernel do(3) <<<*,*>>>` loops: cuFFT takes device arrays directly, the GB-sized work arrays exist only on the device, and it mixes with A2's OpenACC in one executable.
- Everything on the default stream, so ordering is implicit.
- Arrays and cuFFT plans are allocated on the first call and kept; recreated only if (Nx, Ny, Nz) changes.
- Every cuFFT return code is checked.

## B1.4 Array layout and index mapping

All arrays are column-major with x fastest. N = (Nx, Ny, Nz) is the global grid, (2Nx, 2Ny, 2Nz) the doubled grid.

| Device array | Shape | Contents |
|---|---|---|
| `rd` | `real(8) (2Nx,2Ny,2Nz)` | zero-padded ρ, then G, then the inverse-FFT output |
| `rhohat` | `complex(8) (Nx+1,2Ny,2Nz)` | FFT of ρ, kept for the image pass |
| `ghat` | `complex(8) (Nx+1,2Ny,2Nz)` | FFT of G, then the product |
| `cor` | `real(8) (0:Nx+1, 0:Ny+1, 0:2Nz+1)` | corner values of the antiderivative F |
| `stage`, `blk`, `blkimg` | `real(8)` | host↔device staging and each rank's output blocks |

The complex arrays' first dimension is Nx+1, the halved real-to-complex dimension (x, the fastest Fortran index).

- **Pad:** `rd(i,j,k) = rho(i,j,k)` inside (Nx,Ny,Nz), else 0.
- **Mirror offset** of the doubled direct G (shown for x): `ox(i) = i−1` if `i−1 ≤ Nx`, else `2Nx−(i−1)`, mapping i = 1..2Nx to 0,1,…,Nx,Nx−1,…,1 (IMPACT-T's `tmp1(i) = grn(n1−i+2)` fill).
- **8-corner difference** (IMPACT-T's `grn` from `grntmp`, the same sign order, `cor(a,…)` = `grntmp(a+1,…)`): D(a,b,c) = cor(a+1,b+1,c+1) − cor(a,b+1,c+1) − cor(a+1,b,c+1) + cor(a,b,c+1) − cor(a+1,b+1,c) + cor(a,b+1,c) + cor(a+1,b,c) − cor(a,b,c).
- **Extract:** φ′ from `rd(1:Nx,1:Ny,1:Nz)`; φ′_img reversed in z, `rd(i,j,Nz+1−k)`.

## B1.5 Green's-function kernels

The antiderivative of `greenf1tIntnew2` is ported verbatim as an `attributes(device)` function (the same `atan` and `log` terms in the same order, with no guards added), so G matches the CPU's to rounding. The corner value is `recfourpi*F(...)/(hx*hy*hz)`.

- **Direct G**: corners for a = 0..Nx+1, b = 0..Ny+1, c = 0..Nz+1 at (a·hx − hx/2, b·hy − hy/2, c·hz − hz/2); then one kernel over the doubled grid sets `rd(i,j,k) = D(ox(i), oy(j), oz(k))`, fusing IMPACT-T's difference step and its three mirror fills. It reads 8 corners per point instead of evaluating F 8 times.
- **Image G** (`greenf1tIntshift2`): x and y as above; in z two blocks of corners, copied as IMPACT-T writes them: c = 0..Nz at `c*hz - hz/2 + zshift`, and c = Nz+1..2Nz+1 (m = c−(Nz+1)) at `zshift - hz*(Nz-m) - hz/2`. The doubled G uses the lower corner `clo = p` for p = k−1 < Nz, else p+1, which gives z offsets `zshift + p·hz` and `zshift + (p−2Nz)·hz`, as IMPACT-T's `tmp3(i) = x2(i−nz)`.

The direct Ĝ is real to rounding (G is even), the image Ĝ is not; both are kept complex.

## B1.6 cuFFT plans and the per-call sequence

A Fortran array `A(n1,n2,n3)` is row-major `[n3][n2][n1]` to cuFFT, so the dimensions go in reversed: `cufftPlan3d(plan, 2*Nz, 2*Ny, 2*Nx, CUFFT_D2Z)` (and `Z2D`). cuFFT then halves its fastest dimension, Fortran's x. With the dimensions in Fortran order nothing fails, but every result is wrong (the non-cubic 8×6×10 test catches it).

Per call: (1) with several ranks, gather ρ (B1.7); (2) upload, pad into `rd`; (3) D2Z → `rhohat`; (4) direct corners → `cor`, direct G → `rd`; (5) D2Z → `ghat`; (6) `ghat = rhohat*ghat*s`, s = 1/(8·Nx·Ny·Nz); (7) Z2D → `rd`; (8) extract; (9) with image charge: image corners, image G, D2Z, `ghat = −rhohat*ghat*s`, Z2D, reversed extract. Otherwise `rhoImg = 0`.

## B1.7 Several MPI ranks

Every rank gathers the whole ρ and solves it on its own GPU, so no scatter is needed afterwards.

- `MPI_ALLGATHER` of `[y0, inny, z0, innz]` per rank over `comm2d` (independent of the Cartesian rank order), then `MPI_ALLGATHERV` of the local blocks, unpacked on the host into the global grid with a bounds check. Skipped on one rank.
- Each rank extracts its own block; the image reversal is over the global z index.
- Every rank computes the same global solve, so for the same global ρ the result is **bit-identical for any number of ranks**, which the CPU path is not.

## B1.8 Checks

`tests/test_poisson_gpu.f90`, one H100, against `openBC3Dnew` with FFTW on one rank (max error / max value):

| Grid | φ′ | φ′_img, cathode | φ′_img, near | φ′_img, far |
|---|---|---|---|---|
| 8×6×10 | 1.0e-14 | 7e-14 | 1.3e-12 | 9.6e-5 |
| 32³ | 6e-14 | 3.5e-13 | 3.4e-12 | 1.1e-5 |
| 64³ | 1.8e-13 | 7e-13 | 8.7e-12 | 4.6e-6 |
| 128×128×256 | 5.2e-13 | 4.1e-12 | 8.6e-11 | 1.1e-6 |

The image potential far from the cathode looks like a disagreement but is not one: a brute-force sum over direct and mirror point charges shows the GPU and the CPU **equally far** from the true value (7.8e-5 and 7.3e-5 at 8×6×10; 9.0e-6 and 1.2e-5 at 32³). Both suffer the same cancellation in the 8-corner difference of F, with noise of about 1e-16·(r/h)³, which amplifies a 1-ulp difference between host and device `atan`/`log`. `-gpu=nofma` made it worse. The test therefore requires φ′ ≤ 1e-12, φ′_img ≤ 1e-11 at the cathode and ≤ 2e-10 near it, and far from it that the GPU is no further from the brute-force sum than the CPU. This limits `openBC3Dnew`'s own image potential far from the cathode in the same way.

Rank independence (30×26×22, uneven splits): (npy,npz) = (2,2), (1,4), (4,1) bit-identical to one rank. Full runs (20k particles, image charge on, GPU solve against FFTW in the same executable, 1 and 4 ranks, with switches): `fort.18/24/25/26` within 2e-6 (column-scaled), the largest in early rows of tiny columns.

## B1.9 Memory, speed and pitfalls

- **Memory** at 512×512×64 (doubled grid 1024×1024×128): `rd`, `rhohat`, `ghat` about 1.1 GB each, `cor` 0.27 GB, staging 0.4 GB, cuFFT work areas 1–2 GB: about 5–6 GB per rank.
- **Speed** (one H100): 0.079 s per call at 512×512×64 with image charge (the first call adds about 0.5 s of plan setup); 19 ms at 128×128×256 against 3.9 s for `openBC3Dnew` on one core; 1.8 ms at 64³ against 110–160 ms.
- **Pitfalls**, most likely first: cuFFT dimensions in Fortran order; complex arrays shaped `(2Nx,2Ny,Nz+1)`; the image potential's minus sign or its z reversal (or a reversal over the local z index); direct-G corners stopping at Nz instead of Nz+1; scaling by 1/(Nx·Ny·Nz); applying `×hx·hy·hz` twice; scaling `zshift` by γ again.
- **Tried and not kept**: pinned staging buffers, a real-valued direct Ĝ and caching Ĝ between steps were correct but gave no speed-up in full runs, because the mesh follows the beam's extent and its spacing changes every step.

# B2. The rest of the step on the GPU

## B2.1 Scope and main decisions

- **Every per-particle operation of an ordinary step moves to the GPU**: the half drifts and the emission model, the range and centroid reductions (`singlerange`), the lost-particle test, deposition (`deposit1_Depositor`/`deposit1p_Depositor`), `gradEB_FieldQuant`, and the kick in its three variants (mesh `kick2t_BeamBunch`, point-to-point `kickpt2ptImg_BeamBunch`, no space charge `scatter20t_BeamBunch`), each with the external-field lookup.
- **Particles stay on the GPU.** The host `Ebunch(ib)%Pts1` becomes a mirror, refreshed only when host code needs it (B2.3). An ordinary step moves only scalars and a small element table across.
- **No spatial decomposition of particles on the GPU path.** Every rank deposits its particles onto the whole global grid, the ranks sum ρ (`MPI_ALLREDUCE`), and every rank solves the whole grid (as B1 already does). So there is no `ptsmv2_Ptclmger`, no load balancing and no guard-cell exchange; particles never change rank except on losses or a bunch merge; the point-to-point kick, which wants particles split by count, sees the same layout. Mesh-only runs should use one GPU.
- **Rare operations stay on the host**, after syncing the particles there (B2.3): output, steering, maps, heating, rotation, collimation, bunch merging. Steps with features that are not ported run their fields and kick through the existing CPU code (B2.4).
- **Run-time switch** `IMPACTT_PARTICLES=cpu` selects the CPU particle path in the same executable (B1 mode). The GPU particle path needs the GPU Poisson solve, so `IMPACTT_POISSON=cpu` implies the CPU particle path.
- **Programming model** as in B1, with `attributes(global)` kernels for bodies too large for a CUF loop; the point-to-point pair loop stays OpenACC (B2.5 item 7).

## B2.2 Files, modules and build

| File | Module | Contents |
|---|---|---|
| `src/Appl/FieldmapGPU.f90` (new) | `FieldmapGPUclass` | device copies of the field data slots, the per-step element table, `extfld_dev` (B2.6) |
| `src/Appl/ParticleGPU.f90` (new) | `ParticleGPUclass` | particle buffers and residency (B2.3), particle kernels (B2.5), the error flag (B2.8) |
| `src/Func/PoissonGPU.f90` (B1) | `PoissonGPUclass` | the device entry point `poisson_dev` (B2.7); mesh kernels: zero, scaling, ρ sum over ranks, `gradEB_dev` |

Host module variables (`Scxlt`, `Clight`, `Epsilon0`, `Pi`, ...) cannot be read in device code; they are passed to kernels as `value` arguments.

## B2.3 Particle residency

```fortran
type ptsbuf
  real(8), device, allocatable :: p(:,:)  ! (9,cap), same layout as Pts1
  integer :: cap = 0                      ! allocated columns
  integer :: n = 0                        ! valid columns (= Nptlocal)
  integer :: state = 2                    ! 0 in sync, 1 device newer, 2 host newer
end type
type(ptsbuf), allocatable, save :: pb(:)  ! (Nbunch)
```

- The `(9,N)` layout of `Pts1` is kept, so a sync is one contiguous copy (rows: x, px, y, py, z, pz, q/m, weight, id).
- `ensure_device(ib, Ebunch(ib))` uploads when the host is newer (growing `p` if needed and taking `n` from `Nptlocal`); `ensure_host` downloads when the device is newer; `mark_host_newer` after every host write; `_all` variants loop over the bunches. Every kernel wrapper calls `ensure_device` first and marks the device newer after writing.
- Copies use `cudaMemcpy` with `9*n` elements (`Pts1` is a pointer; an array section might make a temporary).
- **Host touch points** in the time loop, each wrapped by `ensure_host_all` before and `mark_host_newer_all` after (if it writes): steering (`-1`), the linear map, instant heating, z-rotation, collimation (`-11`), the lost-particle compaction when this rank lost particles, the fallback step, diagnostics other than `Flagdiag = 2`, the host bend loop, phase, slice and restart output, the bunch merge, the final output.
- Initially `state = 2`: particles are generated or read on the host, and the first kernel uploads them.

## B2.4 The step on the GPU

After the lost-particle test the host decides `gpustep` (every input is the same on every rank, so the MPI collectives stay matched):

```
gpustep = gpupart .and. idbend.ne.1 .and. Flagerr.eq.0 .and. FlagFieldPrint.eq.0
          .and. integrator not 2 or 3
          .and. (for mesh steps: Flagbc.eq.1 .and. flagspc.eq.1 .and. flagazmuth.eq.0
                 .and. no wake or DWA)
          .and. (all elements in range have a supported kind, B2.6)
```

If `gpustep` is false, the existing code of the step runs unchanged between `ensure_host_all` and `mark_host_newer_all`, and the first such step prints the reason. (In the design all conditions applied to every step; in the implementation the mesh conditions apply only to mesh steps, so a point-to-point or space-charge-free step does not fall back because the input lacks the `-5` element.)

The GPU block replaces the fields, kick and final drift:

1. If `zmax ≤ 0`, skip to step 6.
2. Build the element table for the elements in range at time t (B2.6).
3. **Mesh**: `update_CompDom` and the global (Nx, Ny, Nz) as before, but no `ptsmv2`, no load balancing, and none of the host allocations and zeroing of `chgdens`, `exg`…`bzg`, the DWA arrays and `tmppot` (at 512×512×64 zeroing them costs more than the GPU step). Then zero the device fields, and for each bunch: deposit, sum over ranks, scale, `poisson_dev`, `gradEB_dev` of φ (with `tmpflag` from `gbetz`), and of φ_img when the image is on; then the mesh kick.
4. **Point-to-point**: the sources, the gather (several ranks), `pt2ptFields_dev`, the kick.
5. **No space charge**: the kick with external fields only.
6. Final drift and emission.
7. Check the device error flag (B2.8).

## B2.5 Kernels

All grid kernels use the global grid with no ghost layers (the CPU code's one-rank case: `jadd = kadd = 0`). Arithmetic is ported verbatim, with per-call scalars (`hxi`, `tmp2`, `betC`, ...) computed on the host with the CPU's expressions.

1. **Drifts and emission.** Start of step: `drifthalf_BeamBunch` (z > 0 only) or `drifthalforg_BeamBunch`, by `flagcathode`. End of step with `flagcathode = 1`: one fused per-particle kernel doing the half drift, the emission drift and the first-order emission correction in the CPU's order.
2. **`singlerange_gpu`**: one reduction for min and max of x, y, z and the sums of the six coordinates, then `singlerange`'s three `MPI_ALLREDUCE`s on the host. Min and max are exact; the sums differ in the last bits.
3. **Lost test**: a kernel evaluates `lostREC`'s (or `lostDFO`'s) predicate and counts losses. With losses on this rank, the particles go to the host and the original routine compacts them; without, a host routine does what `lostREC` does after its loop (`MPI_ALLREDUCE`, `Current = Current*nptot/Npt`, `Npt = nptot`). Every rank calls one `MPI_ALLREDUCE` either way, and `Current` is updated every step as on the CPU.
4. **Deposition**: one thread per particle, the same index and weight formulas, 8 FP64 `atomicAdd`s into `rho_d`; bounds failures set the error flag. With several ranks `allreduce_rho_gpu` sums ρ through the host. Then scaling by `tmp2` (after the sum; changes only rounding). Atomics make deposition non-deterministic in the last bits.
5. **`gradEB_dev(phi, flag)`**: one kernel over the grid with the stencils of the one-rank `Flagbc = 1` branches (one-sided second order at the edges, central inside), accumulating `ex += gammaz*egx`, `by += gammaz*egx*betC`, `ey += gammaz*egy`, `bx -= gammaz*egy*betC`, `ez += egz` in the CPU's order. There is no space-charge `bz`.
6. **Kicks**, one thread per particle: mesh (`kick2t_BeamBunch`, z > 0 only, the same 8-term trilinear sums, `extfld_dev`, the Boris update), point-to-point (`coefE*` the pair fields plus the external fields, z > 0 only), and none (`scatter20t_BeamBunch`, every particle).
7. **Point-to-point**: the sources (with the per-source γ terms) built on the device; on one rank no host traffic at all; on several, `MPI_ALLGATHERV` through the host. `pt2ptFields_dev` has the same body as `pt2ptFields` with `device` dummies and no data region; the two carry a "keep identical" comment.
8. **Adaptive moments** (`adaptive-step` branch only): `savemom_gpu`/`kickmom_gpu` with the per-id history on the device.

## B2.6 Field maps on the GPU

- **Field data slots.** `fldmp(1..Maxoverlap)` are reused during the run as elements come into range; the device mirrors them. `read1tdata_Data` (105 with file ID > 1000) is copied to `fcoef_d(4,10002,Maxoverlap)`; `read2t_Data` (112) is packed into three device arrays with a per-slot offset (device code cannot index host arrays of derived types). The upload hook sits right after each read, and the upload is chosen by which reader ran.
- **Element table**, rebuilt on the host every step for the elements in range, in increasing order, copied to small device arrays: a `kind` (1 = 105 with discrete data, 2 = 112, 0 = no field, −1 = not ported, which makes the step fall back), the z range, the slot, and the per-kind constants.
- **Per-step constants on the host** with the CPU routines' exact expressions, **single-precision literals included**: for 105 (`getfldt_SolRF`), `clite = 299792458.e0` (299792448 as stored) and `pi = 2*asin(1.0)` (3.14159274), then `ww`, `xlrep`, `theta0`, `cos(ww*t+theta0)`, `sin(...)`; for 112 (`getfldt_EMfldCyl`) the same with its expressions and the mesh spacings. Fixing the literals would move results by about 3e-8 relative.
- **`extfld_dev(x, y, z, ...)`** sums the elements whose range contains z, in table order, as `getfldt_BeamLineElem`'s `extfld = extfld + tmpfld`. Each kind's body is the rest of the CPU routine, verbatim. The 112 `stop` for a radius outside the map becomes an error code.

## B2.7 Poisson solve: a device entry point

`poisson_dev(nx,ny,nz,src_d,phi_d,phiimg_d,hx,hy,hz,flagImage,zshift,scale)` is B1.6 steps 2–9 on device arrays, extracting into `phi_d`/`phiimg_d` with the `×hx·hy·hz` of `update3Otnew_FieldQuant` when `scale` is set. `openBC3D_gpu` keeps its interface and calls it; `test_poisson_gpu` reproduces B1's table exactly.

## B2.8 Errors, timers, counters

- **Error flag**: a device integer plus four values (code, particle index, offending index), set with `atomicCAS` (first error wins); the write is skipped. Codes: 1 deposit bounds, 2 kick bounds, 3 112 radius, 6 CSR histogram index (D1.11). The host checks once per step and prints the CPU's message.
- **Timers** (`cudaDeviceSynchronize` before each reading): `drftgpu` (drifts, emission, range, losses), `depgpu` (deposit, sum, scaling), `gradgpu`, `kickgpu`, `syncgpu` (particle copies), `poisgpu`.
- **Counters**: host↔device particle copies and fallback steps, printed at the end of the run.

## B2.9 Memory and speed

- At 6.24M particles: particles 0.45 GB; point-to-point arrays 1 GB (point-to-point steps only); mesh arrays 1.1 GB at 512×512×64; B1's work arrays 5–6 GB. Measured peak: 5.5 GB at 512×512×64, 10 GB at 256×256×512.
- 6.24M particles, 1 pC, a 2 m photoinjector (2611 steps) on one H100: 167 s at 512×512×64 before C1, of which 40 % was the host diagnostics; 97 s after C1. 1M particles to 25 mm: 17 s against 458 s with the CPU particle path (27×).

## B2.10 Checks

**Unit tests** (`tests/test_particle_gpu.f90`), each kernel against the CPU routine on the same input:

| Kernel | Input | Result |
|---|---|---|
| `extfld_dev` (through the no-space-charge kick) | 1e5 particles, 105 (two cells and a solenoid) and 112 overlapping, 5 RF phases | ≤ 7.3e-15 of the largest kick |
| point-to-point kick | 2e4 particles, image off and on | ≤ 2.9e-13 / 1.8e-14 of the largest kick |
| drifts and emission | 2e5 particles straddling z = 0 | ≤ 3.7e-16 |
| `singlerange_gpu` | 1e6 particles | min and max exact, centroid 4.6e-15 |
| lost test | 0 and 37 lost | identical counts, compaction and `Current` |
| deposit (`flagpos` 0 and 1) | 3e5 particles | 6.0e-15 of max ρ, total charge 2e-16 |
| `gradEB_dev` | random smooth φ | bit-identical |

**Full runs** (20k particles, one-to-one charge, image charge on; GPU against the CPU particle path in the same executable; column-scaled `fort.18/24/25/26`): to 6 mm, to 25 mm (the image switches off), to 2 m, with element 112, at 4 ranks, with three solver switches, with steering and a `-2` output in the run: 5e-8 to 4.4e-6, the largest in early rows of columns with a tiny range; two GPU runs differ as much (atomics). Final particles agree to 1e-8 of the rms. Zero fallback steps, one upload at the start.

## B2.11 Order of implementation

The Poisson split and the field maps first (standalone), then residency and the kernels with their unit tests, then the time-loop integration and the full runs, then the timings.

## B2.12 Pitfalls

1. A host write to `Pts1` with no `mark_host_newer` (or a kernel with no device-newer mark): the stale copy wins silently.
2. `Nptlocal` changed on the host (losses, collimation, merge, a fallback step) without the device count following: `ensure_device` must take `n` from `Nptlocal`.
3. "Fixing" the single-precision literals.
4. Leaving the per-step host allocations and zeroing on the GPU path.
5. Timers read without `cudaDeviceSynchronize`.
6. A host module variable used in device code.
7. Mismatched MPI collectives when ranks take different branches.
8. A field slot not refreshed after the host re-reads it, or the element table built before the step's file reads.
9. Expecting bit-identical reruns or rank-count independence on the GPU path.
10. **Host state that only host steps set** (found in D1): `Nxlocal`/`Nylocal`/`Nzlocal`, `kadd` and `jadd` are set by the host mesh step, not the GPU step. A host routine that runs after GPU steps (the host bend loop) must take them from the current domain.

# Binary particle I/O

- A `-2` element with **V4 > 0** writes `fort.<Bmpstp>` as a stream (`access='stream'`): an `int64` count, then count × 9 `float64` (the text dump's columns, units and order; `Bnseg` still thins it). V4 because lume-impact and other tools write only V1–V3 of a `-2`.
- **`Flagdist = 168`** reads that stream as `partcl.data`, each rank its own slice; otherwise as 166.
- Both print their wall time. At 6.24M particles: dump 0.6 s against 8.8 s for text; read-in 1.1 s against 28 s. The binary dump is bit-identical to a text dump written by gfortran and read back; nvfortran's `g0` prints about 16 digits, so on the GPU executable only the binary dump is exact.

# C1. Diagnostics on the GPU

`diagnostic1avgZ_Output` (`Flagdiag = 2`) sums its moments on the device when the run is device-resident: `diag_zsum_gpu` (sum of z), `diag_pass1_gpu` (the six drifted sums), `diag_pass2_gpu` (the 41 sums, six maxima and the maximum r², in four reduction kernels). Everything after the sums (the `MPI_REDUCE`s, the formulas, the writes) is unchanged, so `fort.18` and `fort.24`–`32` keep their format. The first diagnostic (before the particles are uploaded) and other `Flagdiag` values use the host loop.

At 6.24M particles this removed a 450 MB copy and three host passes every 5 steps: the loop of a 2611-step run fell from 143 s to 73 s at 512×512×64. Against the CPU particle path, the column-scaled differences are at the GPU's run-to-run level, except the energy spread early in the gun (a few 1e-6 of the column: `sqrt(<γ²> − <γ>²)` of two nearly equal numbers, summed in a different order).

# D1. Quadrupoles, 3D maps, bends, CSR, N-body in bends

Validated on a beamline with nine type-1 quadrupoles, a type-111 RF deflector and four type-4 dipoles (set to ±0.0257 T as a chicane for the tests, so the reference particle turns by 33.7 mrad per dipole).

## D1.1 Scope and why

Before D1, three things sent work to the host: every step with a type-1 quadrupole or a type-111 map in range fell back (about 1.4 s per step at 6.24M particles), and the type-4 bend-frame loop ran on the host (about 1.5 s per bend step). On the test beamline that was about 90 % of the run. D1 ports all three, plus the bend diagnostic, CSR in the bend loop, and point-to-point space charge in the bend frame (new: the CPU bend loop had only the mesh solver). Not in scope: `Nbunch > 1` in a bend (host loop); outputs inside bends and fields overlapping a dipole (inside a bend only the dipole's own field acts and no per-step element fires, as on the CPU).

## D1.2 Field kinds: table and slots

The element table widens to `ti(6,*)`, `tr(16,*)`, and `extfld_dev` gets three kinds:

- **Kind 3, quadrupole** (`getfld_Quadrupole`): mode 1 (file profile, `Param(3) ≥ 100`, through `fcoef_d`), mode 2 (Enge, `0 < Param(3) < 100`), mode 3 (hard edge); skew angle and RF quadrupole phase as host-computed `cos`/`sin`. The three fringe routines are inlined verbatim, including their odd details (`if(iz.lt.0) iz = 1`, single-precision `1.5*dd` and `1.0+exp(s1)`). The quadrupole's `4*asin(1.0d0)` is double precision, unlike type 105's.
- **Kind 4, type 111** (`getfldt_EMfldCart`): the six complex arrays packed per slot on the device; the time factor `dcmplx(cos(ww*t+theta0), sin(ww*t+theta0))*escale` computed on the host (with the single-precision `asin(1.0)`); the strict in-element test `zedge < z < zedge+L`, the index and weight formulas, and the eight complex terms per component in the CPU's order. **Outside the map's grid** the CPU reads outside its arrays (undefined); the GPU returns zero field and counts the point (reported at the end of the run).
- **Kind 5, dipole in ordinary steps** (`getfldt_Dipole`): after a bend exit the beam tail is still in the dipole's z range for a step or two. The CPU evaluates the dipole there with the lab z, which in practice gives zero; ported as it is. The dipole data (`Fcoeft(1:22)`) and the host-computed `sin`/`cos` of the face angles go to `dipc_d`.

## D1.3 Quadrupole and 111 integration

Nothing else changes for kinds 3–5: on the test beamline the fallback steps went from 1678 (quadrupoles), 2041 (deflector) and 3358 (bend leg) to 0.

## D1.4 The host bend loop

The GPU loop does what the host loop does, in the same order:

- **Entry**: `convEntr` (z shifted to the bend frame, the reference particle reset), `driftbackhalf`, `t -= Dt/2`.
- **Each bend step**: `drifthalfBd`; `ptref = refptcl`; `rottoT` (rotate x, px, z, pz by the reference angle; the range of the rotated beam; `poscent`); with current: the domain from the rotated range, deposit, Poisson solve with no image and `tmpflag = 0`, `gradEB`, CSR if on, then `kick2tBd`; without: `kick2tBd0`. The kick: the reference particle's own kick with the angle **before** it, then per particle the interpolated mesh fields, the position rotated back, the field of **this dipole only**, the field rotated into the beam frame (no space-charge `bz`), and the Boris update. Then `rotbackT` with the pre-kick angle, `drifthalfBd`, the reference advance, and `diagnostic1avgB_Output` (`fort.34`–`38`).
- **Exit**: `convExit`.
- No lost-particle test, no adaptive step and no other element inside the loop.

## D1.5 The GPU bend loop

Decided once per bend: `gpubend = gpupart .and. Nbunch.eq.1 .and. Flagbc.eq.1`. Otherwise the host loop runs (one line names the reason; counted in the end-of-run line). On the GPU:

1. **Entry**: `convEntr_gpu` (`zsh = zorgin/Scxlt` computed on the host), `driftbackhalf_gpu`, the reference-particle lines on the host verbatim. The host arrays `exg`…`bzg` are not reallocated.
2. **Each step**: `drifthalfBd_gpu`; `cs`, `ss` from `ptref` on the host; `rottoT_gpu` (reads all four values before writing; ±1e20 on a rank with no particles); with current, the domain from the rotated range, `mesh_fields_gpu` with `gbetz = 1` (the bend loop's `tmpflag = 0` regardless of the momentum's sign), no image; CSR (D1.11); `kickref_Bd` on the host; `kick_bend_gpu`; `rotbackT_gpu` with the pre-kick angle; `drifthalfBd_gpu`; the diagnostic on the device (D1.7).
3. **Exit**: `convExit_gpu`; the device copy is the newer one.

**`refptcl` stays a host variable**: it is 6 numbers, the frame and the exit test depend on it, and its host code is kept line for line.

## D1.6 Kernels and host helpers

- **`kickref_Bd_BeamBunch`** (host): the reference-particle half of `kick2tBd`/`kick2tBd0`, moved into its own routine that both CPU routines now call, so the CPU and GPU loops share one copy (the CPU results are bit-identical before and after).
- **`k_kick_bend0`, `k_kick_bend_mesh`**: one thread per particle, every particle (no z > 0 test); the mesh variant interpolates the five fields in `kick2tBd`'s term order (`exn` and `eyn` carry outer parentheses in the CPU code; kept); both finish in a device routine with the dipole field from `fld_dipole`, the rotation into the beam frame (two expressions, with and without the space-charge term, as the CPU writes them), and the Boris update. Two kernels rather than a mode argument, so no unallocated mesh array is passed in the no-current case.
- `rottoT_gpu`, `rotbackT_gpu`, `convEntr_gpu`, `convExit_gpu`, `drifthalfBd_gpu`, `driftbackhalf_gpu`: CUF kernels with the CPU's expressions.
- Timer `bendgpu` for the whole GPU bend loop.

## D1.7 `diagnostic1avgB_Output` on the GPU

As C1: pass 1 (sums of z, pz and γ) and pass 2 (29 sums, six maxima, the maximum r) as device reductions; the `MPI_REDUCE`s, formulas and writes unchanged.

## D1.8 Errors, counters, memory and speed

- Error code 2 is reused by the bend kicks; 6 is the CSR histogram index; the 111 out-of-map count is a counter, not an error.
- The end-of-run line gains the number of bends on the GPU and on the host and the 111 out-of-map count.
- Device memory grows only by the 111 maps (46 MB for a 33×33×439 map) and the dipole data.
- **Measured** (6.24M particles, 128³, one H100): the leg with six quadrupoles and four dipoles (10,980 ordinary steps, 4036 bend steps) in 186 s; a bend step costs 10.7 ms, an ordinary step 10–13 ms. Before D1 the same leg was about 3 h. With CSR in all four dipoles: 197 s.

## D1.9 Checks

**Unit tests** (`tests/test_bend_gpu.f90`), each against the CPU routine:

| Kernel | Result |
|---|---|
| `fld_quad`, modes 1 and 3, skew, RF quadrupole at 5 times | bit-identical |
| `fld_quad`, mode 2 (Enge) | 2.0e-16 (device `exp`) |
| `fld_dipole`, straight and tilted faces, points on the face lines | ≤ 9e-16 of the largest component |
| `fld_emcart`, 5 RF phases, both ends of the strict test | bit-identical |
| 111 out of map, 100 points | zero field, counter 100 |
| `kickref_Bd` and the no-current bend kick, 1e5 particles, reference at 0.1 rad | reference particle bit-identical; final momenta ≤ 3.1e-16 of \|p\| |
| rotations, conversions, bend drifts, 1e6 particles, an empty rank | ≤ 2.1e-16; `poscent` 1.3e-14; ±1e20 on the empty rank |
| bend diagnostic passes, 1e6 particles | sums ≤ 7.3e-14 of the sum of absolute terms, maxima equal |

The criterion for kicks is relative to the momentum, not to the kick: momenta are about 45 and kicks about 1e-5, so the last bit that an FMA moves is 1e-10 to 1e-13 of a kick.

**Full runs** (20k particles; GPU against the CPU particle path; the GPU's own rerun as the noise floor): the leg with mesh, without current, with CSR on in one to all dipoles, with N-body in a dipole, and at 4 ranks; the run from the cathode through three quadrupoles; the run through the deflector at the crest and at the streak phase. All at the GPU's run-to-run difference (column-scaled ≤ 7.3e-8; final particles ≤ 2.3e-10 of the rms, 2.0e-9 with N-body, 2.3e-7 with CSR in every dipole, where CSR amplifies step-to-step differences).

## D1.10 Order of implementation

The field kinds first (independent of the bend loop), then the bend kernels with their unit tests, then the loop integration, then the timings, then CSR, then N-body in bends.

## D1.11 CSR in the GPU bend loop

IMPACT-T's own 1D CSR model (`csrwakeTrIGF_FieldQuant`) is kept on the host: the wake works on Nz points, and keeping it verbatim leaves nothing to validate in it. On the device: the line-density histogram on the rotated z with the CPU's `iizz`/`eeff` formulas and FP64 atomics into `Nz+1` bins (error code 6 outside 1..Nz), copied to the host's `denszlc`; then the host code from the rank sums through the wake, unchanged; then `ez_dev(i,j,k) += ezwake(k)` (on the GPU's global grid the CPU's `kst` arithmetic reduces to k). The bin width keeps the single-precision `1.0000001`. Timer `csr`. Physics check: CSR on against off changes the energy spread by 3.6e-2 and the pz spread by 7.4e-2 (relative, 20k), as expected; at 6.24M the mean γβz drops by 1.5 keV over the four dipoles.

## D1.12 N-body in the bend frame

- **Selection**: a bend uses point-to-point space charge when it is on at the bend entrance (`bendpt2pt = flagpt2pt`), that is, a `-15` before the dipole and a `-19` after it. Rank 0 prints one line per bend. **This changes inputs whose `-15` window spans a bend** (before, the bend used the mesh).
- **The step**, in the rotated frame: every particle is a source and a target (`k_build_src_all`; in the bend frame z is measured from the entrance, so the z > 0 emitted test would drop the bunch tail); image off; `r0` from the last `-15`; the fields from `pt2ptFields_dev` (gather through the host on several ranks); a kick variant that uses `coefE*` the pair fields in place of the interpolated ones, then the dipole field and rotation as in the mesh case, **plus the space-charge `bz`** (the mesh has none; the pair field does). With CSR on, the wake is added at each particle with the histogram's linear weights. `update_CompDom` is still called (cheap; it keeps the range valid for CSR); the deposit, solve and `gradEB` are skipped.
- **The CPU routine** `kickpt2ptBd_BeamBunch` does the same on the host, so the CPU path remains the reference.
- **Checks**: two particles side by side at γ = 57 against the analytic transverse field, 1.7e-11; GPU against CPU kick, final momenta ≤ 3.1e-16 of \|p\|; 5k particles through a dipole, GPU against CPU 2.0e-9 of the rms (GPU rerun 4.7e-10), also at 4 ranks; N-body against the mesh in a dipole at 200k particles: x rms equal to 1e-5, x emittance 0.6 % and the longitudinal spreads 1.7–2.5 % larger with N-body (what a 64×64×128 mesh smooths).
- **Cost**: about 1.1–1.3e11 pair evaluations per second at 200k (the GPU is not full), 235 s per step at 6.24M on one GPU (measured outside a bend), so about 65 h for a dipole of about 1000 steps: a reference calculation.

## D1.13 Pitfalls

1. Using the reference angle after the reference particle's kick (the rotation back and the field rotation use the angle before it).
2. Moving `refptcl` to the device or re-deriving it.
3. An in-place rotation that writes x before reading it for z.
4. Taking `tmpflag` from the sign of the momentum in the bend (the CPU passes 0).
5. Adding other elements' fields in the bend kick (the CPU uses only this dipole's).
6. Adding a lost-particle test, a `-2` or an adaptive step inside the GPU bend loop.
7. "Fixing" the dipole in ordinary steps by subtracting zedge.
8. Computing per-element `sin`/`cos`/`atan` per particle on the device.
9. "Fixing" single-precision literals (`asin(1.0)` in kinds 1, 2, 4; `1.0000001`; the fringe literals).
10. The z > 0 emitted test in the bend frame's source build.
11. A rank with no particles in the rotated range reduction.
12. Host state that only host mesh steps set (`Nxlocal`, `kadd`, `jadd`; B2.12 item 10): before the fix, a host bend after GPU steps allocated its field arrays at the wrong size (heap corruption) and put the CSR wake on the wrong mesh planes.

# E. Later additions (November 2026)

## E1 Type 101 (DTL) on the GPU (`7c5eeff`)

Field kind 6 of `FieldmapGPU.f90` (`fld_dtl`) is `getfldt_DTL` on the device. The harmonics cos(kθ) and sin(kθ) come from the previous harmonic by angle addition, so a particle needs one `sin` and one `cos` instead of eight per harmonic; the sums, the prefactors, the quadrupole-gradient zones and the off-axis expansion keep the CPU routine's order. The per-step constants (`tmpcos`, `tmpsin`, `xlrep`, and the single-precision `clite` and `pi`) are computed on the host and passed in the element table, as for type 105. The Fourier coefficients go to `fcoefF_d(401,Maxoverlap)` through a new `which = 5` of `fldslot_upload`, called after `read1t_Data` for type 101; `build_active_table` refuses a type-101 slot that was not uploaded. `tests/test_dtl_gpu.f90`: ≤ 1.3e-15 of each component's maximum on nine cells, 5 times across an RF period.

## E2 Lost particles removed on the device (`5527e28`, `94572e6`)

When `count_lost` finds a loss, `compact_lost` removes the lost particles without copying the beam: `k_lost_flag` evaluates the host predicate once per particle and counts the survivors of each block of 256, `k_lost_scan` makes the block offsets (an exclusive prefix sum) and finds the first block with a loss, and `k_lost_scatter` writes the survivors of the moved blocks into a second buffer at block offset + rank within the block. The survivors keep the host routines' order (a stable compaction), so the particle array is the one `lostREC_BeamBunch` / `lostDFO_BeamBunch` would leave. Then whichever moves less: the unmoved prefix is copied into the second buffer and the two buffers are swapped (`move_alloc`), or the moved part is copied back. `Nptlocal`, `Npt` and `Current` are updated as on the host, with one `MPI_ALLREDUCE`; only two integers come back to the host. The second buffer (72 bytes a particle) is allocated at the first loss. `IMPACTT_LOSS=host` and `IMPACTT_LOSS=copyback` keep the old path and the copy back, for comparisons; timer `lostgpu`.

## E3 `Flagdiag = 1` moments on the device (`84237ae`)

`diagnostic1avg_Output` takes an optional `devok`. When the caller sets it (the every-5-steps call on the GPU path), the six first-pass column sums (`diag1_pass1_gpu`) and the 35 second-pass sums, six maxima and the largest r² (`diag1_pass2_gpu`, four CUF reductions with the host loop's per-particle expressions) come from the device particles; the MPI reductions, the arithmetic and the writes of `fort.18` and `fort.24-32` are the host routine's. The sums differ from the host's only in summation order (`fort.18` column 7, the rms energy spread, varies by up to 1.1e-7 of its maximum run to run, through ⟨γ²⟩ − ⟨γ⟩²). The t = 0 call, `Flagdiag = 3` and `105` stay on the host. `IMPACTT_DIAG=host` keeps the host path.
