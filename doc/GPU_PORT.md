---
title: "IMPACT-T on NVIDIA GPUs: changes, validation and benchmarks"
subtitle: "Branch gpu-port-mvp"
date: "November 2026"
---

# Summary

The branch `gpu-port-mvp` (on top of `master` at `3e19b7f`) runs the whole time step of IMPACT-T on one or more NVIDIA GPUs: emission, drifts, deposition, the Poisson solve with image charge, the field gradients, the external fields of the main element types (including the DTL), the kick, the removal of lost particles, the diagnostics, the type-4 bend loop with CSR, and the point-to-point (N-body) space charge, including inside bends. Particles stay on the GPU for the whole run, and any step that uses a feature that is not ported falls back to the existing CPU code for that step only. The CPU build is unchanged apart from the bug fixes and one speed change (the N-body pair loop in cache-sized blocks, with bit-identical results) listed in [Changes to the existing code](#changes-to-the-existing-code), and the optional additions in [Input additions](#input-additions); without `USE_GPU` no GPU code is compiled.

On a 1 pC photoinjector beamline (an X-band gun and three linac sections, cathode to 3.1 m, 49,390 steps), one NVIDIA H100 runs a 6.24-million-particle simulation (one macroparticle per electron) in 264 s at 64³ and 336 s at 128³. One 128-core node of AMD EPYC 9534 CPUs takes 1253 s and about 4650 s for the same runs, and the GPU results agree with the CPU results to the eight digits IMPACT-T prints (rms sizes, emittances, bunch length, energy and energy spread at the end of the beamline). Details are in [Validation](#validation) and [Benchmarks](#benchmarks).

While checking the GPU port against the CPU code, we found and fixed a bug in the CPU code's particle exchange between ranks (`ptsmv2_ptclmger`) that corrupts particles at the start of emission in runs with a few million particles on up to 128 ranks; upstream `master` has it too. A cross-check against IMPACT-Z on the same lattices found two more bugs that `master` has: the DTL field (type 101) leaves out Ez''' and has cos for sin in the odd Ez'' terms, and the type-4 bend exit drops the reference particle's overshoot, so the bunch arrives up to one step late after every bend. These and the other fixes to the existing code are described in [Changes to the existing code](#changes-to-the-existing-code). Each is a separate commit and can be taken on its own.

**Existing inputs.** An `ImpactT.in` written for `master` runs unchanged: no header line, element type or parameter was removed or reordered, every addition uses a slot `master` ignores, and its default reproduces the old behaviour. `-15` reads its cutoff r0 in metres, as `master` does since `3e19b7f`. Results change where a fix applies (point-to-point space charge with image charge, type-4 bends, DTLs, read-in bunches straddling z = 0, and the CPU particle-exchange overflow); see [Changes to the existing code](#changes-to-the-existing-code). Every example in `examples/` gives byte-identical output to `master` where no fix applies ([Validation](#the-cpu-path-against-upstream)).

The design of each part, with the reasoning behind it, is in `doc/GPU_PORT_DESIGN.md`. The comments in the GPU code cite its section numbers (for example "B2.4" or "D1.12").

# Branches

| Branch | Contents |
|---|---|
| `gpu-port-mvp` | `master` plus the GPU port, the fixes and these documents (the commits are listed in [Appendix A](#appendix-a-commits)) |
| `adaptive-step` | an optional adaptive time step (`Flagmap` = n > 1, tolerance 10⁻ⁿ) and its GPU part, two commits on top of `gpu-port-mvp` as of 2026-10-27 (on `master` `78c029b`; not yet moved onto the current `gpu-port-mvp`). A separate contribution; nothing in `gpu-port-mvp` depends on it. |

The work was done in stages, named A1–A4 (point-to-point space charge), B1–B2 (mesh solver and particles), C1 (diagnostics) and D1 (quadrupoles, 3D maps, bends, CSR, N-body in bends). The stage names appear in the commit messages and in the code comments. Later commits (November 2026) port the DTL, the loss removal and the `Flagdiag = 1` moments, and add the fixes found against IMPACT-Z.

# Building

GPU build (tested with the NVIDIA HPC SDK 26.3 with CUDA 13.1, Open MPI 5.0.8 built with it, FFTW 3.3.11, on H100 SXM GPUs):

```
cmake -S src -B build-gpu -DUSE_MPI=ON -DUSE_FFTW=ON -DUSE_GPU=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<fftw prefix>
cmake --build build-gpu -j 16          # -> build-gpu/ImpactTexe-mpi-gpu
```

- `USE_GPU=ON` requires `nvfortran` and stops with a message otherwise. It adds `-acc=gpu -gpu=cc90 -gpu=maxregcount:64 -cuda`, links cuFFT, compiles the three GPU source files and defines `USE_GPU`. For another GPU generation, change `cc90` in `src/CMakeLists.txt`.
- `-gpu=maxregcount:64` was chosen by a sweep over the N-body kernel's register cap (+3 % over the default; much lower caps spill).
- CPU builds (`USE_GPU=OFF`, gfortran or nvfortran, serial or MPI) are built as before and contain no GPU code; their source changes are listed in [Changes to the existing code](#changes-to-the-existing-code).
- `src/CMakeLists.txt` needs CMake 3.13 or newer (it was 3.5), for `target_link_options`.
- The tests in `tests/` are not part of the CMake build; `tests/build_test.sh BUILD_DIR tests/<test>.f90` links one against an existing build (see `tests/README.md`).

# Running

Run one MPI rank per GPU with `srun` (with `mpirun` the ranks may not get distinct GPUs):

```
srun --mpi=pmix -n 1 --gpus-per-task=1 ImpactTexe-mpi-gpu
```

The input files are unchanged. The GPU path is the default in the GPU executable; two environment variables select the CPU code in the same executable, which is how the CPU–GPU comparisons below were made without a compiler difference:

| Setting | Particles and kicks | Poisson solve |
|---|---|---|
| (default) | GPU | GPU |
| `IMPACTT_PARTICLES=cpu` | CPU (the existing code) | GPU |
| `IMPACTT_POISSON=cpu` | CPU | CPU (FFTW) |

At the start of a run rank 0 prints `particles and fields on the GPU`, and one warning for each setting that sends steps to the CPU or is untested on the GPU: `Flagerr ≠ 0` (every step on the CPU), `Nbunch > 1`, `Flagdiag` other than 1–3, `Rstartflg = 1`, and, with more than one rank, that the mesh Poisson solve is repeated on every GPU. The first step that falls back to the CPU prints the reason once (for example `CPU fallback for this step: Flagerr or FlagFieldPrint`). The run ends with one summary line:

```
GPU particle path: host->device copies 1, device->host copies 1, CPU fallback steps 0,
bends on the GPU 4, on the host 0, 111 out-of-map points 0
GPU particle path: loss steps on the device K (L particles), on the host 0 (0 particles);
Flagdiag = 1 moments on the device D times
```

- Host→device copies are 1 at the start plus one for each step that ran on the host (fallbacks, `-1` steering, `-11` collimation, a host bend).
- Device→host copies happen for outputs only (`-2` dumps, the final `fort.50`, and diagnostics other than `Flagdiag = 1` and `2`).
- Lost particles are removed on the device (a stable compaction: the survivors keep the order the host routines `lostREC_BeamBunch` / `lostDFO_BeamBunch` leave them in), and the `Flagdiag = 1` moments (`diagnostic1avg_Output`, the same `fort.18`, `fort.24-32`) are summed on the device; neither copies the beam. The second summary line counts them (rank 0). `IMPACTT_LOSS=host` and `IMPACTT_DIAG=host` bring back the host versions (a download and, for losses, an upload at the next step), for comparisons. The device compaction allocates a second particle buffer (72 bytes a particle) at the first loss, scatters the survivors into it and then swaps the two buffers (or, when the first loss is late in the beam, copies the moved part back, whichever moves less); `IMPACTT_LOSS=copyback` always copies back, for comparisons. A third summary line counts the swaps.
- `111 out-of-map points` counts particles that were outside a type-111 map's grid (they get zero field; the CPU code reads outside its arrays there).

**Timing runs.** Any non-empty `IMPACTT_NODUMP` skips the initial (`fort.40`) and final (`fort.50`) particle text dumps, in every build. At 10⁸ particles each would be about 22 GB, and the `fort.50` write is inside the `integ` timer.

**CPU N-body.** On the CPU, the pair loop takes the sources in blocks of `IMPACTT_PT2PT_BLOCK` particles (default 4096, 320 kB) so that a block stays in cache while every target on the rank reads it (see [Changes to the existing code](#changes-to-the-existing-code)). The fields do not depend on the block size. Any size from 256 to 16,384 gave the same speed on an AMD EPYC 9534 node and on an Apple M1 Max, so the default should not need changing.

The timer table at the end has new entries: `pt2ptfields`, `poisgpu`, `drftgpu`, `depgpu`, `gradgpu`, `kickgpu`, `syncgpu`, `bendgpu`, `lostgpu` (the removal of lost particles, device or host), `csr`. The device `Flagdiag = 1` moments are in `diag`, as the host ones.

**More than one GPU.** Particles are split over the ranks by count, not by position, and each rank deposits onto the whole grid; the charge density is summed over the ranks and every rank solves the whole grid on its GPU. Mesh runs therefore gain nothing from more GPUs: use one. N-body steps split their pair work over the GPUs: 96 % efficiency on 8 GPUs at 6.24M particles (see [N-body steps on several GPUs and CPU nodes](#n-body-steps-on-several-gpus-and-cpu-nodes)).

**Several runs on one GPU** (for example the evaluations of an optimizer): start NVIDIA MPS (`nvidia-cuda-mps-control -d`) first. Without MPS the GPU time-slices between processes, and two or more runs at once gave half the throughput of one run at a time; with MPS, 2–8 runs at once gave 1.3× (780k particles) and 1.1× (6.24M) the throughput ([Benchmarks](#several-runs-at-once)).

**CPU runs over InfiniBand clusters with UCX** (any MPI that uses UCX, such as NVIDIA HPC-X): set `UCX_TLS=self,sm` for runs on one node and `UCX_TLS=self,sm,ud` across nodes. With UCX's default transports, the `MPI_ALLTOALLV` transposes of the FFT were up to 17× slower and a 32-rank run 4.5× slower on our cluster. This is a site setting, not a code change, but it made the largest difference to CPU speed of anything we measured.

# What runs on the GPU

| Part of a step | GPU | Otherwise |
|---|---|---|
| Emission from the cathode, drifts, range and centroid, lost-particle test and removal, `Flagdiag = 1` and `2` moments | yes | `IMPACTT_LOSS=host`, `IMPACTT_DIAG=host`: host |
| Deposition, Poisson solve (open boundaries, image charge), field gradients | yes (`Flagbc = 1`) | other `Flagbc`: CPU fallback |
| External fields: type 105 (discrete on-axis data, file ID > 1000), 112, 1 (quadrupole, all modes), 111, 4 (dipole outside the bend loop), 101 (DTL, Fourier on-axis field; the harmonics by recurrence) | yes | other types (102, 103, 104), and 105 with Fourier coefficients: CPU fallback for the steps where the element is in range |
| 2D deposit (azimuthal symmetry, no `-5` element) | | CPU fallback for every step: a GPU run needs a `-5` line |
| Point-to-point space charge (`-15`/`-19`), with image charge, on one or several GPUs | yes | |
| Diagnostics `Flagdiag = 2` | yes | 1, 3, 105: on the host after a copy |
| Type-4 bend loop: mesh or no space charge, CSR, N-body | yes (one bunch, `Flagbc = 1`) | `Nbunch > 1` or `Flagbc ≠ 1`: the host bend loop |
| `-2` dumps (text and binary), `-1` steering, `-11` collimation, maps, heating, rotation, bunch merge, restart output | on the host, after a copy of the particles | |
| Wakes, DWA, `Flagerr = 1`, `FlagFieldPrint`, azimuthal symmetry (no `-5` element), integrators 2 and 3 | | CPU fallback for that step |

Not tested on the GPU path: `Nbunch > 1` and the merge, restart (`-3`, `Rstartflg = 1`), `Flagdiag = 105`, element types beyond those above.

# Input additions

All are optional; existing inputs behave as before (with the one exception noted for bends).

- **`-15` (point-to-point space charge on), V4 = the z at which it switches on** (default 0: from the start, as before). V3 is the cutoff `r0` in metres, as in `master` since `3e19b7f` (before it, `r0` was in units of c·Dt). **V5 = `r0img`**, the same cutoff for the image pairs only (metres; 0 or absent: `r0img = r0`, so older inputs are unchanged). A particle's own image is a pair 2z apart, so a small `r0` (a few nm) loses the particles just emitted unless `r0img` is kept near 10 nm. Example: `0 0 0 -15 0.0 0.0 5.0e-9 0.0008 1.0e-8 /` (r0 = 5 nm, on from z = 0.8 mm, `r0img` = 10 nm). Several `-15` lines are separate switches at their own z; in `master` the last one read sets r0 and the solver is point-to-point for the whole run.
- **`-19` (back to the mesh solver), V3 = z.** Example: `0 0 0 -19 0.0 0.0 0.0020 0.0 /`. Up to 100 `-15`/`-19` switches, sorted by z, each acting on the step whose `[distance, distance+dzz]` contains it (the test the `-4` elements use).
- **Point-to-point space charge in a bend.** A bend uses it when it is on at the bend entrance (a `-15` before the dipole, a `-19` after it). **This changes inputs whose `-15` window spans a bend**: before, the bend used the mesh solver; now it uses point-to-point. Image charge is off in the bend, every particle is a source and a target, and the space-charge magnetic field `bz` is kept.
- **Binary `-2` dump: V4 > 0** (the 8th number on the line: `0.0 1 199 -2 2.0 2.0 2.0 1 /`). It writes `fort.<Bmpstp>` as a stream: an `int64` particle count, then count × 9 `float64` (the 9 columns of the text dump, same units and order). 72 bytes per particle; 6.24M particles take 0.6 s against 8.8 s for text.
- **`Flagdist = 168`** reads that stream as `partcl.data`, each rank its own slice; otherwise as `Flagdist = 166` (charge from column 8, `Nemission = 0`, `Np` must equal the count). 1.1 s for 6.24M particles against 28 s for text.
- **Binary type-111 maps.** `1T<ID>.T7b` is read in place of `1T<ID>.T7` when it exists: an `int32` magic (the bytes `IT7B`), an `int32` version (1), then for x, y and z a `float64` min, a `float64` max and an `int32` number of intervals, then Ex, Ey, Ez, Bx, By, Bz, each (Nx+1)(Ny+1)(Nz+1) `complex128` with x fastest. `utilities/t7_to_t7b.py 1T<ID>.T7` writes it from the text map (`--verify` reads it back and compares bit for bit). The results are the same; the read is faster (two 1.35 GB maps: a 2k-particle run went from 41.6 s to 28.5 s). Remake the `.T7b` when the `.T7` changes: a stale one would be read in its place.
- **Longer discrete on-axis tables.** Type 105 with file ID > 1000 and type-1 profiles take up to 65,537 E rows (was 9,999). A Fourier field converted to a table needs a step of about 20 µm to stay within 1e-6 of the series with the linear interpolation `getfldt_SolRF` uses.
- **Field files stay in memory.** A file already read into one of the `Maxoverlap` slots (now 16, was 10) is not read again, matched by file ID and element type. Before, after the 10-slot ring wrapped, a file still in memory was parsed again whenever the range of active elements widened. Results are unchanged.

# Changes to the existing code

These change the behaviour of the CPU code (except where noted), and each was found by running or checking the GPU port against it. Commits are on `gpu-port-mvp`.

**Particle corruption at emission in the particle exchange** (`75dde0e`, `Func/Ptclmger.f90`). `ptsmv2_ptclmger` has four fixed send buffers of 3·`nptmv` = 300,000 particles. Its first stage, which sends a rank's own particles, moves at most `nptmv` per direction and pass and repeats; but its forwarding stage, which passes on particles received from one neighbour that belong further on, wrote into the same buffers with no bounds check. When more than 300,000 particles had to be forwarded in one direction, the writes ran past the end of the buffer and overwrote other particles' records, with the particle count unchanged. This happens at the start of emission, when the whole unemitted bunch sits in one z slab and moves across several ranks at once. In our benchmark (6.24M and 3.12M particles, `Flagdist = 16`) the runs with 3.12M particles on 32 ranks and 6.24M on 32, 64 and 128 ranks ended with the beam 2.4–6.3 µm off axis and about twice the emittance; an instrumented build reported up to 390,000 forwarded particles in one direction. Upstream `master` gives the same numbers. Runs at ≤ 1.56M particles, at 6.24M on 256 or 512 ranks, and on the GPU (no exchange) were not affected. The buffers are now allocatable (they were 86 MB of local arrays) and the forwarding stage grows them as needed. Without an overflow the results are bit-identical to before; with the fix, the affected runs agree with the GPU and with the unaffected CPU runs to all printed digits. **Old CPU results with millions of particles on up to ~128 ranks should be checked**, for example for a centroid that leaves the axis at the first diagnostic steps.

**DTL field** (`6a9c46c`, `Appl/DTL.f90`). `getfld_DTL` and `getfldt_DTL` set Ez''' to 0 and never summed it, so the r³ term of Er and Bφ was missing, and the Ez'' sum had cos where the odd Fourier terms need sin (`getflderr_DTL` and `getflderrold_DTL` were already right). On IMPACT-Z's Example 1 written as an IMPACT-T input (space charge off, 10⁴ particles), the x emittance after the DTL differed from IMPACT-Z by 4.0e-3, and now by 6.4e-4; y and the energy gain were already close. **Every run with a type-101 element changes.**

**Bend exit** (`eaff33d`, `Contrl/AccSimulator.f90`, CPU and GPU path). The type-4 bend loop advances the reference path length by v·Dt and stops at the first step past the bend length; `convExit` then put the reference at the exit whatever the overshoot, so the bunch lost 0 to 1 step of path per bend and arrived late at every later element. The exit now places the reference at its actual path length. On a four-dipole chicane (Dt = 1e-12 s) the bunch arrives 0.171, 0.924, 0.317 and 0.817 ps earlier at the four exits; against IMPACT-Z (10⁴ particles, five cavities after the chicane) the error in the energy gain fell from 4.9e-3 to 7.9e-6. **Every run with a type-4 bend changes, including straight (V3 = 0) bends**; with CSR on, the energy after bend 2 of the chicane changed by -198 eV.

**Space-charge mesh at z = 0** (`4303006`, `Contrl/AccSimulator.f90`). The lower z edge of the mesh was clamped to 0 whenever the bunch straddled z = 0, even with no cathode (`Nemission = 0`). A read-in bunch behind z = 0 then lost its z < 0 part from the deposit (on the GPU the run stopped with `dep - index out of range`). The clamp now applies only with a cathode; runs that start at a cathode are byte-identical.

**Point-to-point space charge** (A1 `99aa448`, A4 `e72c124`, `Appl/BeamBunch.f90`, `Contrl/AccSimulator.f90`):

- `kickpt2ptImg_BeamBunch` gathered the particles into a stack array (450 MB at 6.24M particles, a stack overflow) with a gather that assumed every rank holds the same number of particles. It now gathers into a heap array with `MPI_ALLGATHERV` and real counts. The pair loop is a separate routine, `pt2ptFields`, which the GPU code shares.
- The image term was always on. It now follows `Flagimg` and stops when the bunch passes `zimage`, as in the mesh solver. (`master` `3e19b7f` now also turns it off with `Flagimg = 0` and past `zimage`, by calling `kickpt2pt_BeamBunch`; here `pt2ptFields` takes a flag, which also serves the GPU path.)
- A particle did not feel its own image; it does now, as in the mesh solver, while the image is on. This needs one-to-one charge (one macroparticle per electron) and an image cutoff near 10 nm (`r0`, or `r0img` when `r0` is smaller): with a cutoff of 0 or a few nm the self-image of a newly emitted electron is nearly uncapped, and the beam returns to the cathode. With macroparticles much heavier than an electron the self-image is too strong by their charge ratio for any small cutoff.
- The call passed particle counts taken once before the time loop. Once a run can switch from the mesh to point-to-point (after the mesh steps have moved particles between ranks), that read and wrote outside the particle array (a segmentation fault in a 4-rank run). It now passes the current counts.
- The pair formula computes 1/r once per pair and precomputes γ²/(γ+1) and γ/(γ+1) per source particle: 1.25× faster on the CPU, with the fields equal to an independent pair sum to 6e-12.

**CPU N-body pair loop in cache-sized blocks** (`275120f`, `Appl/BeamBunch.f90`), speed only. For each target, `pt2ptFields` read every source (80 bytes) from memory, so with many sources a full node was limited by memory bandwidth: with 6.24M sources, 128 ranks on a 2 × 64-core AMD EPYC 9534 node evaluated 6.8e9 pairs/s against 1.23e10 when the sources fit in cache, while one core alone ran at the same rate either way. The CPU loop now passes over the sources in blocks of `IMPACTT_PT2PT_BLOCK` (default 4096) and runs every target over each block, keeping each target's sums between blocks. The sources are still added in the same order, so the fields are bit-identical for any block size (`tests/test_pt2pt_block.f90`; a 128-rank N-body run gives byte-identical outputs). At 6.24M sources a node now runs at 1.23e10 pairs/s, 1.81× faster. The GPU loop is unchanged.

**Particle read-in** (`6b9a40e`, `a6ce288`, `Appl/Distribution.f90`):

- `readold_Dist` (`Flagdist = 16`) and `read_Dist` (166) had every rank parse the whole `partcl.data` and keep its own slice. With 6.24M particles on 128 ranks that took 396 s (128 parses of a 911 MB file on a shared file system). Rank 0 now parses the file once and sends each rank its slice: 14 s, bit-identical results.
- The particle ids of `readold_Dist`, `readParmela_Dist` and `readElegant_Dist` were `j + myid*avgpts`, which repeats some ids and skips others when the count does not divide by the number of ranks (the first `nleft` ranks hold one particle more); 51,200 particles on 6 ranks had 51,198 distinct ids. They are now the particles' positions in the file. Coordinates were not affected.

**Field-file readers** (`eba221e`, `DataStruct/Data.f90`): `read1t_Data` (Fourier coefficients, dipole data) read numbers until the end of the file into `Fcoeft(401)`, and `read1tdata_Data` (discrete on-axis data) read the file's own row counts into `Fcoeftdata(4,10002)`, both without a bounds check; a longer file overwrote memory silently. Both now stop the run with a message naming the file and the limit (`MPI_ABORT`; a serial stub is added to `mpistub.f90`). `Fcoeftdata` now holds 65,540 rows (`3189cc3`, see [Input additions](#input-additions)), so each field slot takes 2 MB instead of 0.3 MB; with 16 slots that is about 33 MB per rank, on the host and on the GPU.

**Diagnostic outputs** (`7a98cac`, `f1f8aa0`, `cbfc4e4`, `Contrl/Output.f90`), output only:

- `fort.29` (third-moment roots) was NaN wherever a third central moment was negative (`m3**(1.0/3.0)`); 493 of 840 entries in a short test run. The 19 such lines in the diagnostic routines now compute `sign(abs(m3)**(1.0/3.0), m3)`, bit-identical where the old value was not NaN. (The lines that already took `abs()` are unchanged, so their sign is lost; the exponent `1.0/3.0` is a default-real constant in both.)
- `fort.60`/`fort.70` (slice quantities): the slice emittance was the square root of a number that rounding makes slightly negative for a slice of one particle or of nearly zero emittance, and the slice energy spread divided by a zero count for an empty slice. Both now give 0 there.

**Values read before they were set** (all found by a build that initializes every real to a signalling NaN and traps invalid operations; results byte-identical before and after):

- `tbtwstart` (`Contrl/AccSimulator.f90`, `f1f8aa0`) was compared with t every step but set only by a `-6` element. Harmless (its flag is used only by the wake calculation), now initialized.
- `sum(Np)` (end-of-beam test, `totnpts`) summed all `Nbunchmax` entries of `Np` while only `1..Nbunch` are set (`2916fd2`); module variables happen to start at zero in practice. `Np`, `Nplocal` and `gbetz` are now zeroed first.
- `gradEB_FieldQuant` left the guard layers at the edges of the global domain uncomputed, as stack garbage (`fb0ce5f`). They are never interpolated from (zeroing them gives byte-identical results with ranks split in y and z), and are now zeroed.
- `showtime_aux` divided the `MPI_REDUCE` result on every rank, where it is defined on rank 0 only (`cbfc4e4`).
- A dipole with zero field (V3 = 0) divided by zero for the bend radius (`cbfc4e4`); the radius is now set to +∞ explicitly, the value the division gave.

**Zero-length MPI buffers** (`a9b1112`, `7a98cac`): `ptsmv2_ptclmger` and `phase_Output` passed an element just past the end of an array (or of an empty array) as the buffer of a receive or send of zero elements. Harmless, but it stops a bounds-checked build; one spare column, and whole-array arguments (the serial build's MPI stub, `mpistub.f90`, gets the matching 2-D variants of `MPI_SEND` and `MPI_RECV`).

**Noted, not changed:**

- The built-in generators (Uniform, Gauss3, Waterbag, KV3d, Semigauss, Combine, ParobGauss, SemicirGauss) create `nproc·floor(Npt/nproc)` particles but keep the header `Npt` for the charge per particle, so up to `nproc − 1` particles and their charge are missing.
- The CSR constant `kkcsr = -2/(3*r0bend)**(1.0/3.0)` and several cube roots use a default-real exponent (an error of about 1e-8 relative).
- `psiroot` (`Appl/Field.f90`, CSR) takes `(24*deltas/r0)**(1.0d0/3.0d0)`, NaN if `deltas < 0`; not checked.
- The exchange routine's `msk` is an automatic array of one logical per local particle (25 MB of stack if one rank held 6.24M particles).
- External fields are applied only for the elements that overlap the bunch's `[zmin, zmax]`. While the leading particle is in a gap between two field maps (44–89 µm between the cells of our gun), the previous cell's field is dropped for the whole bunch, including the tail still inside that cell. This holds for every solver; for a longer bunch it lowered γ by 2.8e-4 at a cell exit. A fix would take the element range from every element that overlaps the bunch.
- A type-1 quadrupole with file ID 0 (hard-edge mode) calls `read1t_Data(…, 0)`, which reads the unconnected unit 14: gfortran reads an empty `fort.14`, an nvfortran build stops with `FIO-F-209`. Work-around: an empty `fort.14` in the run directory.
- Element crossings (`-2`, `-4`, `-15`, `-19`, ...) act only on the step whose `[distance, distance+dzz]` contains them, and `dzz` uses the speed at the start of the step, so in the gun, where the beam accelerates, there are gaps between successive windows: an element placed in a gap never acts, and then blocks the later elements of its type. An element inside a bend's z range is skipped in the same way. On the first step the crossings are tested with `distance = 0` and `dzz` from `Bkenergy` (the beam's position is computed later in the step), so in a run started from a particle file mid-beamline (`Flagdist = 166/168`), a `-15` meant to act from the start needs V4 = 0; a V4 at the beam's own position is never crossed.

# Design overview

Each item below is a section of `doc/GPU_PORT_DESIGN.md`.

- **Programming model.** CUDA Fortran (device arrays, `!$cuf kernel do` loops and `attributes(global)` kernels) for the mesh solver and the particle kernels, cuFFT for the FFTs, and OpenACC for the N-body pair loop, all compiled by `nvfortran` into one executable. Everything runs on the default stream.
- **Poisson solve (B1).** The open-boundary solve with the integrated Green's function is done on the doubled grid with cuFFT: one forward transform of ρ, one of G, a product and an inverse, plus the same for the image potential with the z-shifted G and a reversal in z. The Green's-function kernels are verbatim ports of `greenf1tIntnew2`/`greenf1tIntshift2`. With several ranks, every rank gathers ρ and solves the whole grid.
- **Particles on the device (B2).** `Pts1` keeps its `(9,N)` layout on the device, and the host array becomes a mirror with a three-state flag (in sync, device newer, host newer); every host routine that touches particles is wrapped by `ensure_host`/`mark_host_newer`. Kernels are verbatim ports of the CPU routines, with single-precision literals kept, so they agree with the CPU to the last bits (fused multiply-adds move the last bit of some values).
- **External fields (B2.6, D1.2).** A table of the elements in range is built on the host each step, with every per-step constant (RF phases, rotation angles) computed on the host with the CPU's expressions; `extfld_dev` sums the elements in table order, as `getfldt_BeamLineElem` does.
- **Fallback.** Each step decides on the host whether it can run on the GPU (`gpustep`); if not, the existing code runs that step between a copy of the particles to the host and back.
- **Bends (D1).** The bend-frame loop runs on the GPU with the reference particle kept on the host, line for line. CSR keeps the wake computation on the host (it works on Nz points) and does the histogram and the field addition on the device.
- **Point-to-point (A2, A3, D1.12).** The pair loop runs as an OpenACC loop over targets with a sequential loop over sources, the sums in registers. It runs at 2.0e11 pair evaluations per second on one H100 at 1M particles and is compute-bound in FP64.

# Validation

All comparisons below are between the GPU path and the CPU code, mostly in the same executable (`IMPACTT_PARTICLES=cpu`), so no compiler difference enters. The GPU results are not bit-identical run to run: deposition uses FP64 atomic additions, whose order varies, so the GPU is compared against tolerances and against its own run-to-run difference.

## Unit tests

Each test in `tests/` compares GPU kernels with the CPU routines they port, on the same inputs (see `tests/README.md`). All pass, with synthetic input files (`tests/make_test_inputs.py`). The GPU tests were last run on an H100 at `94572e6`; the commits after it change host code only. The CPU tests pass at the last commit.

| Test | What it compares | Agreement |
|---|---|---|
| `test_pt2ptfields` | `pt2ptFields` (GPU and CPU) against an independent pair sum; analytic self-image | 5.9e-12 (image on), 7.4e-13 (off), exact; GPU and CPU fields bit-identical; 1, 2, 4, 8 GPUs bit-identical |
| `test_pt2pt_block` | the blocked CPU pair loop against the unblocked loop, block sizes 1 to more than the source count, image on and off | bit-identical |
| `test_poisson_gpu` | `openBC3D_gpu` against `openBC3Dnew` (FFTW), and a direct sum | φ ≤ 5e-13 up to 128×128×256; image potential as close to the direct sum as the CPU's (the 8-corner difference of the integrated Green's function limits both); rank counts bit-identical |
| `test_dtl_gpu` | the type-101 field (kind 6) against `getfldt_DTL` on nine cells, 5 times across an RF period | ≤ 1.3e-15 of each component's maximum |
| `test_lostdiag_gpu` | loss removal on the device against `lostREC_BeamBunch`/`lostDFO_BeamBunch` (edge cases, 0 to all lost, repeated calls, buffer growth, swap and copy back); `Flagdiag = 1` sums against the host loop; the written `fort.18`, `fort.24-32` | survivors identical in all 9 columns and in order; sums 5.5e-14 of the sum of absolute terms; files within the last printed digit |
| `test_particle_gpu` | drifts, emission, range, losses, deposit, gradients, the three kicks, external fields of 105 and 112 | kicks ≤ 7.3e-15 of the largest kick (point-to-point 2.9e-13); range, losses and gradients exact; deposit 6e-15 of max ρ |
| `test_bend_gpu` | quadrupole, dipole and type-111 fields; bend-frame rotations, conversions, drifts; bend kicks; bend diagnostics; N-body kick in the bend frame | ≤ 2e-16 (quadrupole), ≤ 9e-16 (dipole), exact (111); final momenta ≤ 3.1e-16 of \|p\| |
| `test_bendnbody_cpu` | the CPU N-body kick in the bend frame against the analytic two-particle field | 1.7e-11 |
| `bench_pt2ptfields` | pair rate | 1.95–1.98e11 pairs/s on one H100 at 1M particles |

## Full runs, GPU against CPU

On a 5-cell X-band gun and drift (1 pC, 20k and 1M particles, one-to-one charge, image charge on), the GPU path against the CPU particle path of the same executable:

| Case | Largest difference in fort.18/24/25/26 (each column scaled by its own range) |
|---|---|
| 20k, 0 → 6 mm, 25 mm, 2 m; element 105; element 112; 4 ranks; three N-body/mesh switches | 3e-7 to 4e-6; the largest are in early rows of columns with a tiny range; later rows ≤ 1e-7 |
| 20k, two GPU runs of the same input | of the same size (atomics) |
| 1M, 0 → 25 mm, 128×128×256 | ≤ 1.3e-6, except the beam centroids (~1e-11 m), which differ as much between two GPU runs |

On a beamline with quadrupoles, a type-111 RF deflector and a four-dipole chicane (20k particles), with and without space charge, with CSR on in one to all dipoles, with N-body in a dipole, and at 4 ranks: GPU against CPU at the GPU's run-to-run difference. Diagnostics agree to ≤ 7.3e-8 (column-scaled). Final particles agree to ≤ 2.3e-10 of the rms without CSR, to 2.0e-9 with N-body in a dipole, and to 2.3e-7 with CSR in every dipole, where CSR amplifies the step-to-step differences (two GPU runs differ by 2.4e-8 there). The CPU path is unchanged by these stages to the bit.

**The benchmark beamline** (below), end of the beamline, GPU against the CPU, at 64³ (780k to 6.24M particles, 32 to 512 ranks) and at 128³ (780k to 3.12M particles, 128 ranks): every rms size, emittance, bunch length and kinetic energy in `fort.18/24/25/26` is equal to all eight printed digits, and so is the energy spread except in one case (3.12M at 128³, the last digit). The beam centroids, which are below 1e-10 m, differ in the eighth digit.

## The CPU path against upstream

**The examples in `examples/`.** `gpu-port-mvp` (`05aee00`) and `master` (`3e19b7f`), CPU builds with the same compiler, flags and FFTW (gfortran 15, Open MPI, macOS arm64), each example in `examples/` as distributed, serial and on 4 ranks (2 × 2), every `fort.*` file compared byte by byte:

| Example | Serial | 4 ranks |
|---|---|---|
| `Sample1`, `Sample2_TWS`, `Sample3_3Dfield`, `Sample4_2Dfield`, `Sample6_cavWake` | identical except `fort.29` (NaN replaced by the signed cube root) | identical except `fort.29` (`Sample4`: both stop with `MPI_Abort` after the same steps) |
| `DemoPhaseScan` (one particle) | identical except `fort.60`/`fort.70` (NaN replaced by 0) | both stop with a segmentation fault (one particle on 4 ranks) |
| `Sample5_2bunch` (two read-in bunches) | differs: the second bunch straddles z = 0 with `Nemission = -1`, and the mesh-clamp fix (`4303006`) now deposits its z < 0 part. With only that commit reverted, every file is identical except the NaN fixes in `fort.60`/`fort.70` | both go to NaN after about 60 steps (`master` `78c029b` too) |

The same comparison of the branch before the rebase against `78c029b` gave the same result. The two `master` versions give identical outputs on every example (on 4 ranks, all but `Sample5_2bunch`, whose runs go to NaN in both). The `.T7b` path was checked on `Sample3_3Dfield`: with only `1T1.T7b` (made by `utilities/t7_to_t7b.py`) in place of `1T1.T7`, all 14 outputs are byte-identical.

**The benchmark case.** `gpu-port-mvp`'s CPU build and `master`, same compiler and flags, on the benchmark case with 51,200 particles from the cathode to 0.2 m (emission, image charge, gun): **every `fort.*` file byte-identical**, serial and with 8 and 32 ranks. (This was checked before the bug fixes; each fix since was checked byte-identical to the previous build except where it removes a NaN or corrects ids, and the exchange fix only where the old code overflowed.)

## Run-time checks of the CPU code

Two gfortran builds of the final code: one with run-time checks of array bounds, loop counts, pointers and allocations (`-fcheck=bounds,do,mem,pointer,recursion`), one that initializes every real to a signalling NaN and traps invalid operations, division by zero and overflow (`-finit-real=snan -ffpe-trap=invalid,zero,overflow`). Six cases each: the benchmark from the cathode to 0.2 m (8 ranks) and to 3.1 m (32 ranks), 3.12M particles on 32 ranks through the start of emission, a `Flagdist = 166` start on 6 ranks (an uneven split), point-to-point space charge with image charge (32 ranks), and a 20k leg with six quadrupoles and four dipoles from a `Flagdist = 168` file (4 ranks). All twelve runs complete with no error and no trap. The fixes for the values read before they were set, and for the NaN outputs, came from earlier rounds of these checks.

# Benchmarks

**Case.** A 1 pC photoinjector: a 9.3 GHz X-band gun at 150 MV/m with a solenoid, then three 9.3 GHz linac sections, from the cathode to 3.1 m (28.8 MeV). All cavities are element 105 with discrete on-axis data. The beam is emitted from the cathode (`Flagdist = 16`, 200 emission steps) with a radial profile of 20 µm rms and a 100 fs FWHM Gaussian time profile; 49,390 steps (41,100 at 2.5e-13 s, 8,000 at 1e-14 s across the RF edges). `Flagdiag = 2`, image charge to 2 cm. Particle counts 6.24M (one macroparticle per electron), 3.12M, 1.56M and 780k.

**Hardware.** CPU: one node with 2 × AMD EPYC 9534 (Zen 4, 2 × 64 cores, SMT off, 512 GB), exclusive, one rank per core, Open MPI 4.1.5 (NVIDIA HPC-X) with `UCX_TLS=self,sm`, gfortran 11.2 `-O3`, FFTW 3.3.11. GPU: one NVIDIA H100 80GB HBM3 (SXM, 1980 MHz), nvfortran 26.3. Every run logged its clocks and temperatures every 10 s: no thermal or power throttling on either.

**Times** are IMPACT-T's own `integ` timer (the time loop) in seconds, one run each. The read-in of 6.24M particles adds 14 s on the CPU and 20 s on the GPU, the final `fort.50` about 25 s.

| Particles | 64³: CPU 16 / 32 / 64 / 128 cores | 64³: H100 | 128³: CPU 32 / 64 / 128 cores | 128³: H100 | 256³: H100 |
|----|------------|----|----------|----|----|
| 780k | 1056 / 864 / 614 / 517 | 50 | 6114 / 5186 / 4720 | 117 | 643 |
| 1.56M | 1719 / 1192 / 786 / 594 | 80 | 6588 / 5406 / 4831 | 147 | 673 |
| 3.12M | — / 2138 / 1123 / 858 | 140 | 6994 / 5897 / 4888 | 209 | 738 |
| 6.24M | — / 4037 / 2138 / 1253 | 264 | 8985 / 6004 / 4653 | 336 | 872 |

One H100 against one 128-core node: 4.7–10× at 64³ and 14–40× at 128³ (the CPU's 128³ times are dominated by the FFT transposes, nearly independent of the particle count). The 128³ CPU runs at 3.12M on 32 ranks and 6.24M on 32, 64 and 128 ranks were made before the exchange fix, which affected their results; at 64³ the fixed reruns took between 1 % less and 3 % more time than the affected runs.

**CPU, upstream against `gpu-port-mvp`** (6.24M, 64³, loop time): 4031 / 2141 / 1217 s against 4037 / 2138 / 1253 s on 32 / 64 / 128 cores. The upstream runs have the exchange bug in their results. With the faster read-in, the wall time at 128 cores is 1272 s against 1615 s.

**Several CPU nodes** (6.24M): 256 ranks on 2 nodes and 512 on 4 are 1.42× and 1.80× faster than 128 ranks on one node at 64³, and 1.15× and 1.22× at 128³. The same 128 ranks spread over 2 or 4 nodes gained 2–6 % (these runs were before the exchange fix).

## N-body steps on several GPUs and CPU nodes

Four point-to-point steps of the same beam at 6.24M particles, started at z = 0.2 m from a binary `-2` dump (`Flagdist = 168`, `-15` with r0 = 10 nm from the first step, no image charge past 2 cm), so each step is N² = 3.9e13 pair evaluations. One rank per H100 on one node; each rank computes the fields on its share of the targets (6.24M/P) from all 6.24M sources, which are gathered through the host every step. Time per step from IMPACT-T's `kickgpu` timer (gather, pair kernel and kick) and `pt2ptfields` timer (pair kernel), one run each:

| GPUs | Step (s) | Pair kernel (s) | Gather and kick (s) | Speed-up | Efficiency |
|---|---|---|---|---|---|
| 1 | 235.3 | 235.3 | 0.0 | 1 | |
| 2 | 120.1 | 119.7 | 0.5 | 1.96 | 98.0 % |
| 4 | 60.5 | 60.1 | 0.5 | 3.89 | 97.2 % |
| 8 | 30.6 | 30.1 | 0.5 | 7.70 | 96.3 % |

The pair kernel runs at 1.65e11 pair evaluations per second per GPU here. The 1.98e11 of the unit benchmark (`tests/bench_pt2ptfields.f90`) is with the image term on, where each source read serves two pairs (direct and image): the floating-point work per pair is the same (53 FP64 instructions), but the loads, flag tests and loop overhead per pair are smaller (43 against 55 other instructions, Nsight Compute). Measured with the same routine on one H100, image off gives 1.56e11 and image on 1.95e11, for 1M or 6.24M sources and for synthetic particles or this beam alike. The host gather of the 500 MB source array costs about 0.5 s a step at any GPU count, so it would matter only at much smaller particle counts or with many more GPUs. All GPUs ran at 1980 MHz; one reached its thermal limit at 80 °C in two 10 s samples, at 1965 MHz.

**On CPU nodes**, the same beam and input, 2 steps, with the blocked CPU pair loop (`275120f`, `-O3`) on 1–8 exclusive nodes of 2 × AMD EPYC 9534 (128 ranks a node; `UCX_TLS=self,sm` on one node, `self,sm,ud` on several). Every rank receives all 6.24M sources each step (`MPI_ALLGATHERV`). Step time from the time-stamped log, pair loop from the `pt2ptfields` timer, one run each:

| Nodes | Ranks | Step (s) | Pair loop (s) | Speed-up | Efficiency |
|---|---|---|---|---|---|
| 1 | 128 | 3177 | 3167 | 1 | |
| 2 | 256 | 1586 | 1585 | 2.00 | 100 % |
| 4 | 512 | 795 | 793 | 4.00 | 100 % |
| 8 | 1024 | 397 | 396 | 8.00 | 100 % |

That is 1.23e10 pair evaluations per second per node at every node count. The gather itself takes 0.3–0.4 s on 1–8 nodes (timed separately; the transport and the collective component made no more than 0.1 s of difference). Without the blocked loop a node runs at 6.8e9 pairs/s, about 5,700 s a step on one node. Built with `-march=native` (AVX-512 and FMA, which changes the last bits of the fields), one node takes 2409 s a step. One H100 (235 s) does a step 13.5× faster than one node, and eight H100s (30.6 s) 13× faster than eight nodes.

## A lossy DTL lattice, 10⁶ to 10⁸ particles

IMPACT-Z's Example 1 (a DTL lattice with apertures that remove particles) written as an IMPACT-T input, one H100, `IMPACTT_NODUMP=1`, the `integ` timer. Before `5527e28` and `84237ae`, every step with a lost particle and every `Flagdiag = 1` diagnostic copied the whole beam to the host, which was 94 % of the 10⁷ run. "Now" is at `84237ae`, except the two 32×32×64 rows at 10⁷ and 10⁸, which include the buffer swap of `94572e6` (90.1 → 89.2 s and 838 → 819 s):

| Particles | Grid | Before (s) | Now (s) |
|---|---|---|---|
| 10⁶ | 32×32×64 | 77.6 | 12.8 |
| 10⁷ | 32×32×64 | 1208 | 89.2 |
| 10⁷ | 64³ | 1468 | 90.2 |
| 10⁷ | 128³ | 1159 | 109 |
| 10⁷ | 256³ | 1389 | 235 |
| 10⁸ | 32×32×64 | stopped at 14,000 (of about 20,000) | 819 |

At 10⁸ the kernels take 82 % of the time and the loss removal 102 s (4,777 steps with a loss); the peak device memory is 14.7 GB. The type-101 field on the GPU (`7c5eeff`) is what makes this lattice run on the GPU at all: before it every DTL step fell back to the CPU (10⁵ particles: 9–15 s on the GPU path against 174 s with `IMPACTT_PARTICLES=cpu`).

## Several runs at once

Evaluations per hour at 64³, for an optimizer that runs many independent simulations:

| Setup | 780k | 1.56M | 6.24M |
|---|---|---|---|
| one CPU node, one run on 128 cores | 6.8 | 6.0 | |
| one CPU node, 2 × 64 / 4 × 32 / 8 × 16 cores at once | 11.1 / 15.9 / 21.0 | 8.7 / 11.6 / 12.4 | |
| one H100, one run at a time | 63 | | 12.4 |
| one H100, 2 / 4 / 8 runs at once, no MPS | 30.7 / 31.0 / 31.2 | | 10.7 / 10.9 / — |
| one H100, 2 / 4 / 8 runs at once, MPS | 81.7 / 84.5 / 85.7 | | 13.8 / 14.0 / — |

Running several smaller CPU runs side by side on a node slows each by 2–3 % (2 × 64, 4 × 32) to 28–34 % (8 × 16). One H100 with MPS gives about as many 780k evaluations per hour as four CPU nodes.

## CPU or GPU?

From the measurements above (one H100 against one 2 × 64-core EPYC 9534 node):

- **One run, as fast as possible: the GPU.** It is 4.7–10× faster than a whole CPU node at 64³ and 14–40× at 128³ (256³ was not run on the CPU). More CPU nodes do not close the gap: 6.24M particles on 4 nodes (512 ranks) took 2.6× (64³) and 11× (128³) as long as one GPU. The GPU's lead grows with the grid, because the CPU's FFT transposes dominate, and at small grids it shrinks as the particle count grows.
- **N-body: the GPU.** A 6.24M-particle step takes 235 s on one H100 and 31 s on eight, against 3177 s on one CPU node and 397 s on eight (1024 ranks; the CPU scales linearly) with the blocked pair loop: one H100 does the work of about 13 CPU nodes.
- **Many independent runs (scans, optimizers): both.** Per device the GPU gives 4–5× the runs per hour of a CPU node (780k at 64³: 86 with MPS against 21 for eight 16-core runs at once; 6.24M: 14 against 2.9 for one 128-core run). A machine with many more CPU nodes than GPUs can get a similar total from its CPU nodes, and using both adds up.
- **The CPU, for lattices that use what the GPU path does not port** (`Nbunch > 1`, restart, `Flagdiag = 105`, `Flagerr ≠ 0`, bends with `Flagbc ≠ 1`): those steps run on the host of the GPU executable, at about 35× the cost of a GPU step at 6.24M, so a lattice that needs them often is better on CPU nodes.
- **Very small runs** (thousands of particles) were not measured on the GPU. There its per-step overhead dominates and a CPU node can run one per core, so the CPU may well be the better choice.

# Known limitations

- **Determinism.** The GPU path is not bit-reproducible run to run (FP64 atomics in the deposit and the CSR histogram); two runs differ at the 1e-7 level in column-scaled diagnostics, more in near-zero centroids. The CPU path in the GPU executable is not exactly reproducible either (about 5e-15 in a dump; not investigated).
- **Text output from nvfortran** (`g0` in `phase_Output`) prints about 16 significant digits, so a text dump written by the GPU executable does not read back bit-exactly; the binary `-2` dump does.
- **Loss removal on several GPUs** was checked by reading the code only; every run with the device compaction was on one GPU.
- **`Flagdiag = 3` and `105`** still copy the beam to the host for each diagnostic.
- **Multi-GPU** gives no speed-up for mesh steps (the solve is repeated on every GPU), and the N-body gather goes through the host (`MPI_ALLGATHERV` of host buffers): about 0.5 s a step at 6.24M particles, 2 % of an 8-GPU step. Multi-GPU runs were measured on one node only.
- **N-body cost** is O(N²): 235 s per step without image charge at 6.24M particles on one H100 (31 s on 8), so a full point-to-point run of a beamline at that size is a reference calculation, not a tuning tool.
- **Not tested on the GPU path:** `Nbunch > 1`, restart, `Flagdiag = 105`; untested element types fall back to the CPU per step.
- **Element crossings** use a one-step window (see [Changes to the existing code](#changes-to-the-existing-code), "Noted, not changed"); the GPU path behaves as the CPU path.

# Files

| File | Change |
|---|---|
| `src/Func/PoissonGPU.f90` | new (B1): GPU Poisson solve with cuFFT, Green's-function kernels, `poisson_dev`, `gradEB_dev`, ρ sum over ranks |
| `src/Appl/ParticleGPU.f90` | new (B2, C1, D1): device particle buffers and residency; drift, emission, range, loss, deposit, kick, N-body source and kick kernels; diagnostics sums; bend-frame kernels and loop helpers; CSR histogram; error flag; counters |
| `src/Appl/FieldmapGPU.f90` | new (B2.6, D1.2): device copies of the field data slots, the per-step element table, `extfld_dev` with kinds 105, 112, 1, 111, 4, 101 |
| `src/Contrl/AccSimulator.f90` | GPU hooks and the GPU step and bend blocks (under `USE_GPU`); `-15` V4, V5 and `-19` switches; N-body in bends; field-file slot search; the fixes above |
| `src/Appl/BeamBunch.f90` | `pt2ptFields` and `pt2ptFields_dev` (pair loop, CPU and GPU, kept identical), restructured `kickpt2ptImg_BeamBunch`, `kickref_Bd_BeamBunch`, `kickpt2ptBd_BeamBunch` (N-body in bends, CPU) |
| `src/Appl/Field.f90` | the GPU Poisson call in `update3Otnew_FieldQuant`; zeroed gradient arrays |
| `src/Appl/Distribution.f90` | read-in on rank 0 (`readslices_Dist`), binary `readbin_Dist` (168), ids |
| `src/Contrl/Output.f90` | GPU sums in `diagnostic1avgZ_Output`, `diagnostic1avgB_Output` and `diagnostic1avg_Output`; binary `-2` dump; `IMPACTT_NODUMP`; NaN fixes; MPI buffer fix |
| `src/Func/Ptclmger.f90` | the exchange fix |
| `src/DataStruct/Data.f90` | reader bounds checks; tables up to 65,540 rows; binary `.T7b` maps |
| `src/DataStruct/NumConst.f90` | `Maxoverlap` 10 → 16 |
| `src/Appl/DTL.f90` | the DTL field fix |
| `utilities/t7_to_t7b.py` | new: `.T7` to `.T7b` converter |
| `src/Func/Timer.f90` | new timers; the rank-0 division |
| `src/mpistub.f90` | rank-2 `MPI_ALLGATHERV`, `MPI_ABORT` |
| `src/CMakeLists.txt` | `USE_GPU` option |
| `tests/` | unit tests, the run comparison scripts, `build_test.sh`, `make_test_inputs.py`, `README.md` |
| `doc/GPU_PORT.md`, `doc/GPU_PORT_DESIGN.md` | this document and the design |

# Appendix A: commits

| Commit | Stage | Description |
|---|---|---|
| `99aa448` | A1 | `pt2ptFields` split out of `kickpt2ptImg_BeamBunch`; gather and image-term fixes; self-image |
| `e72c124` | A4 | `-15` V4 and `-19` switches; the particle-count fix at the N-body call |
| `bf88b7d` | Phase 0 | `USE_GPU` CMake option; `compare_runs.py` |
| `34a875d`, `82436c4`, `79f9f23`, `a48a379` | A2 | `pt2ptFields` on the GPU (OpenACC); per-source γ terms; profiling harness; register cap |
| `31aef91` | A3 | several GPUs for N-body |
| `8e909c2` | B1 | GPU Poisson solve |
| `800fdb8` | B2 | particles, fields and kicks on the GPU |
| `62589e2`, `fb42f8a` | | binary `-2` dump, `Flagdist = 168`; I/O timing prints |
| `c2f7c83` | C1 | `Flagdiag = 2` moments on the GPU |
| `7f86669`, `77398cf`, `7ea5333` | D1 | field kinds for types 1, 111, 4; bend-frame kernels; the bend loop on the GPU |
| `1c3d051` | D1.11 | CSR in the GPU bend loop |
| `6731f11` | D1.12 | N-body in the bend frame, CPU and GPU |
| `6b25811`, `b4c530b`, `97729a8` | | start-of-run warnings; test documentation and synthetic test inputs |
| `6b9a40e` | | particle read-in on rank 0 |
| `75dde0e` | | particle exchange overflow |
| `a6ce288`, `eba221e`, `a9b1112`, `7a98cac`, `f1f8aa0`, `2916fd2`, `fb0ce5f`, `cbfc4e4` | | ids, reader bounds, MPI buffers, diagnostics NaN, uninitialized values, zero-field bend |
| `275120f` | | CPU N-body pair loop in cache-sized blocks; `tests/test_pt2pt_block.f90` |
| `9693d54` | | serial build: 2-D `MPI_SEND`/`MPI_RECV` in the MPI stub |
| `e22e63d`, `64c6b0d` | | `-15` V5 = `r0img`; its GPU test |
| `6a9c46c`, `eaff33d`, `4303006` | | DTL field, bend exit, mesh z clamp |
| `3189cc3`, `97e2772` | | tables up to 65,537 rows; `IMPACTT_NODUMP` |
| `7c5eeff` | | type 101 (DTL) on the GPU; `tests/test_dtl_gpu.f90` |
| `5527e28`, `84237ae`, `5e4d97c`, `94572e6` | | loss removal and `Flagdiag = 1` moments on the device; `tests/test_lostdiag_gpu.f90`; buffer swap |
| `f0f7edc` | | field files kept in memory (`Maxoverlap` 16); binary `.T7b` maps |
| `8c9926b` | | `-15` V3 and V5 in metres, as `master` `3e19b7f` |
| `05aee00` | | `utilities/t7_to_t7b.py` |
| `3fa2241` and later | | these documents |
