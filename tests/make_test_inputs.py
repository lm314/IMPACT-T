#!/usr/bin/env python3
"""
Write synthetic input files for the GPU tests into a directory (standard library only):

    python3 tests/make_test_inputs.py bend RUN_DIR       # for test_bend_gpu
    python3 tests/make_test_inputs.py particle RUN_DIR   # for test_particle_gpu

bend:
- rfdata111  quadrupole gradient profile (type 1, file mode): a smooth flat-top over 0..0.262 m
             with Enge-like edges; rows G, G', G'', G''' on a uniform grid (read1tdata_Data).
- rfdata15   dipole data (type 4): CSR flag 0, reference gamma, four straight pole-face lines
             at z = 0, 0.15, 0.15, 0.3 m, eight Enge coefficients and the two fringe lengths
             (read1t_Data).
- 1T7.T7     3D Cartesian map (type 111), x, y in -5..5 mm, z in 0..0.108 m: a standing-wave
             deflecting-mode-like field, E real and B imaginary, every component nonzero (read3t_Data).

particle:
- rfdata1001 RF cell (type 105, discrete on-axis data, file ID > 1000), 0..5.5 mm: Ez = sin(pi z/L).
- rfdata1002 RF cell, 0..12 mm: Ez = sin(2 pi z/L).
- rfdata1003 solenoid (type 105, no E field, Bz in the second block), 0..0.2 m: a Gaussian Bz.
  Each file is a block "n z0 z1 L" + n rows (f, f', f'', f''') for Ez, then one for Bz; an empty
  block is "1 0 0 0" and one row of zeros (read1tdata_Data).
- 1T1.T7     the type-112 map of examples/Sample4_2Dfield, copied from this repository.

The fields are made up: the tests compare the GPU evaluation of each file with the CPU routine
on the same data, so only smoothness and nonzero components matter.
"""
import math
import shutil
import sys
from pathlib import Path

EMPTY_BLOCK = "1 0.0 0.0 0.0\n0.0 0.0 0.0 0.0\n"


def block(n, length, funcs):
    """A read1tdata_Data block: n points on 0..length, funcs = (f, f', f'', f''') of z."""
    out = [f"{n} 0.0 {length!r} {length!r}"]
    for i in range(n):
        z = length * i / (n - 1)
        out.append(" ".join(f"{g(z):.17e}" for g in funcs))
    return "\n".join(out) + "\n"


def quad_profile(path, n=1000, z1=0.262, a=0.03, b=0.232, w=0.006):
    def g(z):
        return 1.0 / (1.0 + math.exp(-(z - a) / w)) / (1.0 + math.exp(-(b - z) / w))

    h = 1e-4

    def d(z, k):  # k-th derivative by nested central differences
        return g(z) if k == 0 else (d(z + h, k - 1) - d(z - h, k - 1)) / (2 * h)

    Path(path).write_text(block(n, z1, [lambda z, k=k: d(z, k) for k in range(4)]) + EMPTY_BLOCK)


def dipole(path):
    gamma = 40.0
    faces = [0.0, 0.0, 0.0, 0.15, 0.0, 0.15, 0.0, 0.3]   # (slope, intercept) of the four face lines
    widths = [0.15, 0.15]
    enge = [0.478959, 1.911289, -1.185953, 1.630554, -1.082657, 0.318111, 0.0, 0.0]
    lengths = [0.075, 0.225]
    vals = [0.0, gamma] + faces + widths + enge + lengths
    Path(path).write_text("\n".join(repr(v) for v in vals) + "\n")


def cart_map(path, nx=20, ny=20, nz=108, half=0.005, length=0.108, e0=3.0e6, b0=1.0e-2):
    with open(path, "w") as f:
        f.write(f"{-half!r} {half!r} {nx}\n{-half!r} {half!r} {ny}\n0.0 {length!r} {nz}\n")
        for k in range(nz + 1):
            z = length * k / nz
            s, c = math.sin(math.pi * z / length), math.cos(math.pi * z / length)
            for j in range(ny + 1):
                y = -half + 2 * half * j / ny
                for i in range(nx + 1):
                    x = -half + 2 * half * i / nx
                    u, v = x / half, y / half
                    e = (e0 * s * (1 + 0.1 * u * u), e0 * 0.2 * s * u * v, e0 * 0.5 * c * u)
                    bb = (b0 * 0.3 * s * u * v, b0 * s * (1 + 0.2 * v * v), b0 * 0.1 * c * v)
                    vals = [complex(t, 0.02 * t) for t in e] + [complex(0.0, t) for t in bb]
                    f.write(" ".join(f"({t.real:.10e},{t.imag:.10e})" for t in vals) + "\n")


def rf_cell(path, n, length, e0, periods):
    k = math.pi * periods / length
    funcs = (lambda z: e0 * math.sin(k * z), lambda z: e0 * k * math.cos(k * z),
             lambda z: -e0 * k * k * math.sin(k * z), lambda z: -e0 * k ** 3 * math.cos(k * z))
    Path(path).write_text(block(n, length, funcs) + EMPTY_BLOCK)


def solenoid(path, n=201, length=0.2, b0=0.3, zc=0.1, w=0.03):
    def g(z, k):  # d^k/dz^k of b0 exp(-u^2/2), u = (z - zc)/w
        u = (z - zc) / w
        herm = (1.0, -u, u * u - 1.0, -u ** 3 + 3 * u)[k]
        return b0 * herm * math.exp(-0.5 * u * u) / w ** k

    Path(path).write_text(EMPTY_BLOCK + block(n, length, [lambda z, k=k: g(z, k) for k in range(4)]))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("bend", "particle"):
        sys.exit(__doc__)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "bend":
        quad_profile(out / "rfdata111")
        dipole(out / "rfdata15")
        cart_map(out / "1T7.T7")
        print(f"wrote {out}/rfdata111, rfdata15, 1T7.T7")
    else:
        rf_cell(out / "rfdata1001", 201, 0.0055, 1.25e8, 1)
        rf_cell(out / "rfdata1002", 202, 0.012, 1.0e8, 2)
        solenoid(out / "rfdata1003")
        here = Path(__file__).resolve().parent
        shutil.copy(here.parent / "examples" / "Sample4_2Dfield" / "1T1.T7", out / "1T1.T7")
        print(f"wrote {out}/rfdata1001, rfdata1002, rfdata1003, 1T1.T7")
