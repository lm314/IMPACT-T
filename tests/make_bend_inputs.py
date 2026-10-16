#!/usr/bin/env python3
"""
Write synthetic input files for test_bend_gpu into a directory (standard library only):

    python3 tests/make_bend_inputs.py RUN_DIR

- rfdata111  quadrupole gradient profile (type 1, file mode): a smooth flat-top over 0..0.262 m
             with Enge-like edges; rows G, G', G'', G''' on a uniform grid (read1tdata_Data).
- rfdata15   dipole data (type 4): CSR flag 0, reference gamma, four straight pole-face lines
             at z = 0, 0.15, 0.15, 0.3 m, eight Enge coefficients and the two fringe lengths
             (read1t_Data).
- 1T7.T7     3D Cartesian map (type 111), x, y in -5..5 mm, z in 0..0.108 m: a standing-wave
             deflecting-mode-like field, E real and B imaginary, every component nonzero (read3t_Data).

The fields are made up: the test compares the GPU evaluation of each file with the CPU routine
on the same data, so only smoothness and nonzero components matter.
"""
import math
import sys
from pathlib import Path


def quad_profile(path, n=1000, z0=0.0, z1=0.262, a=0.03, b=0.232, w=0.006):
    def g(z):
        return 1.0 / (1.0 + math.exp(-(z - a) / w)) / (1.0 + math.exp(-(b - z) / w))

    h = 1e-4
    def d(f, z, k):  # k-th derivative by nested central differences
        if k == 0:
            return f(z)
        return (d(f, z + h, k - 1) - d(f, z - h, k - 1)) / (2 * h)

    with open(path, "w") as f:
        f.write(f"{n} {z0!r} {z1!r} {z1 - z0!r}\n")
        for i in range(n):
            z = z0 + (z1 - z0) * i / (n - 1)
            f.write(" ".join(f"{d(g, z, k):.17e}" for k in range(4)) + "\n")
        f.write("1 0.0 0.0 0.0\n0.0 0.0 0.0 0.0\n")


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


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    quad_profile(out / "rfdata111")
    dipole(out / "rfdata15")
    cart_map(out / "1T7.T7")
    print(f"wrote {out}/rfdata111, rfdata15, 1T7.T7")
