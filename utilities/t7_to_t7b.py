#!/usr/bin/env python3
"""Convert an IMPACT-T type-111 3D field map 1T<ID>.T7 (text) to 1T<ID>.T7b (binary).

IMPACT-T reads 1T<ID>.T7b in place of 1T<ID>.T7 when it exists (read3t_Data in
src/DataStruct/Data.f90). The binary file holds the same numbers, so a run gives
the same results; only the read is faster (a 1.35 GB text map: minutes -> seconds).

Usage:
    python3 utilities/t7_to_t7b.py 1T1.T7                 # writes 1T1.T7b
    python3 utilities/t7_to_t7b.py 1T1.T7 -o other.T7b
    python3 utilities/t7_to_t7b.py 1T1.T7 --verify        # also read it back and compare

Text format (as read3t_Data reads it, list-directed):
    xmin xmax Nx        (metres, metres, number of intervals)
    ymin ymax Ny
    zmin zmax Nz
    then one record per grid point, x fastest, then y, then z:
    (Ex) (Ey) (Ez) (Bx) (By) (Bz), each a complex number written "(re, im)".
    Anything after the third number of a header line is ignored, as Fortran does.
    If the file has fewer points than (Nx+1)(Ny+1)(Nz+1), the missing ones are 0,
    as in the text reader.

Binary format (version 1, little-endian stream, no record markers):
    int32  magic   = 1110922313 (the bytes "IT7B")
    int32  version = 1
    3 x { float64 min, float64 max, int32 intervals }      for x, y, z
    Ex, Ey, Ez, Bx, By, Bz: each (Nx+1)(Ny+1)(Nz+1) complex128, x fastest

Keep the .T7b next to its .T7 and remake it whenever the .T7 changes: IMPACT-T
uses the .T7b whenever it exists, so a stale one would be read in place of an
edited text map.

Needs numpy only.
"""
import argparse
import os
import re
import struct
import sys
import warnings

import numpy as np

MAGIC = 1110922313  # bytes "IT7B" as a little-endian int32
VERSION = 1
NCOMP = 6  # Ex, Ey, Ez, Bx, By, Bz


def _fortran_number(tok):
    return float(tok.replace("d", "e").replace("D", "E"))


def _header_line(line, name):
    toks = re.split(r"[\s,]+", line.strip().split("/")[0].strip())
    toks = [t for t in toks if t]
    if len(toks) < 3:
        raise ValueError(f"{name}: header line {line!r} has fewer than 3 numbers")
    lo, hi = _fortran_number(toks[0]), _fortran_number(toks[1])
    n = int(round(_fortran_number(toks[2])))
    return lo, hi, n


def read_t7(path):
    """Return ((xmin,xmax,nx),(ymin,ymax,ny),(zmin,zmax,nz)), fields[6,nx+1,ny+1,nz+1]."""
    with open(path, "r") as f:
        heads = [_header_line(f.readline(), path) for _ in range(3)]
        body = f.read()
    if "*" in body:
        raise ValueError(f"{path}: repeat counts (r*value) are not supported")
    npair = body.count("(")
    if body.count(")") != npair:
        raise ValueError(f"{path}: unbalanced parentheses in the data")
    if npair % NCOMP != 0:
        raise ValueError(f"{path}: {npair} complex values, not a multiple of {NCOMP}")
    # every value is "(re, im)": drop the punctuation and read the numbers in C
    clean = body.translate(str.maketrans({"(": " ", ")": " ", ",": " ", "d": "e", "D": "E"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        vals = np.fromstring(clean, dtype=np.float64, sep=" ")
    if vals.size != 2 * npair:
        raise ValueError(f"{path}: read {vals.size} numbers for {npair} complex values")
    (_, _, nx), (_, _, ny), (_, _, nz) = heads
    npts = (nx + 1) * (ny + 1) * (nz + 1)
    nread = npair // NCOMP
    if nread > npts:
        raise ValueError(f"{path}: {nread} grid points, more than the header's {npts}")
    if nread < npts:
        print(f"warning: {path}: {nread} of {npts} grid points; the rest are 0 "
              "(as in the text reader)", file=sys.stderr)
    cvals = vals.view(np.complex128).reshape(nread, NCOMP)
    fields = np.zeros((NCOMP, npts), dtype=np.complex128)
    fields[:, :nread] = cvals.T
    # point n (0-based) is i + (nx+1)*(j + (ny+1)*k): x fastest, i.e. Fortran order
    fields = fields.reshape(NCOMP, nx + 1, ny + 1, nz + 1, order="F")
    return heads, fields


def write_t7b(path, heads, fields):
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", MAGIC, VERSION))
        for lo, hi, n in heads:
            f.write(struct.pack("<ddi", lo, hi, n))
        for c in range(NCOMP):
            f.write(np.asfortranarray(fields[c]).astype("<c16").tobytes(order="F"))


def read_t7b(path):
    with open(path, "rb") as f:
        magic, version = struct.unpack("<ii", f.read(8))
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{path}: not a version-{VERSION} .T7b file")
        heads = [struct.unpack("<ddi", f.read(20)) for _ in range(3)]
        shape = tuple(n + 1 for _, _, n in heads)
        npts = int(np.prod(shape))
        data = np.frombuffer(f.read(), dtype="<c16")
    if data.size != NCOMP * npts:
        raise ValueError(f"{path}: {data.size} values, expected {NCOMP * npts}")
    fields = data.reshape(NCOMP, npts).reshape((NCOMP,) + shape, order="F")
    return heads, fields


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("t7", help="text map, e.g. 1T1.T7")
    ap.add_argument("-o", "--output", help="binary map (default: the input name + 'b')")
    ap.add_argument("--verify", action="store_true",
                    help="read the binary file back and compare it with the text bit for bit")
    a = ap.parse_args(argv)
    out = a.output or a.t7 + "b"
    if os.path.abspath(out) == os.path.abspath(a.t7):
        ap.error("the output would overwrite the input")
    heads, fields = read_t7(a.t7)
    write_t7b(out, heads, fields)
    shape = "x".join(str(n + 1) for _, _, n in heads)
    print(f"wrote {out}: grid {shape}, {os.path.getsize(out)} bytes")
    if a.verify:
        h2, f2 = read_t7b(out)
        same = ([tuple(h) for h in heads] == [tuple(h) for h in h2]
                and fields.astype("<c16").tobytes(order="F") == f2.tobytes(order="F"))
        print("verify: identical" if same else "verify: DIFFERENT")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
