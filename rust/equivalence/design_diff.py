#!/usr/bin/env python3
"""Diff the design matrices the two implementations hand to the elastic net.

R's dumps (design_probe.R) are numbered by call order; the port's are named by
target. They are paired by matching the response column, so no assumption is
made about the order MORE happens to iterate targets in.

Usage: design_diff.py <r_dump_dir> <rust_dump_dir>
"""
import sys
from pathlib import Path


def read(p):
    lines = [l.rstrip("\n") for l in p.open() if l.strip()]
    header = lines[0].split("\t")
    rows = [[float(v) for v in l.split("\t")] for l in lines[1:]]
    cols = {name: [r[j] for r in rows] for j, name in enumerate(header)}
    return header, cols


def key(vals):
    return tuple(round(v, 9) for v in vals)


def main():
    rdir, sdir = Path(sys.argv[1]), Path(sys.argv[2])
    rmats = {}
    for f in sorted(rdir.glob("*.tsv")):
        h, c = read(f)
        rmats[key(c[h[0]])] = (f.name, h, c)

    same_shape = same_names = same_values = 0
    problems = []
    for f in sorted(sdir.glob("*.tsv")):
        h, c = read(f)
        target = f.stem
        hit = rmats.get(key(c[h[0]]))
        if hit is None:
            problems.append(f"{target}: no R design matrix has this response column")
            continue
        rname, rh, rc = hit
        rpred, spred = rh[1:], h[1:]
        if len(rpred) != len(spred):
            problems.append(f"{target}: {len(spred)} predictors vs R's {len(rpred)} ({rname})")
            continue
        same_shape += 1
        if rpred != spred:
            only_r = [x for x in rpred if x not in spred]
            only_s = [x for x in spred if x not in rpred]
            problems.append(f"{target}: column names differ; only-R={only_r[:6]} only-Rust={only_s[:6]}")
            continue
        same_names += 1
        worst, where = 0.0, None
        for name in rpred:
            for a, b in zip(rc[name], c[name]):
                d = abs(a - b)
                if d > worst:
                    worst, where = d, name
        if worst > 1e-9:
            problems.append(f"{target}: values differ, max |delta| = {worst:.3e} at column {where}")
            continue
        same_values += 1

    total = len(list(sdir.glob("*.tsv")))
    print(f"targets compared        : {total}")
    print(f"same predictor count    : {same_shape}")
    print(f"same column names       : {same_names}")
    print(f"identical values (1e-9) : {same_values}")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print("  " + p)
        return 1
    print("\nevery design matrix is identical to R's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
