#!/usr/bin/env python3
"""Diff the port's elastic-net cross-validation against real cv.glmnet.

Runs `en_probe.R` (which dumps the design matrices MORE hands to ElasticNet
and reports what glmnet decides for each of the eleven alphas), then runs the
binary's `MORE_RS_EN_PROBE` mode over the same TSVs, and compares row by row.

The decision columns -- which alpha wins, how many lambdas the path has, and
how many coefficients survive -- are compared exactly. cvm/cvup are floating
point and compared relatively; a disagreement there only matters when it
changes a decision, which the exact columns already catch.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "target" / "release" / "more-rs"
FIXTURES = HERE / "fixtures" / "en"

ROW = re.compile(
    r"a=(?P<alpha>[\d.]+)\s+nlam=\s*(?P<nlam>\d+)\s+lmax=(?P<lmax>\S+)\s+"
    r"lmin_path=(?P<lmin>\S+)\s+lambda\.min=(?P<lmin_sel>\S+)\s+"
    r"cvm=(?P<cvm>\S+)\s+cvup=(?P<cvup>\S+)\s+nz=(?P<nz>\d+)"
)
MATRIX = re.compile(r"^== matrix (\d+)")
WINNER = re.compile(r"WINNER a=(?P<alpha>[\d.]+)\s+cvup=(?P<cvup>\S+)")


def parse(text):
    """-> {matrix_id: {"rows": {alpha: dict}, "winner": alpha}}"""
    out, cur = {}, None
    for line in text.splitlines():
        m = MATRIX.match(line)
        if m:
            cur = int(m.group(1))
            out[cur] = {"rows": {}, "winner": None}
            continue
        if cur is None:
            continue
        m = ROW.search(line)
        if m:
            d = m.groupdict()
            out[cur]["rows"][round(float(d["alpha"]), 1)] = {
                "nlam": int(d["nlam"]),
                "nz": int(d["nz"]),
                "lambda_min": float(d["lmin_sel"]),
                "cvm": float(d["cvm"]),
                "cvup": float(d["cvup"]),
            }
            continue
        m = WINNER.search(line)
        if m:
            out[cur]["winner"] = round(float(m.group("alpha")), 1)
    return out


def rel(a, b):
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale


def main():
    r_out = subprocess.run(
        ["Rscript", str(HERE / "en_probe.R"), str(FIXTURES)],
        cwd=HERE, capture_output=True, text=True,
    ).stdout
    r = parse(r_out)
    if not r:
        sys.exit("en_probe.R produced no parsable output:\n" + r_out[-2000:])

    # The Rust side prints one matrix at a time; label each block the same way.
    chunks = []
    for i in sorted(r):
        f = FIXTURES / f"des_{i:02d}.tsv"
        if not f.exists():
            continue
        got = subprocess.run(
            [str(BIN)], env={"MORE_RS_EN_PROBE": str(f), "PATH": "/usr/bin:/bin", "MORE_RS_EN_THRESH": os.environ.get("MORE_RS_EN_THRESH", "1e-5")},
            capture_output=True, text=True,
        ).stdout
        chunks.append(f"== matrix {i}\n{got}")
    rs = parse("\n".join(chunks))

    hard = []          # decision-level disagreements
    worst_cvm = 0.0
    worst_lam = 0.0
    n_rows = 0
    for i in sorted(r):
        if i not in rs:
            continue
        if r[i]["winner"] != rs[i]["winner"]:
            hard.append(f"matrix {i}: winning alpha R={r[i]['winner']} rust={rs[i]['winner']}")
        for a, rr in sorted(r[i]["rows"].items()):
            rrs = rs[i]["rows"].get(a)
            if rrs is None:
                hard.append(f"matrix {i} a={a}: missing on the rust side")
                continue
            n_rows += 1
            if rr["nlam"] != rrs["nlam"]:
                hard.append(f"matrix {i} a={a}: nlam R={rr['nlam']} rust={rrs['nlam']}")
            if rr["nz"] != rrs["nz"]:
                hard.append(f"matrix {i} a={a}: nonzero R={rr['nz']} rust={rrs['nz']}")
            worst_cvm = max(worst_cvm, rel(rr["cvm"], rrs["cvm"]))
            worst_lam = max(worst_lam, rel(rr["lambda_min"], rrs["lambda_min"]))

    print(f"matrices compared : {len(rs)}")
    print(f"alpha rows compared: {n_rows}")
    print(f"max relative |cvm| difference       : {worst_cvm:.3e}")
    print(f"max relative |lambda.min| difference: {worst_lam:.3e}")
    if hard:
        print(f"\nDECISION DISAGREEMENTS ({len(hard)}):")
        for h in hard:
            print("  " + h)
        return 1
    print("\nevery decision column agrees: nlam, nonzero count, winning alpha")
    return 0


if __name__ == "__main__":
    sys.exit(main())
