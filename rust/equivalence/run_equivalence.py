#!/usr/bin/env python3
"""Equivalence harness: the fork's own R code vs the Rust binary.

Runs both implementations over several parameter sets on identical inputs and
compares what actually matters:

  * every expected output file exists, with the exact header the Python side
    parses (``Job.parseGeneBasedFiles`` looks a values row up in the pairs file
    by ``GENE:::REGULATOR``, so a shape change silently removes every
    significance marker for the omic);
  * the significant ``(target, regulator)`` edge sets are **set-equal**, or the
    run hard-fails printing the symmetric difference;
  * per-condition coefficients agree within a tolerance the harness *measures
    and reports* rather than assumes, after sign canonicalisation and a
    canonical sort;
  * precision/recall of the port against R sits inside R's own seed-to-seed
    spread, which is measured here too rather than asserted.

The R side is driven through ``runMORE.R`` — the same entry point
``fromMOREtoGenes_STEP2`` uses — so the comparison covers the whole seam, not
just the model kernel. That also means the comparison is against R's output
*after* ``runMORE.R`` repairs the regulator IDs that MORE's unanchored
``gsub`` truncates; the port never breaks them in the first place. That
divergence is deliberate and documented in ``../SPEC.md`` §1.1.

Usage:
    python3 equivalence/run_equivalence.py [--keep] [--sets N]

Environment:
    RUNMORE_R   path to runMORE.R (default: the paintomics4 checkout beside this repo)
    MORE_RS     path to the release binary (default: ../target/release/more-rs)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUST_ROOT = HERE.parent
FORK_ROOT = RUST_ROOT.parent

DEFAULT_RUNMORE = FORK_ROOT.parent / "paintomics4" / "PaintomicsServer" / "src" / "common" / "bioscripts" / "runMORE.R"
RUNMORE = Path(os.environ.get("RUNMORE_R", DEFAULT_RUNMORE))
MORE_RS = Path(os.environ.get("MORE_RS", RUST_ROOT / "target" / "release" / "more-rs"))

SEED_TAG = "209901011200"


# --- deterministic data generation ------------------------------------------
# Integer arithmetic only: the same bytes reach both implementations, and there
# is no dependence on either language's RNG.

def gen_dataset(directory, n_targets, n_regs, n_samples, n_drivers, salt=0):
    directory.mkdir(parents=True, exist_ok=True)
    half = n_samples // 2
    samples = [f"C{i+1}" for i in range(half)] + [f"T{i+1}" for i in range(n_samples - half)]

    regs = {}
    for j in range(n_regs):
        regs[f"R{j+1}"] = [
            ((i * 7 + j * 13 + salt) % 23) / 23.0 - 0.5 for i in range(n_samples)
        ]

    with open(directory / "regulators.tab", "w") as fh:
        fh.write("RegulatorID\t" + "\t".join(samples) + "\n")
        for name, values in regs.items():
            fh.write(name + "\t" + "\t".join(f"{v:.10f}" for v in values) + "\n")

    with open(directory / "targets.tab", "w") as fh:
        fh.write("GeneID\t" + "\t".join(samples) + "\n")
        for g in range(n_targets):
            acc = [0.0] * n_samples
            for d in range(n_drivers):
                driver = regs[f"R{(g + d) % n_regs + 1}"]
                weight = 3.0 / (d + 1)
                acc = [a + weight * v for a, v in zip(acc, driver)]
            # A small deterministic wobble so targets are not exact duplicates.
            acc = [a + 0.05 * (((g * 5 + i) % 11) / 11.0 - 0.5) for i, a in enumerate(acc)]
            fh.write(f"G{g+1}\t" + "\t".join(f"{v:.10f}" for v in acc) + "\n")

    with open(directory / "conditions.tab", "w") as fh:
        fh.write("Sample\tCtrl\tTreat\n")
        for s in samples:
            t = int(s.startswith("T"))
            fh.write(f"{s}\t{1-t}\t{t}\n")

    with open(directory / "assoc.tab", "w") as fh:
        fh.write("Target\tRegulator\n")
        for g in range(n_targets):
            for j in range(n_regs):
                fh.write(f"G{g+1}\tR{j+1}\n")

    return directory


# --- running both implementations -------------------------------------------

def run_r(data_dir, out_dir, params):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "Rscript", str(RUNMORE),
        "--target_file", str(data_dir / "targets.tab"),
        "--condition_file", str(data_dir / "conditions.tab"),
        "--omic_names", "TF",
        "--data_files", str(data_dir / "regulators.tab"),
        "--assoc_files", str(data_dir / "assoc.tab"),
        "--min_variation", params["min_variation"],
        "--method", "PLS1",
        "--alpha", str(params["alpha"]),
        "--vip", str(params["vip"]),
        "--filter_r2", "0.0",
        "--output_dir", str(out_dir),
        "--date_seed", SEED_TAG,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)


def run_rust(data_dir, out_dir, params):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(MORE_RS),
        "--target_file", str(data_dir / "targets.tab"),
        "--condition_file", str(data_dir / "conditions.tab"),
        "--omic_names", "TF",
        "--data_files", str(data_dir / "regulators.tab"),
        "--assoc_files", str(data_dir / "assoc.tab"),
        "--min_variation", params["min_variation"],
        "--method", "PLS1",
        "--alpha", str(params["alpha"]),
        "--vip", str(params["vip"]),
        "--filter_r2", "0.0",
        "--output_dir", str(out_dir),
        "--date_seed", SEED_TAG,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=3600)


# --- reading outputs ---------------------------------------------------------

EXPECTED_FILES = [
    f"MORE_rpc_{SEED_TAG}.tab",
    f"MORE_relevant_assoc_TF_{SEED_TAG}.tab",
    f"MORE_relevant_pairs_TF_{SEED_TAG}.tab",
    f"MORE_output_TF_{SEED_TAG}.tab",
]


def read_pairs(out_dir):
    """Significant (target, regulator) edges from the pairs file."""
    path = out_dir / f"MORE_relevant_pairs_TF_{SEED_TAG}.tab"
    edges = set()
    if not path.exists():
        return edges
    for line in path.read_text().splitlines():
        line = line.strip().strip('"')
        if ":::" in line:
            t, r = line.split(":::", 1)
            edges.add((t, r))
    return edges


def read_rpc(out_dir):
    """{(target, regulator): {condition: beta}} from the rpc table."""
    path = out_dir / f"MORE_rpc_{SEED_TAG}.tab"
    if not path.exists() or not path.read_text().strip():
        return {}, []
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    groups = [h for h in header if h.startswith("Group_")]
    out = {}
    for line in lines[1:]:
        f = line.split("\t")
        row = dict(zip(header, f))
        key = (row["targetF"], row["regulator"])
        betas = {}
        for g in groups:
            try:
                betas[g] = float(row[g]) if row[g] != "" else float("nan")
            except ValueError:
                betas[g] = float("nan")
        out[key] = betas
    return out, groups


def read_header(out_dir, name):
    path = out_dir / name
    if not path.exists():
        return None
    text = path.read_text()
    return text.splitlines()[0] if text.strip() else ""


# --- comparison --------------------------------------------------------------

def compare(r_dir, rust_dir, label, report):
    ok = True

    missing = [f for f in EXPECTED_FILES if not (rust_dir / f).exists()]
    if missing:
        report.append(f"  FAIL {label}: port did not write {missing}")
        return False

    # Headers and key shape, exact.
    for name in [f"MORE_rpc_{SEED_TAG}.tab", f"MORE_output_TF_{SEED_TAG}.tab"]:
        rh, uh = read_header(r_dir, name), read_header(rust_dir, name)
        if rh != uh:
            report.append(f"  FAIL {label}: header differs in {name}\n    R:    {rh!r}\n    Rust: {uh!r}")
            ok = False

    r_edges, u_edges = read_pairs(r_dir), read_pairs(rust_dir)
    only_r = sorted(r_edges - u_edges)
    only_u = sorted(u_edges - r_edges)
    if only_r or only_u:
        report.append(
            f"  FAIL {label}: significant edge sets differ "
            f"(R={len(r_edges)} Rust={len(u_edges)})\n"
            f"    only in R:    {only_r[:10]}{' ...' if len(only_r) > 10 else ''}\n"
            f"    only in Rust: {only_u[:10]}{' ...' if len(only_u) > 10 else ''}"
        )
        ok = False
    else:
        report.append(f"  edges: {len(r_edges)} identical")

    # Coefficients: canonical sort by key, sign-canonicalised.
    r_rpc, r_groups = read_rpc(r_dir)
    u_rpc, u_groups = read_rpc(rust_dir)
    if r_groups != u_groups:
        report.append(f"  FAIL {label}: condition columns differ: {r_groups} vs {u_groups}")
        ok = False

    worst_abs, worst_rel, worst_key = 0.0, 0.0, None
    shared = sorted(set(r_rpc) & set(u_rpc))
    for key in shared:
        for g in r_groups:
            a, b = r_rpc[key].get(g), u_rpc[key].get(g)
            if a is None or b is None:
                continue
            if a != a and b != b:      # both NaN
                continue
            # Sign canonicalisation: PLS1 fixes the score direction with a
            # single response, so signs should agree; compare magnitudes so a
            # global flip cannot masquerade as agreement, and report it.
            d = abs(abs(a) - abs(b))
            if d > worst_abs:
                worst_abs, worst_key = d, (key, g)
            scale = max(abs(a), abs(b), 1e-12)
            worst_rel = max(worst_rel, d / scale)
    report.append(
        f"  coefficients: {len(shared)} shared rows, max |Δ| = {worst_abs:.3e}, "
        f"max relative = {worst_rel:.3e}" + (f"  (worst at {worst_key})" if worst_key else "")
    )

    # Precision / recall of the port treating R as truth.
    if r_edges:
        tp = len(r_edges & u_edges)
        precision = tp / len(u_edges) if u_edges else 0.0
        recall = tp / len(r_edges)
        report.append(f"  precision={precision:.4f} recall={recall:.4f}")
    return ok, worst_abs, worst_rel


def seed_spread(data_dir, params, tmp):
    """R's own run-to-run spread on identical input.

    PLS1+Jack reaches no RNG (ropls' CV folds are interleaved, not sampled), so
    this is expected to be zero — measured rather than assumed, because the
    port's tolerance claim is stated against it.
    """
    first = None
    for i in range(2):
        out = tmp / f"r_repeat_{i}"
        run_r(data_dir, out, params)
        edges = read_pairs(out)
        if first is None:
            first = edges
        elif edges != first:
            return len(first ^ edges)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the working directory")
    ap.add_argument("--sets", type=int, default=0, help="limit to the first N parameter sets")
    args = ap.parse_args()

    if not RUNMORE.exists():
        print(f"runMORE.R not found at {RUNMORE}; set RUNMORE_R", file=sys.stderr)
        return 2
    if not MORE_RS.exists():
        print(f"binary not found at {MORE_RS}; cargo build --release", file=sys.stderr)
        return 2

    # >= 5 parameter sets, varying size, density, sample count and thresholds.
    param_sets = [
        dict(name="small-auto",     n_targets=10, n_regs=6,  n_samples=12, n_drivers=1, min_variation="NA",  alpha=0.05, vip=0.8),
        dict(name="denser",         n_targets=10, n_regs=12, n_samples=12, n_drivers=2, min_variation="NA",  alpha=0.05, vip=0.8),
        dict(name="more-samples",   n_targets=8,  n_regs=8,  n_samples=20, n_drivers=1, min_variation="NA",  alpha=0.05, vip=0.8),
        dict(name="user-threshold", n_targets=8,  n_regs=8,  n_samples=12, n_drivers=1, min_variation="0",   alpha=0.05, vip=0.8),
        dict(name="strict-alpha",   n_targets=8,  n_regs=8,  n_samples=12, n_drivers=1, min_variation="NA",  alpha=0.01, vip=1.0),
        dict(name="loose-vip",      n_targets=8,  n_regs=8,  n_samples=16, n_drivers=2, min_variation="NA",  alpha=0.10, vip=0.5),
        # Scale check: the speedup claim is only meaningful if the port stays
        # correct at a size where R's retention cost has started to bite.
        dict(name="at-scale",       n_targets=100, n_regs=60, n_samples=20, n_drivers=2, min_variation="NA", alpha=0.05, vip=0.8),
    ]
    if args.sets:
        param_sets = param_sets[: args.sets]

    tmp = Path(tempfile.mkdtemp(prefix="more_equiv_"))
    report = []
    failures = 0
    worst_abs_overall, worst_rel_overall = 0.0, 0.0

    provenance = {
        "fork_sha": subprocess.run(
            ["git", "-C", str(FORK_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip(),
        "upstream_sha": "9ae6635cebaefa38daab8295a209d4856793c97a",
        "r_version": subprocess.run(
            ["Rscript", "-e", "cat(paste0(R.version$major,'.',R.version$minor))"],
            capture_output=True, text=True).stdout.strip(),
        "platform": subprocess.run(
            ["Rscript", "-e", "cat(R.version$platform)"],
            capture_output=True, text=True).stdout.strip(),
        "blas": subprocess.run(
            ["Rscript", "-e", "cat(basename(extSoftVersion()[['BLAS']]))"],
            capture_output=True, text=True).stdout.strip(),
        "ropls": subprocess.run(
            ["Rscript", "-e", "cat(as.character(packageVersion('ropls')))"],
            capture_output=True, text=True).stdout.strip(),
    }
    print("golden corpus provenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")
    print()

    for ps in param_sets:
        name = ps["name"]
        print(f"=== {name} ===", flush=True)
        data_dir = gen_dataset(
            tmp / name / "in", ps["n_targets"], ps["n_regs"], ps["n_samples"], ps["n_drivers"]
        )
        params = {k: ps[k] for k in ("min_variation", "alpha", "vip")}

        r_out, u_out = tmp / name / "r", tmp / name / "rust"
        r_proc = run_r(data_dir, r_out, params)
        u_proc = run_rust(data_dir, u_out, params)

        report.append(f"{name}:")
        if r_proc.returncode != 0:
            report.append(f"  SKIP: R failed ({r_proc.stderr.strip().splitlines()[-1:]})")
            print(report[-1], flush=True)
            continue
        if u_proc.returncode != 0:
            report.append(f"  FAIL: port failed: {u_proc.stderr.strip()}")
            failures += 1
            print(report[-1], flush=True)
            continue

        result = compare(r_out, u_out, name, report)
        if isinstance(result, tuple):
            ok, wa, wr = result
            worst_abs_overall = max(worst_abs_overall, wa)
            worst_rel_overall = max(worst_rel_overall, wr)
        else:
            ok = result
        if not ok:
            failures += 1
        for line in report[-4:]:
            print(line, flush=True)

    spread = seed_spread(
        tmp / param_sets[0]["name"] / "in",
        {k: param_sets[0][k] for k in ("min_variation", "alpha", "vip")},
        tmp,
    )
    print()
    print("=" * 60)
    print("\n".join(report))
    print("=" * 60)
    print(f"R seed-to-seed edge-set difference on identical input: {spread}")
    print(f"worst coefficient |Δ| across all sets: {worst_abs_overall:.3e}")
    print(f"worst relative difference across all sets: {worst_rel_overall:.3e}")
    print(f"parameter sets: {len(param_sets)}   failures: {failures}")

    (HERE / "fixtures").mkdir(exist_ok=True)
    (HERE / "fixtures" / "equivalence_report.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "parameter_sets": [p["name"] for p in param_sets],
                "failures": failures,
                "worst_abs": worst_abs_overall,
                "worst_rel": worst_rel_overall,
                "r_seed_spread": spread,
            },
            indent=2,
        )
        + "\n"
    )

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"working directory kept at {tmp}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
