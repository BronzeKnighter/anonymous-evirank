#!/usr/bin/env python3
import argparse
import csv

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired two-sided sign-flip permutation test.")
    parser.add_argument("csv_file", help="CSV containing paired per-query metric values")
    parser.add_argument("--left", default="evirank")
    parser.add_argument("--right", default="baseline")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.csv_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or args.left not in rows[0] or args.right not in rows[0]:
        raise SystemExit(f"CSV must contain nonempty {args.left!r} and {args.right!r} columns")

    delta = np.asarray(
        [float(row[args.left]) - float(row[args.right]) for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    signs = rng.choice((-1.0, 1.0), size=(args.permutations, delta.size))
    permuted = np.abs((signs * delta).mean(axis=1))
    p_value = (np.count_nonzero(permuted >= abs(delta.mean())) + 1) / (args.permutations + 1)
    print(f"n={delta.size} mean_delta={delta.mean():.6f} p={p_value:.6f}")


if __name__ == "__main__":
    main()
