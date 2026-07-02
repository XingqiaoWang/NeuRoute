#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser(
        description="Sample a subdataset from a .npy file (without replacement, no memmap)."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Path to input .npy")
    ap.add_argument("--out", required=True, help="Path to output .npy")
    ap.add_argument("--n", type=int, default=None, help="Number of samples to take")
    ap.add_argument("--frac", type=float, default=None, help="Fraction in (0,1] to sample")
    ap.add_argument("--axis", type=int, default=0, help="Axis to sample along (default: 0)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--save-indices", default=None, help="Optional path to save chosen indices (.npy)")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load entire array into RAM (no memmap)
    arr = np.load(args.inp, allow_pickle=False)
    ndim = arr.ndim
    axis = args.axis if args.axis >= 0 else args.axis + ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"axis={args.axis} out of range for array with ndim={ndim}")

    N = arr.shape[axis]

    # Decide sample size
    if (args.n is None) == (args.frac is None):
        raise ValueError("Specify exactly one of --n or --frac")

    if args.frac is not None:
        if not (0.0 < args.frac <= 1.0):
            raise ValueError("--frac must be in (0,1]")
        n_samples = max(1, int(math.floor(N * args.frac)))
    else:
        n_samples = int(args.n)
        if not (1 <= n_samples <= N):
            raise ValueError(f"--n must be in [1, {N}]")

    # Choose indices without replacement
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=n_samples, replace=False)
    idx.sort()  # optional, for stable ordering

    # Take along axis and save
    sampled = np.take(arr, idx, axis=axis)
    np.save(args.out, sampled)
    print(f"[OK] Wrote {n_samples} samples to {args.out} (shape {sampled.shape}, dtype={sampled.dtype})")

    if args.save_indices:
        np.save(args.save_indices, idx)
        print(f"[OK] Saved sampled indices to {args.save_indices}")

if __name__ == "__main__":
    main()
