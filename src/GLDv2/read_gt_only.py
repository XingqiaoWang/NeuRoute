#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
read_gt_only.py

Load only GT_I and GT_D from a ground-truth JSON file.

Enhancements:
  1) Supports unwrapping nested JSON structures commonly used in ANN benchmarks.
  2) Optional slicing to first-k neighbors.
  3) Optional saving to NPY files.
  4) NEW: Optional manual transform for gt_D (e.g., 1-y for cosine similarity).

Typical use cases:
  - GT_I: ground truth neighbor ids, shape [nq, k]
  - GT_D: ground truth distances or similarities, shape [nq, k] (optional)

Examples:
  # Default: keep gt_D as is (assume it's already L2 / L2^2)
  python read_gt_only.py --gt xxx.json --n_vectors 10000000 --k 10 --save_prefix out/gt

  # If gt_D is cosine similarity, convert to cosine distance by 1 - y
  python read_gt_only.py --gt xxx.json --k 10 --D_transform one_minus --save_prefix out/gt

  # If gt_D is cosine similarity and you want L2^2 (unit vectors required):
  python read_gt_only.py --gt xxx.json --k 10 --D_transform cos_sim_to_l2sq --save_prefix out/gt

  # If gt_D is cosine distance (1-cos), convert to L2^2:
  python read_gt_only.py --gt xxx.json --k 10 --D_transform cos_dist_to_l2sq --save_prefix out/gt
"""

import argparse
import json
import sys
from typing import Any, Tuple, Optional

import numpy as np


def eprint(*a, **k):
    """Print to stderr."""
    print(*a, file=sys.stderr, **k)


def _unwrap_gt_root(obj: Any, n_vectors: Optional[int] = None) -> Any:
    """
    Try to unwrap common ground-truth JSON structures.

    Supports patterns like:
      {"cosine": {"<n_vectors>": {...}}}
      {"l2": {"<n_vectors>": {...}}}
      {"ip": {"<n_vectors>": {...}}}
      {"<n_vectors>": {...}}
      {"ground_truth_indices": ..., "ground_truth_distances": ...}

    Returns the innermost dict that contains ground_truth_indices/ground_truth_distances.
    """
    if not isinstance(obj, dict):
        return obj

    # Common metric wrapper keys
    for metric_key in ("cosine", "ip", "l2", "l2sq", "inner_product"):
        if metric_key in obj and isinstance(obj[metric_key], dict):
            obj = obj[metric_key]
            break

    if not isinstance(obj, dict):
        return obj

    # If n_vectors provided, try that first
    if n_vectors is not None:
        nk = str(int(n_vectors))
        if nk in obj and isinstance(obj[nk], dict):
            return obj[nk]

    # If there is exactly one digit key, unwrap it
    if len(obj) == 1:
        only_key = next(iter(obj.keys()))
        if isinstance(only_key, str) and only_key.isdigit() and isinstance(obj[only_key], dict):
            return obj[only_key]

    return obj


def _apply_D_transform(
    gt_D: np.ndarray,
    transform: str,
    clip_cosine: bool = True,
) -> np.ndarray:
    """
    Apply manual transform to gt_D.

    Supported transforms:
      - "none": keep as is
      - "one_minus": y <- 1 - y
            (cosine similarity -> cosine distance)
      - "cos_sim_to_l2sq": L2^2 = 2 - 2*cos_sim
            (requires L2-normalized vectors)
      - "cos_dist_to_l2sq": L2^2 = 2*cos_dist
            (requires L2-normalized vectors)

    Notes:
      - This function does NOT square L2 distances automatically.
      - It only applies what you ask.
    """
    D = np.asarray(gt_D, dtype=np.float32)

    t = transform.lower().strip()
    if t == "none":
        return D

    if t == "one_minus":
        return (1.0 - D).astype(np.float32, copy=False)

    if t == "cos_sim_to_l2sq":
        # Numerical safety for cosine similarity
        if clip_cosine:
            D = np.clip(D, -1.0, 1.0)
        # L2^2 = 2 - 2*cos
        return (2.0 - 2.0 * D).astype(np.float32, copy=False)

    if t == "cos_dist_to_l2sq":
        # cosine distance should be >= 0, but clip small negatives
        D = np.maximum(D, 0.0)
        # L2^2 = 2 * cos_dist
        return (2.0 * D).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported --D_transform: {transform}")


def load_gt_I_D(gt_path: str, n_vectors: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load GT indices (gt_I) and GT distances/similarities (gt_D) from JSON.

    Expected keys:
      - ground_truth_indices (required)
      - ground_truth_distances OR ground_truth_D (optional)

    Returns:
      gt_I: int64 [nq, k]
      gt_D: float32 [nq, k] or None
    """
    with open(gt_path, "r") as f:
        obj: Any = json.load(f)

    obj = _unwrap_gt_root(obj, n_vectors=n_vectors)

    if not (isinstance(obj, dict) and "ground_truth_indices" in obj):
        keys = list(obj.keys())[:50] if isinstance(obj, dict) else [str(type(obj))]
        raise ValueError(f"GT unexpected format; missing 'ground_truth_indices'. keys={keys}")

    gt_I = np.asarray(obj["ground_truth_indices"], dtype=np.int64)
    if gt_I.ndim != 2:
        raise ValueError(f"gt_I must be 2D (nq, topK). got shape={gt_I.shape}")

    gt_D = None
    if isinstance(obj, dict):
        if "ground_truth_distances" in obj:
            gt_D = np.asarray(obj["ground_truth_distances"], dtype=np.float32)
        elif "ground_truth_D" in obj:
            gt_D = np.asarray(obj["ground_truth_D"], dtype=np.float32)

    if gt_D is not None:
        if gt_D.ndim != 2:
            raise ValueError(f"gt_D must be 2D (nq, topK). got shape={gt_D.shape}")
        if gt_D.shape != gt_I.shape:
            eprint(f"[warn] gt_D shape {gt_D.shape} != gt_I shape {gt_I.shape} (still returning both)")

    return gt_I, gt_D


def main():
    ap = argparse.ArgumentParser(description="Load only GT_I and GT_D from ground truth JSON.")
    ap.add_argument("--gt", required=True, help="GT .json (contains ground_truth_indices, optionally distances)")
    ap.add_argument(
        "--n_vectors",
        type=int,
        default=None,
        help="Optional: database vector count (used to unwrap nested dict by key '<n_vectors>')",
    )
    ap.add_argument("--k", type=int, default=10, help="Optional: slice first k columns (k>0).")
    ap.add_argument(
        "--save_prefix",
        type=str,
        default="",
        help="If set, save to <prefix>_I.npy and <prefix>_D.npy (if gt_D exists).",
    )

    # NEW: manual gt_D transform
    ap.add_argument(
        "--D_transform",
        type=str,
        default="none",
        choices=["none", "one_minus", "cos_sim_to_l2sq", "cos_dist_to_l2sq"],
        help=(
            "Manual transform applied to gt_D.\n"
            "  none: keep as-is (default)\n"
            "  one_minus: y <- 1-y (cosine similarity -> cosine distance)\n"
            "  cos_sim_to_l2sq: L2^2 = 2 - 2*cos_sim (requires normalized vectors)\n"
            "  cos_dist_to_l2sq: L2^2 = 2*cos_dist (requires normalized vectors)\n"
        ),
    )
    ap.add_argument(
        "--clip_cosine",
        action="store_true",
        help="If set, clip cosine similarity into [-1,1] before conversion (recommended).",
    )

    args = ap.parse_args()

    gt_I, gt_D = load_gt_I_D(args.gt, n_vectors=args.n_vectors)

    # slice k if needed
    k = int(args.k)
    if k > 0 and gt_I.shape[1] >= k:
        gt_I_k = gt_I[:, :k]
        if gt_D is not None and gt_D.shape[1] >= k:
            gt_D_k = gt_D[:, :k]
        else:
            gt_D_k = gt_D
    else:
        gt_I_k = gt_I
        gt_D_k = gt_D

    # apply transform if requested
    if gt_D_k is not None:
        before_min, before_max = float(np.min(gt_D_k)), float(np.max(gt_D_k))
        gt_D_k = _apply_D_transform(gt_D_k, transform=args.D_transform, clip_cosine=bool(args.clip_cosine))
        after_min, after_max = float(np.min(gt_D_k)), float(np.max(gt_D_k))
        eprint(f"[gt_D] transform={args.D_transform} clip_cosine={bool(args.clip_cosine)}")
        eprint(f"[gt_D] range before: min={before_min:.6g}, max={before_max:.6g}")
        eprint(f"[gt_D] range after : min={after_min:.6g}, max={after_max:.6g}")

    eprint(f"[gt_I] shape={gt_I_k.shape} dtype={gt_I_k.dtype}")
    eprint(f"[gt_D] shape={None if gt_D_k is None else gt_D_k.shape} dtype={None if gt_D_k is None else gt_D_k.dtype}")

    # print a small sanity check
    eprint(f"[sample] I[0,:min(5,k)] = {gt_I_k[0, :min(5, gt_I_k.shape[1])].tolist()}")
    if gt_D_k is not None:
        eprint(f"[sample] D[0,:min(5,k)] = {gt_D_k[0, :min(5, gt_D_k.shape[1])].tolist()}")

    if args.save_prefix:
        outI = args.save_prefix + "_I.npy"
        np.save(outI, gt_I_k, allow_pickle=False)
        eprint(f"[save] {outI}")

        if gt_D_k is not None:
            outD = args.save_prefix + "_D.npy"
            np.save(outD, gt_D_k, allow_pickle=False)
            eprint(f"[save] {outD}")

    print("OK", flush=True)


if __name__ == "__main__":
    main()
