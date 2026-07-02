#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_dataset_builder.py

Build two NPZ datasets for your masked-margin upper-bound pipeline.

============================================================
1) Train NPZ (sample training set from database vectors)
============================================================
- Sample similar pairs (i, j) from base vectors using metric: l2 / l2sq / cosine.
- For each pair, compute TWO masked L1 margin samples (Scheme A):
    x_q: query-side (i-side)
    x_c: candidate-side (j-side)
  Then build:
    x2 = concat(x_q, x_c)  -> 2P points
    y2 = concat(y,   y)    -> 2P points

This matches your requirement:
  "training set needs double-x"

============================================================
2) Eval NPZ (from ground-truth pairs)
============================================================
- Use GT pairs (pair_i, pair_j, y).
- Compute ONLY query-side x_q_only (i-side).
  (Still uses pair_j to build XOR mask, but does NOT store x_c.)

This matches your requirement:
  "gt does not take j-side"

============================================================
NEW (GT plotting preparation exports)
============================================================
- QC-Hamming distribution for GT pairs
- GT flip-rank distribution (query margin rank of flipped bits)
- Query margin distribution (sorted margins from small->large bit)

============================================================
GT sources supported:
============================================================
A) benchmark.datasets (BigANN / Deep1B): ds.get_groundtruth(k)
B) user-provided numpy GT files (.npy or .npz)

============================================================
IMPORTANT FIX (your request):
============================================================
Some GT files may store cosine *similarity* (sim), not distance.
We handle this by transforming GT y automatically:

  cosine_distance = 1 - cosine_similarity

This is controlled by:
  GroundTruthLoadConfig.gt_y_transform = "identity" | "one_minus" | "auto"

Default = "auto" (recommended).

============================================================
No argparse/CLI is used inside this module.
All functions are import-friendly.

Author: ChatGPT
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Literal

import numpy as np


# ============================================================
# Small helpers
# ============================================================

def _make_rng(seed: Optional[int]) -> np.random.Generator:
    """
    Create a numpy RNG.

    seed:
      - int   -> deterministic RNG
      - None  -> nondeterministic RNG (OS entropy)
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def _quantile_stats(x: np.ndarray) -> Dict[str, float]:
    """
    Simple percentile report for float-like arrays.

    Returns:
      dict containing:
        n, min, max, p0, p1, p5, ... p100
    """
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return {"n": 0}

    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(x.astype(np.float64), qs)

    out = {"n": int(x.size), "min": float(x.min()), "max": float(x.max())}
    for q, v in zip(qs, vals):
        out[f"p{int(q)}"] = float(v)
    return out


def _transform_gt_y(
    y: np.ndarray,
    mode: Literal["identity", "one_minus", "auto"],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Transform GT y-values if they are similarity (cosine sim), into distance.

    mode:
      - "identity" : keep y as-is
      - "one_minus": force y <- (1 - y)
      - "auto"     : auto detect by range:
            if y lies mostly in [-1,1] or [0,1] -> treat as cosine similarity -> 1-y

    Returns:
      (y_out, meta_info)
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    meta: Dict[str, Any] = {
        "gt_y_transform_mode": mode,
        "gt_y_transform_applied": False,
        "reason": None,
    }

    if y.size == 0:
        meta["reason"] = "empty"
        return y, meta

    if mode == "identity":
        meta["reason"] = "identity"
        return y, meta

    if mode == "one_minus":
        meta["gt_y_transform_applied"] = True
        meta["reason"] = "one_minus"
        return (1.0 - y).astype(np.float32, copy=False), meta

    # auto mode
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    meta["y_min"] = y_min
    meta["y_max"] = y_max

    # Heuristic:
    # - cosine similarity commonly in [-1,1] (or [0,1])
    # - cosine distance in [0,2]
    # - L2/L2sq can be larger
    #
    # We apply 1-y if it looks like similarity.
    if (y_min >= -1.0001) and (y_max <= 1.0001):
        meta["gt_y_transform_applied"] = True
        meta["reason"] = "auto_detect_similarity([-1,1])"
        return (1.0 - y).astype(np.float32, copy=False), meta

    if (y_min >= -1e-4) and (y_max <= 1.0001):
        meta["gt_y_transform_applied"] = True
        meta["reason"] = "auto_detect_similarity([0,1])"
        return (1.0 - y).astype(np.float32, copy=False), meta

    # Otherwise treat as distance already
    meta["reason"] = "auto_detect_distance"
    return y, meta


# ============================================================
# Threshold loader (your config format)
# ============================================================

def load_threshold_from_config(config_json_path: str) -> np.ndarray:
    """
    Load threshold vector (margin_position) from your AutoHash JSON config.

    Expected config format:
    {
      "build_index": {
        "hidden_dim": 22,
        "margin_position": [... length = hidden_dim ...]
      }
    }

    Returns:
      thr: float32 array [B]
    """
    with open(config_json_path, "r") as f:
        cfg = json.load(f)

    if "build_index" not in cfg:
        raise KeyError("Config missing top-level key: 'build_index'")

    bi = cfg["build_index"]
    if "margin_position" not in bi:
        raise KeyError("Config missing key: build_index['margin_position']")

    thr = np.asarray(bi["margin_position"], dtype=np.float32)
    if thr.ndim != 1:
        raise ValueError(f"margin_position must be 1D, got shape={thr.shape}")

    if "hidden_dim" in bi:
        B = int(bi["hidden_dim"])
        if thr.size != B:
            raise ValueError(
                f"threshold length mismatch: len(margin_position)={thr.size} != hidden_dim={B}"
            )
    return thr


# ============================================================
# Pairwise distance metrics (within a batch)
# ============================================================

def _pairwise_l2_or_l2sq(X: np.ndarray, *, squared: bool) -> np.ndarray:
    """
    Pairwise L2 / L2^2 inside a batch.

    X: [bs, d] float32

    Returns:
      D: [bs, bs] float32
    """
    X = np.asarray(X, dtype=np.float32)
    xx = (X * X).sum(axis=1, keepdims=True)  # [bs,1]
    D2 = xx + xx.T - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0).astype(np.float32, copy=False)
    if squared:
        return D2
    return np.sqrt(D2).astype(np.float32, copy=False)


def _pairwise_cosine_distance(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Pairwise cosine distance inside a batch:
      dist = 1 - cosine_similarity
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / (norms + eps)
    sim = (Xn @ Xn.T).astype(np.float32, copy=False)
    dist = (1.0 - sim).astype(np.float32, copy=False)
    dist = np.maximum(dist, 0.0)
    return dist


# ============================================================
# Sample similar pairs from database vectors
# ============================================================

@dataclass
class SimilarPairSamplingConfig:
    """
    Controls pair sampling from base vectors.

    Two ways to filter "similar" pairs:

    A) keep_frac (recommended, scale-free):
       Keep the smallest keep_frac distances per batch.
       Example: keep_frac = 0.005 -> keep smallest 0.5% per batch.

    B) y_keep_max (optional hard threshold):
       Keep only pairs with distance < y_keep_max.
       Metric-scale dependent.

    You can use both:
      - first apply keep_frac,
      - then apply y_keep_max.

    Notes:
      - Sampling is done per batch.
      - max_pairs_per_batch caps each batch output.

    seed:
      - int  -> deterministic
      - None -> random each run
    """
    batch_size: int = 4096
    num_batches: int = 32
    metric: Literal["l2", "l2sq", "cosine"] = "l2"

    keep_frac: Optional[float] = 0.005
    y_keep_max: Optional[float] = None

    max_pairs_per_batch: Optional[int] = 1_000_000
    seed: Optional[int] = 123
    verbose: bool = True


def sample_similar_pairs_from_base_vectors(
    vecs: np.ndarray,
    cfg: SimilarPairSamplingConfig,
) -> Dict[str, Any]:
    """
    Randomly sample batches from base vectors, compute pairwise distances,
    and keep similar pairs.

    Returns:
      {
        "pair_i": int64 [P],
        "pair_j": int64 [P],
        "y":      float32 [P],
        "meta":   dict
      }
    """
    if cfg.keep_frac is None and cfg.y_keep_max is None:
        raise ValueError("At least one of keep_frac or y_keep_max must be provided.")

    metric_l = cfg.metric.lower().strip()
    if metric_l not in ("l2", "l2sq", "cosine"):
        raise ValueError(f"metric must be one of ['l2','l2sq','cosine'], got {cfg.metric}")

    keep_frac = None
    if cfg.keep_frac is not None:
        keep_frac = float(cfg.keep_frac)
        if not (0.0 < keep_frac < 1.0):
            raise ValueError(f"keep_frac must be in (0,1), got {keep_frac}")

    rng = _make_rng(cfg.seed)

    vecs = np.asarray(vecs)
    if vecs.ndim != 2:
        raise ValueError(f"base vectors must be 2D [N,d], got shape={vecs.shape}")
    N, d = int(vecs.shape[0]), int(vecs.shape[1])

    all_i, all_j, all_y = [], [], []

    for b in range(int(cfg.num_batches)):
        batch_ids = rng.choice(N, size=int(cfg.batch_size), replace=False)
        X = vecs[batch_ids]
        Xf = X.astype(np.float32, copy=False) if X.dtype != np.float32 else X

        # IMPORTANT:
        # cosine metric here means cosine distance (1 - similarity)
        if metric_l == "cosine":
            D = _pairwise_cosine_distance(Xf)
        else:
            D = _pairwise_l2_or_l2sq(Xf, squared=(metric_l == "l2sq"))

        bs = int(cfg.batch_size)
        iu, ju = np.triu_indices(bs, k=1)
        Du = D[iu, ju]  # [bs*(bs-1)/2]

        keep = np.ones(Du.shape[0], dtype=np.bool_)

        q_thr = None
        if keep_frac is not None:
            q_thr = float(np.quantile(Du, keep_frac))
            keep &= (Du <= q_thr)

        if cfg.y_keep_max is not None:
            keep &= (Du < float(cfg.y_keep_max))

        iu = iu[keep]
        ju = ju[keep]
        y = Du[keep].astype(np.float32, copy=False)

        # Cap pairs per batch (avoid huge outputs)
        if cfg.max_pairs_per_batch is not None and y.size > int(cfg.max_pairs_per_batch):
            sel = rng.choice(y.size, size=int(cfg.max_pairs_per_batch), replace=False)
            iu, ju, y = iu[sel], ju[sel], y[sel]

        pair_i = batch_ids[iu].astype(np.int64, copy=False)
        pair_j = batch_ids[ju].astype(np.int64, copy=False)

        all_i.append(pair_i)
        all_j.append(pair_j)
        all_y.append(y)

        if cfg.verbose:
            msg = f"[sample] batch={b+1}/{cfg.num_batches} kept_pairs={y.size:,}"
            if keep_frac is not None:
                msg += f" keep_frac={keep_frac} q_thr={q_thr:.6g}"
            if cfg.y_keep_max is not None:
                msg += f" y_keep_max={float(cfg.y_keep_max):.6g}"
            print(msg)

    pair_i_all = np.concatenate(all_i, axis=0) if all_i else np.empty(0, dtype=np.int64)
    pair_j_all = np.concatenate(all_j, axis=0) if all_j else np.empty(0, dtype=np.int64)
    y_all = np.concatenate(all_y, axis=0) if all_y else np.empty(0, dtype=np.float32)

    if cfg.verbose:
        print(f"[sample] total_pairs={y_all.size:,}")
        if y_all.size > 0:
            print(
                f"[sample] y stats: min={y_all.min():.6g} "
                f"p50={np.percentile(y_all,50):.6g} "
                f"p99={np.percentile(y_all,99):.6g} "
                f"max={y_all.max():.6g}"
            )

    meta = {
        "metric": metric_l,
        "keep_frac": keep_frac,
        "y_keep_max": (float(cfg.y_keep_max) if cfg.y_keep_max is not None else None),
        "batch_size": int(cfg.batch_size),
        "num_batches": int(cfg.num_batches),
        "max_pairs_per_batch": (int(cfg.max_pairs_per_batch) if cfg.max_pairs_per_batch is not None else None),
        "seed": (int(cfg.seed) if cfg.seed is not None else None),
        "N": int(N),
        "d": int(d),
    }

    return {"pair_i": pair_i_all, "pair_j": pair_j_all, "y": y_all, "meta": meta}


# ============================================================
# Masked L1 margin (Scheme A)
# ============================================================

def compute_masked_l1_margin_schemeA_two_sides(
    enc: np.ndarray,
    thr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Scheme A (verified logic) when i/j refer to rows inside the SAME enc matrix:

      addr = (enc > thr)
      mask = XOR(addr_i, addr_j)

      x_q = sum_{k: mask_k=1} |enc[i,k] - thr[k]|
      x_c = sum_{k: mask_k=1} |enc[j,k] - thr[k]|

      x2  = concat(x_q, x_c)  -> 2P

    Returns:
      {
        "x_q": float32 [P],
        "x_c": float32 [P],
        "x2":  float32 [2P]
      }
    """
    enc = np.asarray(enc)
    thr = np.asarray(thr, dtype=np.float32)

    if enc.ndim != 2:
        raise ValueError(f"enc must be 2D, got shape={enc.shape}")

    B = int(enc.shape[1])
    if thr.shape != (B,):
        raise ValueError(f"thr shape mismatch: got {thr.shape}, expected {(B,)}")

    pair_i = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    if pair_i.size != pair_j.size:
        raise ValueError("pair_i and pair_j must have the same length")

    P = int(pair_i.size)
    if P == 0:
        return {
            "x_q": np.empty(0, dtype=np.float32),
            "x_c": np.empty(0, dtype=np.float32),
            "x2": np.empty(0, dtype=np.float32),
        }

    A = enc[pair_i]   # [P,B]
    C = enc[pair_j]   # [P,B]

    mask = np.logical_xor(A > thr, C > thr)  # [P,B]

    x_q = (np.abs(A - thr) * mask).sum(axis=1).astype(np.float32, copy=False)
    x_c = (np.abs(C - thr) * mask).sum(axis=1).astype(np.float32, copy=False)

    x2 = np.concatenate([x_q, x_c], axis=0).astype(np.float32, copy=False)
    return {"x_q": x_q, "x_c": x_c, "x2": x2}


def compute_masked_l1_margin_schemeA_qside_only_cross(
    q_enc: np.ndarray,
    base_enc: np.ndarray,
    thr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """
    Correct GT/Eval version for retrieval datasets:

      pair_i indexes q_enc (query-side encoder output)
      pair_j indexes base_enc (database-side encoder output)

      mask = XOR( (q_enc[i] > thr), (base_enc[j] > thr) )
      x_q  = sum_{k:mask=1} |q_enc[i,k] - thr[k]|

    Output:
      x_q_only: float32 [P]
    """
    q_enc = np.asarray(q_enc)
    base_enc = np.asarray(base_enc)
    thr = np.asarray(thr, dtype=np.float32)

    if q_enc.ndim != 2 or base_enc.ndim != 2:
        raise ValueError(f"q_enc/base_enc must be 2D, got q={q_enc.shape}, base={base_enc.shape}")

    B = int(q_enc.shape[1])
    if int(base_enc.shape[1]) != B:
        raise ValueError(f"hidden_dim mismatch: q_enc B={q_enc.shape[1]} vs base_enc B={base_enc.shape[1]}")
    if thr.shape != (B,):
        raise ValueError(f"thr shape mismatch: got {thr.shape}, expected {(B,)}")

    pair_i = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    if pair_i.size != pair_j.size:
        raise ValueError("pair_i and pair_j must have the same length")

    P = int(pair_i.size)
    if P == 0:
        return np.empty(0, dtype=np.float32)

    A = q_enc[pair_i]        # [P,B]
    C = base_enc[pair_j]     # [P,B]

    mask = np.logical_xor(A > thr, C > thr)
    x_q = (np.abs(A - thr) * mask).sum(axis=1).astype(np.float32, copy=False)
    return x_q


# ============================================================
# GT plotting preparation: query margins / QC-hamming / flip-rank
# ============================================================

def _compute_query_margin_sorted_stats(
    q_enc: np.ndarray,
    thr: np.ndarray,
    *,
    qs=(10, 50, 90),
) -> Dict[str, np.ndarray]:
    """
    Query margin distribution from smallest->largest bit.

    For each query:
      margins = |q_enc - thr| -> [B]
      sorted_margins = sort(margins) -> [B] (ascending)

    Returns:
      {
        "mean": float32 [B],
        "p10":  float32 [B],
        "p50":  float32 [B],
        "p90":  float32 [B],
      }
    """
    q_enc = np.asarray(q_enc, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)

    if q_enc.ndim != 2:
        raise ValueError(f"q_enc must be 2D, got {q_enc.shape}")
    B = int(q_enc.shape[1])
    if thr.shape != (B,):
        raise ValueError(f"thr shape mismatch: {thr.shape} vs {(B,)}")

    margins = np.abs(q_enc - thr)                 # [nq,B]
    sorted_margins = np.sort(margins, axis=1)     # [nq,B]

    mean = sorted_margins.mean(axis=0).astype(np.float32)

    out = {"mean": mean}
    for q in qs:
        out[f"p{int(q)}"] = np.percentile(sorted_margins, q, axis=0).astype(np.float32)
    return out


def _sample_sorted_margins(
    q_enc: np.ndarray,
    thr: np.ndarray,
    *,
    max_queries: int = 2000,
    seed: Optional[int] = 123,
) -> np.ndarray:
    """
    Save a small matrix [Q_samp, B] for visualization.
    Each row: sorted margins (small->large).

    Note:
      - Query-side is sampled for visualization only.
      - The full statistics above is computed using all queries.
    """
    q_enc = np.asarray(q_enc, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)

    nq = int(q_enc.shape[0])
    B = int(q_enc.shape[1])
    if nq == 0:
        return np.empty((0, B), dtype=np.float32)

    rng = _make_rng(seed)
    Qs = min(int(max_queries), nq)
    sel = rng.choice(nq, size=Qs, replace=False)

    margins = np.abs(q_enc[sel] - thr)            # [Qs,B]
    return np.sort(margins, axis=1).astype(np.float32, copy=False)


def _compute_qc_hamming_flat_from_pairs(
    q_enc: np.ndarray,
    base_enc: np.ndarray,
    thr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """
    Compute Hamming distance for each GT pair (query i, candidate j).

    IMPORTANT:
      - This function does NOT build the full base-code matrix (safe for 1B base_enc),
        instead it gathers only P rows used by GT pairs.

    Returns:
      ham: uint16 [P]
    """
    q_enc = np.asarray(q_enc, dtype=np.float32)
    base_enc = np.asarray(base_enc, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)

    pair_i = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    if pair_i.size != pair_j.size:
        raise ValueError("pair_i/pair_j must have same length")

    A = q_enc[pair_i]        # [P,B]
    C = base_enc[pair_j]     # [P,B]
    ham = np.count_nonzero((A > thr) ^ (C > thr), axis=1).astype(np.uint16, copy=False)
    return ham


def _hist_counts_int(values: np.ndarray, *, n_bins: int) -> np.ndarray:
    """
    Histogram counts for integer values assumed in [0, n_bins-1].

    Returns:
      int64 [n_bins]
    """
    v = np.asarray(values).reshape(-1)
    if v.size == 0:
        return np.zeros(n_bins, dtype=np.int64)
    h = np.bincount(v.astype(np.int64), minlength=n_bins).astype(np.int64, copy=False)
    if h.size > n_bins:
        h = h[:n_bins]
    return h


def _compute_gt_flip_rank_distribution(
    q_enc: np.ndarray,
    base_enc: np.ndarray,
    thr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    sample_max: int = 2_000_000,
    seed: Optional[int] = 123,
) -> Dict[str, Any]:
    """
    Compute GT flip-rank distribution.

    Definition:
      - For each query q, compute margins |q_enc - thr| -> [B]
      - Rank bits by margin ASC (closest to threshold rank=0)
      - For a GT pair (q, c), flipped bits are XOR(code_q, code_c)
      - Each flipped bit contributes its rank (0..B-1)

    Key requirement from you:
      - Query-side is full (we compute rank_maps for ALL queries).
      - sample is only for saving a subset of ranks for plotting.

    Returns:
      {
        "counts": int64 [B],
        "probs":  float32 [B],
        "sample": uint16 [?],
        "avg_rank": float,
        "top4": float (percentage),
        "top8": float (percentage),
        "n_flips_total": int
      }
    """
    q_enc = np.asarray(q_enc, dtype=np.float32)
    base_enc = np.asarray(base_enc, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)

    pair_i = np.asarray(pair_i, dtype=np.int64).reshape(-1)
    pair_j = np.asarray(pair_j, dtype=np.int64).reshape(-1)
    if pair_i.size != pair_j.size:
        raise ValueError("pair_i/pair_j must have same length")

    nq, B = q_enc.shape
    P = int(pair_i.size)

    if P == 0:
        return {
            "counts": np.zeros(B, dtype=np.int64),
            "probs": np.zeros(B, dtype=np.float32),
            "sample": np.empty(0, dtype=np.uint16),
            "avg_rank": float("nan"),
            "top4": float("nan"),
            "top8": float("nan"),
            "n_flips_total": 0,
        }

    # ---- Query-side FULL rank maps ----
    # rank_maps[q, bit] = rank of this bit within query q margins (0=closest to threshold)
    margins = np.abs(q_enc - thr)                       # [nq,B]
    margin_sorted_idx = np.argsort(margins, axis=1)     # [nq,B]
    rank_maps = np.argsort(margin_sorted_idx, axis=1).astype(np.int16, copy=False)  # [nq,B]

    # ---- Pair-side codes computed only for pairs ----
    A = q_enc[pair_i]                    # [P,B]
    C = base_enc[pair_j]                 # [P,B]
    q_codes_pairs = (A > thr)            # [P,B]
    c_codes_pairs = (C > thr)            # [P,B]

    counts = np.zeros(B, dtype=np.int64)

    rng = _make_rng(seed)
    sample_cap = int(sample_max)
    sample_buf: list[int] = []

    # Streaming over pairs
    for idx in range(P):
        diffs = (q_codes_pairs[idx] ^ c_codes_pairs[idx])  # [B] bool
        if not np.any(diffs):
            continue

        flipped_bits = np.nonzero(diffs)[0]                 # indices of flipped bits
        ranks = rank_maps[pair_i[idx], flipped_bits]         # ranks in [0,B-1]

        # Update counts
        bc = np.bincount(ranks.astype(np.int64), minlength=B)
        counts += bc.astype(np.int64, copy=False)

        # Sample some ranks for plotting (randomized)
        if sample_cap > 0 and len(sample_buf) < sample_cap:
            take = min(sample_cap - len(sample_buf), int(ranks.size))
            if take > 0:
                if take < ranks.size:
                    sel = rng.choice(ranks.size, size=take, replace=False)
                    sample_buf.extend(ranks[sel].astype(np.uint16, copy=False).tolist())
                else:
                    sample_buf.extend(ranks.astype(np.uint16, copy=False).tolist())

    total_flips = int(counts.sum())
    probs = (counts / max(total_flips, 1)).astype(np.float32)

    if total_flips > 0:
        avg_rank = float((np.arange(B, dtype=np.float64) * counts.astype(np.float64)).sum() / total_flips)
        top4 = float(counts[:4].sum() / total_flips * 100.0)
        top8 = float(counts[:8].sum() / total_flips * 100.0)
    else:
        avg_rank = float("nan")
        top4 = float("nan")
        top8 = float("nan")

    sample_arr = np.asarray(sample_buf, dtype=np.uint16) if sample_buf else np.empty(0, dtype=np.uint16)

    return {
        "counts": counts,
        "probs": probs,
        "sample": sample_arr,
        "avg_rank": avg_rank,
        "top4": top4,
        "top8": top8,
        "n_flips_total": total_flips,
    }


# ============================================================
# GT loading utilities (benchmark / npy)
# ============================================================

@dataclass
class GroundTruthLoadConfig:
    """
    How to load ground-truth pairs for Eval NPZ.

    Supported modes:
      - mode="benchmark":
          load using benchmark.datasets + DATASETS[dataset_name]().get_groundtruth(k)
      - mode="npy":
          load from numpy files

    For mode="benchmark":
      - must provide basedir and dataset_name

    For mode="npy":
      - can provide either:
          (A) gt_I_path + gt_D_path (both .npy), or
          (B) gt_npz_path (contains arrays: gt_I, gt_D)

    gt_y_transform:
      - "identity" : keep y as-is (distance GT)
      - "one_minus": force y <- 1 - y (cosine sim -> distance)
      - "auto"     : auto detect and convert if y looks like cosine similarity
    """
    mode: Literal["benchmark", "npy"] = "benchmark"

    # benchmark.datasets mode
    basedir: Optional[str] = None
    dataset_name: Optional[str] = None
    k: int = 10

    # numpy mode
    gt_I_path: Optional[str] = None
    gt_D_path: Optional[str] = None
    gt_npz_path: Optional[str] = None

    # name of arrays in npz
    npz_key_I: str = "gt_I"
    npz_key_D: str = "gt_D"

    # y correction
    gt_y_transform: Literal["identity", "one_minus", "auto"] = "identity"


def _flatten_gt_to_pairs(gt_I: np.ndarray, gt_D: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert FAISS-style GT matrices [nq,k] to flat arrays.

    Returns:
      pair_i: int64 [P]  (query ids)
      pair_j: int64 [P]  (base ids)
      y:      float32[P] (distance or similarity depending on GT)
    """
    gt_I = np.asarray(gt_I)
    gt_D = np.asarray(gt_D)

    if gt_I.ndim != 2 or gt_D.ndim != 2:
        raise ValueError(f"gt_I/gt_D must be 2D, got I={gt_I.shape}, D={gt_D.shape}")
    if gt_I.shape != gt_D.shape:
        raise ValueError(f"gt_I/gt_D shape mismatch, got I={gt_I.shape}, D={gt_D.shape}")

    nq, k = gt_I.shape
    pair_i = np.repeat(np.arange(nq, dtype=np.int64), k)
    pair_j = gt_I.reshape(-1).astype(np.int64, copy=False)
    y = gt_D.reshape(-1).astype(np.float32, copy=False)
    return pair_i, pair_j, y


def load_groundtruth_pairs(cfg: GroundTruthLoadConfig) -> Dict[str, Any]:
    """
    Load GT pairs as flat arrays (pair_i, pair_j, y).

    IMPORTANT FIX:
      Some GT might provide cosine similarity; we support:
        cosine_distance = 1 - cosine_similarity

      Controlled by cfg.gt_y_transform.
    """
    if cfg.mode == "benchmark":
        if cfg.basedir is None or cfg.dataset_name is None:
            raise ValueError("benchmark mode requires basedir and dataset_name")

        # Import inside the function for import-friendliness
        sys.path.append('/path/to/big-ann-benchmarks/big-ann-benchmarks')
        from benchmark import datasets as _datasets  # type: ignore

        _datasets.BASEDIR = str(cfg.basedir)

        DATASETS = getattr(_datasets, "DATASETS", None)
        if DATASETS is None:
            raise RuntimeError(
                "Cannot find DATASETS in benchmark.datasets. "
                "Please ensure your environment provides DATASETS[dataset_name]()."
            )

        ds = DATASETS[str(cfg.dataset_name)]()
        gt_I, gt_D = ds.get_groundtruth(k=int(cfg.k))

        pair_i, pair_j, y_raw = _flatten_gt_to_pairs(gt_I, gt_D)

        # Apply y transform if needed (cosine similarity -> distance)
        y, y_meta = _transform_gt_y(y_raw, mode=cfg.gt_y_transform)

        meta = {
            "source": "benchmark.datasets",
            "dataset_name": str(cfg.dataset_name),
            "basedir": str(cfg.basedir),
            "k": int(cfg.k),
            "gt_shape": [int(gt_I.shape[0]), int(gt_I.shape[1])],
            "gt_y_transform": y_meta,
        }
        return {"pair_i": pair_i, "pair_j": pair_j, "y": y, "meta": meta}

    if cfg.mode == "npy":
        if cfg.gt_npz_path is not None:
            z = np.load(cfg.gt_npz_path, allow_pickle=False)
            if cfg.npz_key_I not in z or cfg.npz_key_D not in z:
                raise KeyError(
                    f"npz missing keys: need '{cfg.npz_key_I}' and '{cfg.npz_key_D}', "
                    f"got keys={list(z.keys())}"
                )
            gt_I = z[cfg.npz_key_I]
            gt_D = z[cfg.npz_key_D]
        else:
            if cfg.gt_I_path is None or cfg.gt_D_path is None:
                raise ValueError("npy mode requires either gt_npz_path OR (gt_I_path + gt_D_path)")
            gt_I = np.load(cfg.gt_I_path, allow_pickle=False)
            gt_D = np.load(cfg.gt_D_path, allow_pickle=False)

        pair_i, pair_j, y_raw = _flatten_gt_to_pairs(gt_I, gt_D)

        y, y_meta = _transform_gt_y(y_raw, mode=cfg.gt_y_transform)

        meta = {
            "source": "npy_or_npz",
            "gt_I_path": cfg.gt_I_path,
            "gt_D_path": cfg.gt_D_path,
            "gt_npz_path": cfg.gt_npz_path,
            "k": int(gt_I.shape[1]),
            "gt_shape": [int(gt_I.shape[0]), int(gt_I.shape[1])],
            "gt_y_transform": y_meta,
        }
        return {"pair_i": pair_i, "pair_j": pair_j, "y": y, "meta": meta}

    raise ValueError(f"Unsupported GT mode: {cfg.mode}")


# ============================================================
# NPZ builders
# ============================================================

def save_npz(path: str, **arrays) -> None:
    """
    Save arrays to NPZ (creates parent directory if missing).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **arrays)
    print(f"[save] {path}")


def build_train_npz_from_database_sampling(
    *,
    base_vectors: np.ndarray,
    enc: np.ndarray,
    thr: np.ndarray,
    sampling_cfg: SimilarPairSamplingConfig,
    out_npz_path: str,
) -> Dict[str, Any]:
    """
    Build the TRAIN NPZ:

      1) sample similar pairs from base_vectors -> (pair_i, pair_j, y)
      2) compute masked margins -> x_q/x_c/x2
      3) y2 = concat(y,y)
      4) save NPZ

    Saved fields:
      - pair_i, pair_j, y
      - x_q, x_c
      - x2, y2
      - thr
      - meta (json string)
    """
    sampled = sample_similar_pairs_from_base_vectors(base_vectors, sampling_cfg)
    pair_i = sampled["pair_i"]
    pair_j = sampled["pair_j"]
    y = sampled["y"]
    meta = sampled["meta"]

    margins = compute_masked_l1_margin_schemeA_two_sides(enc, thr, pair_i, pair_j)
    x_q = margins["x_q"]
    x_c = margins["x_c"]
    x2 = margins["x2"]

    y2 = np.concatenate([y, y], axis=0).astype(np.float32, copy=False)

    save_npz(
        out_npz_path,
        pair_i=pair_i,
        pair_j=pair_j,
        y=y,
        x_q=x_q,
        x_c=x_c,
        x2=x2,
        y2=y2,
        thr=np.asarray(thr, dtype=np.float32),
        meta=json.dumps(meta),
    )

    return {"out_npz": out_npz_path, "P": int(y.size), "P2": int(y2.size), "meta": meta}


def build_eval_npz_from_gt_pairs(
    *,
    q_enc: np.ndarray,
    base_enc: np.ndarray,
    thr: np.ndarray,
    gt_pair_i: np.ndarray,
    gt_pair_j: np.ndarray,
    gt_y: np.ndarray,
    out_npz_path: str,
    meta: Optional[Dict[str, Any]] = None,
    drop_invalid_base_ids: bool = True,
    # plotting controls
    margin_sample_max_queries: int = 2000,
    fliprank_sample_max: int = 2_000_000,
    seed: Optional[int] = 123,
) -> Dict[str, Any]:
    """
    Build the EVAL NPZ from GT pairs.

    IMPORTANT:
      - compute ONLY x_q_only (query side),
      - do NOT store candidate-side x_c.

    Saved fields:
      - pair_i, pair_j, y
      - x_q_only
      - thr
      - meta (json string)

    Also exports for plotting:
      - qc_hamming / qc_hamming_hist
      - gt_flip_rank_counts / probs / sample
      - q_margin_sorted_mean/p10/p50/p90/sample
    """
    q_enc = np.asarray(q_enc)
    base_enc = np.asarray(base_enc)
    thr = np.asarray(thr, dtype=np.float32)

    gt_pair_i = np.asarray(gt_pair_i, dtype=np.int64).reshape(-1)
    gt_pair_j = np.asarray(gt_pair_j, dtype=np.int64).reshape(-1)
    gt_y = np.asarray(gt_y, dtype=np.float32).reshape(-1)

    if not (gt_pair_i.size == gt_pair_j.size == gt_y.size):
        raise ValueError("gt_pair_i, gt_pair_j, gt_y must have the same length")

    if drop_invalid_base_ids:
        N = int(base_enc.shape[0])
        valid = (gt_pair_j >= 0) & (gt_pair_j < N)
        if not np.all(valid):
            gt_pair_i = gt_pair_i[valid]
            gt_pair_j = gt_pair_j[valid]
            gt_y = gt_y[valid]

    # 1) masked margin x_q_only
    x_q_only = compute_masked_l1_margin_schemeA_qside_only_cross(
        q_enc=q_enc,
        base_enc=base_enc,
        thr=thr,
        pair_i=gt_pair_i,
        pair_j=gt_pair_j,
    )

    meta2 = {} if meta is None else dict(meta)
    meta2.update({
        "note": "GT eval dataset: only query-side x_q_only is stored (no j-side x_c).",
        "P": int(gt_y.size),
        "nq": int(q_enc.shape[0]),
        "N_base": int(base_enc.shape[0]),
        "B": int(q_enc.shape[1]),
        "seed": (int(seed) if seed is not None else None),
    })

    # ------------------------------------------------------------
    # Prepare plotting/statistics exports
    # ------------------------------------------------------------

    # (A) Query margin distribution (full stats on all queries)
    margin_sorted_stats = _compute_query_margin_sorted_stats(q_enc, thr, qs=(10, 50, 90))
    margin_sorted_sample = _sample_sorted_margins(
        q_enc, thr,
        max_queries=int(margin_sample_max_queries),
        seed=seed
    )

    # (B) QC-Hamming distribution for GT pairs
    qc_ham = _compute_qc_hamming_flat_from_pairs(q_enc, base_enc, thr, gt_pair_i, gt_pair_j)  # uint16 [P]
    B = int(q_enc.shape[1])
    qc_ham_hist = _hist_counts_int(qc_ham.astype(np.int64), n_bins=B + 1).astype(np.int64)

    # (C) GT flip-rank distribution (query-side FULL, sample only for saving ranks)
    flip = _compute_gt_flip_rank_distribution(
        q_enc=q_enc,
        base_enc=base_enc,
        thr=thr,
        pair_i=gt_pair_i,
        pair_j=gt_pair_j,
        sample_max=int(fliprank_sample_max),
        seed=seed,
    )

    # Add quick summaries to meta (human-readable)
    meta2.update({
        "qc_hamming_stats": _quantile_stats(qc_ham.astype(np.float32)),
        "qc_hamming_total_pairs": int(qc_ham.size),
        "gt_flip_rank": {
            "avg_rank": float(flip["avg_rank"]),
            "top4_coverage_pct": float(flip["top4"]),
            "top8_coverage_pct": float(flip["top8"]),
            "n_flips_total": int(flip["n_flips_total"]),
        },
        "query_margin_sorted_stats_quick": {
            "mean_min": float(margin_sorted_stats["mean"][0]),
            "mean_max": float(margin_sorted_stats["mean"][-1]),
        },
    })

    # Save NPZ
    save_npz(
        out_npz_path,
        pair_i=gt_pair_i,
        pair_j=gt_pair_j,
        y=gt_y,
        x_q_only=x_q_only,
        thr=np.asarray(thr, dtype=np.float32),

        # plotting exports
        qc_hamming=qc_ham,                        # uint16 [P]
        qc_hamming_hist=qc_ham_hist,              # int64  [B+1]

        gt_flip_rank_counts=flip["counts"],       # int64  [B]
        gt_flip_rank_probs=flip["probs"],         # float32[B]
        gt_flip_rank_sample=flip["sample"],       # uint16 [?] sampled ranks

        q_margin_sorted_mean=margin_sorted_stats["mean"],   # float32 [B]
        q_margin_sorted_p10=margin_sorted_stats["p10"],     # float32 [B]
        q_margin_sorted_p50=margin_sorted_stats["p50"],     # float32 [B]
        q_margin_sorted_p90=margin_sorted_stats["p90"],     # float32 [B]
        q_margin_sorted_sample=margin_sorted_sample,        # float32 [Qs,B]

        meta=json.dumps(meta2),
    )

    return {"out_npz": out_npz_path, "P": int(gt_y.size), "meta": meta2}


def build_eval_npz_from_groundtruth_source(
    *,
    gt_cfg: GroundTruthLoadConfig,
    q_enc: np.ndarray,
    base_enc: np.ndarray,
    thr: np.ndarray,
    out_npz_path: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = 123,
    # plotting controls
    margin_sample_max_queries: int = 2000,
    fliprank_sample_max: int = 2_000_000,
) -> Dict[str, Any]:
    """
    High-level helper:
      1) load GT pairs from benchmark OR numpy files
      2) build eval npz using q_enc + base_enc

    IMPORTANT:
      - gt_y is already corrected (auto 1-y if cosine similarity) in load_groundtruth_pairs()
    """
    gt = load_groundtruth_pairs(gt_cfg)

    meta = dict(gt["meta"])
    if extra_meta is not None:
        meta.update(extra_meta)

    return build_eval_npz_from_gt_pairs(
        q_enc=q_enc,
        base_enc=base_enc,
        thr=thr,
        gt_pair_i=gt["pair_i"],
        gt_pair_j=gt["pair_j"],
        gt_y=gt["y"],
        out_npz_path=out_npz_path,
        meta=meta,
        drop_invalid_base_ids=True,
        margin_sample_max_queries=int(margin_sample_max_queries),
        fliprank_sample_max=int(fliprank_sample_max),
        seed=seed,
    )
