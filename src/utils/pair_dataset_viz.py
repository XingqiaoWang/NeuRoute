#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pair_dataset_viz.py

Visualization + Early-stop upper bound export/eval for your masked-margin pipeline.

This module supports TWO major features:

(A) Scatter visualization for datasets
-------------------------------------
Training NPZ (from build_train_npz_from_database_sampling):
  - y      : float32 [P]     (depends on sampling metric: can be L2 or L2^2)
  - x_q    : float32 [P]
  - x_c    : float32 [P]
  - x2     : float32 [2P]
  - y2     : float32 [2P]    (= concat(y,y), NOT automatically squared!)
  - meta   : json string, contains "metric" (l2 / l2sq / cosine), etc.

Eval NPZ (from build_eval_npz_from_gt_pairs):
  - y        : float32 [P]
      IMPORTANT: for bigann/deep1b FAISS groundtruth, this y is typically L2^2 (squared L2)
  - x_q_only : float32 [P]

It generates and saves:
  - train_scatter_xq_vs_y.png
  - train_scatter_xc_vs_y.png
  - train_scatter_x2_vs_y2.png
  - eval_scatter_xqonly_vs_y.png

(B) Early-stop upper bound function in y_space = L2^2
-----------------------------------------------------
We build an upper bound function x <= ub(y_space), where:

  y_space means "squared L2 distance" (L2^2), as the canonical space.

We support a transform from NPZ raw y to y_space:

  y_transform="identity" : y_space = y_raw
      Use this when y_raw is already L2^2
      - bigann/deep1b gt_D is typically L2^2
      - sampling metric="l2sq" also produces L2^2

  y_transform="square"   : y_space = (y_raw)^2
      Use this when y_raw is L2 (sqrt), but you want L2^2 space
      - sampling metric="l2"

Upper bound models:
  (B1) mode="patch" (original):
      1) Origin bound: x <= a0 * y_space
         where a0 = quantile_q( x / y_space ) on y_space>eps
      2) Flat patch near origin:
         y0 = min(y_space in training data)
         e  = a0*y0
         ub(y) = e          if y <= y0
                 a0*y       otherwise

  (B2) mode="shift_b" (NEW, requested):
      ub(y) = a0 * y_space + b
      - Fit a0 same as origin bound (ratio quantile)
      - Sweep b by targeting different violation rates on training set

Exports for patch:
  - earlystop_params.json
  - earlystop_params.npz
  - earlystop_train_scatter_yspace.png
  - earlystop_eval_scatter_yspace.png
  - earlystop_eval_report.json

Exports for shift_b sweep:
  - earlystop_shiftb_tXXXX_params.json
  - earlystop_shiftb_tXXXX_train_scatter_yspace.png
  - earlystop_shiftb_tXXXX_eval_scatter_yspace.png
  - earlystop_shiftb_tXXXX_eval_report.json

Important policy:
  - Training can use x2,y2_raw (2P) for better density.
  - Eval MUST use x_q_only,y_raw (q-side only). (GT does not take j-side)

Author: ChatGPT
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Basic helpers
# ============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _maybe_parse_meta(meta_obj: Any) -> Optional[Dict[str, Any]]:
    """
    NPZ meta is usually json.dumps(meta), stored as a numpy scalar string or bytes.
    """
    if meta_obj is None:
        return None
    try:
        if isinstance(meta_obj, np.ndarray) and meta_obj.shape == ():
            meta_obj = meta_obj.item()
        if isinstance(meta_obj, (bytes, bytearray)):
            meta_obj = meta_obj.decode("utf-8", errors="ignore")
        if isinstance(meta_obj, str):
            return json.loads(meta_obj)
    except Exception:
        return None
    return None


def _subsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly subsample to at most max_points for scatter plotting.
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x/y size mismatch: x={x.size}, y={y.size}")

    n = x.size
    if n <= max_points:
        return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_points), replace=False)
    return x[idx].astype(np.float32, copy=False), y[idx].astype(np.float32, copy=False)


def _plot_scatter_save(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    alpha: float = 0.15,
    s: float = 2.0,
    max_points: int = 200_000,
    seed: int = 123,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """
    Scatter plot with subsampling + save. Returns quick stats.
    """
    xs, ys = _subsample_xy(x, y, max_points=max_points, seed=seed)

    stats = {
        "n_total": int(np.asarray(x).size),
        "n_plot": int(xs.size),
        "x_min": float(np.min(xs)) if xs.size else None,
        "x_p50": float(np.percentile(xs, 50)) if xs.size else None,
        "x_p99": float(np.percentile(xs, 99)) if xs.size else None,
        "x_max": float(np.max(xs)) if xs.size else None,
        "y_min": float(np.min(ys)) if ys.size else None,
        "y_p50": float(np.percentile(ys, 50)) if ys.size else None,
        "y_p99": float(np.percentile(ys, 99)) if ys.size else None,
        "y_max": float(np.max(ys)) if ys.size else None,
    }

    plt.figure(figsize=(9, 6))
    plt.scatter(xs, ys, s=s, alpha=alpha)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=200)
    plt.close()
    return stats


# ============================================================
# Part A: Scatter visualization
# ============================================================

def visualize_training_npz(
    train_npz: str,
    *,
    out_dir: str,
    prefix: str = "train",
    max_points: int = 200_000,
    seed: int = 123,
    alpha: float = 0.12,
    s: float = 2.0,
) -> Dict[str, Any]:
    """
    Visualize training dataset NPZ.

    Expected keys:
      - y, x_q, x_c, x2, y2
    """
    _ensure_dir(out_dir)
    data = _load_npz(train_npz)

    required = ["y", "x_q", "x_c", "x2", "y2"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Training NPZ missing keys: {missing}. Found keys={list(data.keys())}")

    y = data["y"]
    x_q = data["x_q"]
    x_c = data["x_c"]
    x2 = data["x2"]
    y2 = data["y2"]
    meta = _maybe_parse_meta(data.get("meta", None))

    metric_str = None
    if meta is not None:
        metric_str = meta.get("metric", None)

    out1 = os.path.join(out_dir, f"{prefix}_scatter_xq_vs_y.png")
    st1 = _plot_scatter_save(
        x_q, y,
        title=f"{prefix}: x_q vs y (P points)  metric={metric_str}",
        xlabel="x_q (masked L1 margin, q-side)",
        ylabel="y (pair distance, raw)",
        out_path=out1,
        alpha=alpha, s=s, max_points=max_points, seed=seed,
    )

    out2 = os.path.join(out_dir, f"{prefix}_scatter_xc_vs_y.png")
    st2 = _plot_scatter_save(
        x_c, y,
        title=f"{prefix}: x_c vs y (P points)  metric={metric_str}",
        xlabel="x_c (masked L1 margin, c-side)",
        ylabel="y (pair distance, raw)",
        out_path=out2,
        alpha=alpha, s=s, max_points=max_points, seed=seed + 1,
    )

    out3 = os.path.join(out_dir, f"{prefix}_scatter_x2_vs_y2.png")
    st3 = _plot_scatter_save(
        x2, y2,
        title=f"{prefix}: x2 vs y2 (2P points, double-x)  metric={metric_str}",
        xlabel="x2 = concat(x_q, x_c)",
        ylabel="y2_raw = concat(y, y)  (NOT auto-squared)",
        out_path=out3,
        alpha=alpha, s=s, max_points=max_points, seed=seed + 2,
    )

    return {
        "train_npz": train_npz,
        "out_dir": out_dir,
        "meta": meta,
        "plots": {
            "xq_vs_y": {"path": out1, "stats": st1},
            "xc_vs_y": {"path": out2, "stats": st2},
            "x2_vs_y2": {"path": out3, "stats": st3},
        },
    }


def visualize_eval_npz(
    eval_npz: str,
    *,
    out_dir: str,
    prefix: str = "eval",
    max_points: int = 200_000,
    seed: int = 123,
    alpha: float = 0.12,
    s: float = 2.0,
) -> Dict[str, Any]:
    """
    Visualize evaluation dataset NPZ (GT).

    Expected keys:
      - y, x_q_only

    IMPORTANT:
      - If eval NPZ was built from FAISS groundtruth (bigann/deep1b),
        y is typically L2^2 (squared L2 distance).
    """
    _ensure_dir(out_dir)
    data = _load_npz(eval_npz)

    required = ["y", "x_q_only"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Eval NPZ missing keys: {missing}. Found keys={list(data.keys())}")

    y = data["y"]
    x_q_only = data["x_q_only"]
    meta = _maybe_parse_meta(data.get("meta", None))

    out1 = os.path.join(out_dir, f"{prefix}_scatter_xqonly_vs_y.png")
    st1 = _plot_scatter_save(
        x_q_only, y,
        title=f"{prefix}: x_q_only vs y (GT, q-side only)",
        xlabel="x_q_only (masked L1 margin, query side only)",
        ylabel="y (GT distance, usually L2^2 for bigann/deep1b)",
        out_path=out1,
        alpha=alpha, s=s, max_points=max_points, seed=seed,
    )

    return {
        "eval_npz": eval_npz,
        "out_dir": out_dir,
        "meta": meta,
        "plots": {"xqonly_vs_y": {"path": out1, "stats": st1}},
    }


# ============================================================
# Part B: Early-stop upper bound in y_space = L2^2 (patch + shift_b sweep)
# ============================================================

def _percentile_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return {"n": 0}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(x, qs)
    out = {"n": int(x.size), "min": float(x.min()), "max": float(x.max())}
    for q, v in zip(qs, vals):
        out[f"p{int(q)}"] = float(v)
    return out


def _to_yspace_l2sq(y_raw: np.ndarray, y_transform: str) -> np.ndarray:
    """
    Convert raw y to y_space = L2^2.

    y_transform:
      - "identity": y_space = y_raw
      - "square":   y_space = y_raw^2
    """
    y_raw = np.asarray(y_raw, dtype=np.float32).reshape(-1)
    t = str(y_transform).lower().strip()
    if t == "identity":
        return y_raw.astype(np.float32, copy=False)
    if t == "square":
        return (y_raw.astype(np.float64) ** 2).astype(np.float32)
    raise ValueError(f"Invalid y_transform='{y_transform}', must be 'identity' or 'square'.")


def fit_origin_upper_bound_yspace(
    y_space: np.ndarray,
    x: np.ndarray,
    *,
    q: float = 0.995,
    eps: float = 1e-12,
) -> float:
    """
    Fit x <= a0*y_space using ratio quantile:
      a0 = quantile_q( x / y_space ) for y_space > eps
    """
    y_space = np.asarray(y_space, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if y_space.size != x.size:
        raise ValueError("y_space and x must have same shape")

    if y_space.size == 0:
        return 0.0

    m = y_space > eps
    if np.count_nonzero(m) == 0:
        return 0.0

    ratio = x[m] / y_space[m]
    return float(np.quantile(ratio, float(q)))


# ----------------------------
# (B1) Original patch model
# ----------------------------

def eval_origin_patch_yspace(
    a0: float,
    y_space_query: np.ndarray,
    y_space_min_data: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Piecewise upper bound in y_space:
      e = a0 * y_space_min_data
      ub(y) = e         if y <= y_space_min_data
              a0*y      otherwise
    """
    yq = np.asarray(y_space_query, dtype=np.float32).reshape(-1)
    y0 = float(y_space_min_data)

    e = float(a0 * y0)
    x_origin = (a0 * yq).astype(np.float32)
    x_flat = np.full_like(yq, fill_value=np.float32(e), dtype=np.float32)

    ub = np.where(yq <= y0, x_flat, x_origin).astype(np.float32)
    ub = np.maximum(ub, 0.0).astype(np.float32)

    meta = {"a0": float(a0), "y_space_min_data": float(y0), "e": float(e)}
    return ub, meta


def _violation_rate(x: np.ndarray, ub: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x).reshape(-1)
    ub = np.asarray(ub).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x > (ub + eps)))


def _violation_by_y_quantiles(y_space: np.ndarray, x: np.ndarray, ub: np.ndarray) -> Dict[str, float]:
    """
    Report violation rates in regions based on y_space quantiles: <=p10, p10..p90, >p90
    """
    y_space = np.asarray(y_space).reshape(-1)
    x = np.asarray(x).reshape(-1)
    ub = np.asarray(ub).reshape(-1)
    if x.size == 0:
        return {}

    eps = 1e-8
    viol = (x > (ub + eps))

    p10, p90 = np.quantile(y_space, [0.1, 0.9]).astype(np.float32)
    m_lo = y_space <= p10
    m_mid = (y_space > p10) & (y_space <= p90)
    m_hi = y_space > p90

    def rate(m):
        if int(np.sum(m)) == 0:
            return float("nan")
        return float(np.mean(viol[m]))

    return {
        "overall": float(np.mean(viol)),
        "y<=p10": rate(m_lo),
        "p10<y<=p90": rate(m_mid),
        "y>p90": rate(m_hi),
        "p10": float(p10),
        "p90": float(p90),
    }


def _load_xy_for_earlystop(
    npz_path: str,
    *,
    x_key: str,
    y_key: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    dat = np.load(npz_path, allow_pickle=False)

    if x_key not in dat.files:
        raise KeyError(f"Missing x_key='{x_key}' in {npz_path}. keys={dat.files}")
    if y_key not in dat.files:
        raise KeyError(f"Missing y_key='{y_key}' in {npz_path}. keys={dat.files}")

    x = dat[x_key].astype(np.float32, copy=False).reshape(-1)
    y_raw = dat[y_key].astype(np.float32, copy=False).reshape(-1)

    if x.size != y_raw.size:
        raise ValueError(f"x/y size mismatch: x={x.size} y={y_raw.size}")

    meta = None
    if "meta" in dat.files:
        meta = _maybe_parse_meta(dat["meta"])
    return x, y_raw, meta


def _plot_scatter_with_bounds_yspace(
    *,
    x: np.ndarray,
    y_space: np.ndarray,
    a0: float,
    y_space_min_data: float,
    out_png: str,
    max_points: int = 300_000,
    seed: int = 123,
    title_prefix: str = "",
):
    """
    Scatter plot in y_space (L2^2) with:
      - origin: a0*y
      - origin+patch
    """
    _ensure_dir(os.path.dirname(out_png) or ".")

    n = int(x.size)
    if n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=int(max_points), replace=False)
        xs = x[sel]
        ys = y_space[sel]
    else:
        xs = x
        ys = y_space

    plt.figure(figsize=(9, 6))
    plt.scatter(ys, xs, s=2, alpha=0.20, label="samples")

    y_max = float(np.max(y_space)) if y_space.size else 1.0
    y_line = np.linspace(0.0, y_max, 512, dtype=np.float32)

    ub_origin = (a0 * y_line).astype(np.float32)
    ub_patch, _ = eval_origin_patch_yspace(a0, y_line, y_space_min_data)

    plt.plot(y_line, ub_origin, linewidth=2.5, linestyle="--", label="origin: a0*y")
    plt.plot(y_line, ub_patch, linewidth=3.0, label="patch: origin + flat")

    plt.title(f"{title_prefix} x vs y_space (L2^2) early-stop (patch)")
    plt.xlabel("y_space = L2^2")
    plt.ylabel("x")
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def train_export_earlystop_from_npz(
    *,
    train_npz: str,
    out_dir: str,
    q: float = 0.995,
    x_key: str = "x2",
    y_key: str = "y2",
    y_transform: Optional[str] = None,   # "identity" or "square"; if None -> auto from meta.metric
    save_prefix: str = "earlystop",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Train + export early-stop parameters on training NPZ. (PATCH mode)
    """
    _ensure_dir(out_dir)

    x, y_raw, meta = _load_xy_for_earlystop(train_npz, x_key=x_key, y_key=y_key)

    if y_transform is None:
        if meta is not None and "metric" in meta:
            m = str(meta["metric"]).lower()
            if m == "l2":
                y_transform = "square"
            elif m == "l2sq":
                y_transform = "identity"
            else:
                y_transform = "identity"
        else:
            y_transform = "identity"

    y_space = _to_yspace_l2sq(y_raw, y_transform=y_transform)

    y_space_min_data = float(np.min(y_space)) if y_space.size else 0.0
    a0 = fit_origin_upper_bound_yspace(y_space, x, q=q)
    ub_patch, meta_patch = eval_origin_patch_yspace(a0, y_space, y_space_min_data)

    v_patch = _violation_rate(x, ub_patch)
    v_by = _violation_by_y_quantiles(y_space, x, ub_patch)

    params = {
        "method": "origin_plus_flat_patch_in_yspace_L2sq",
        "q": float(q),
        "a0": float(a0),
        "y_space_min_data": float(y_space_min_data),
        "e": float(meta_patch["e"]),
        "train_npz": train_npz,
        "x_key": x_key,
        "y_key": y_key,
        "y_transform": str(y_transform),
        "train_violation_patch": float(v_patch),
        "train_violation_by_y_quantiles": v_by,
        "train_meta": meta,
        "x_stats": _percentile_stats(x),
        "y_space_stats": _percentile_stats(y_space),
    }

    out_json = os.path.join(out_dir, f"{save_prefix}_params.json")
    out_npz = os.path.join(out_dir, f"{save_prefix}_params.npz")

    with open(out_json, "w") as f:
        json.dump(params, f, indent=2)

    np.savez(
        out_npz,
        q=np.float32(q),
        a0=np.float32(a0),
        y_space_min_data=np.float32(y_space_min_data),
        e=np.float32(meta_patch["e"]),
        meta=json.dumps(params),
    )

    out_plot = None
    if plot:
        out_plot = os.path.join(out_dir, f"{save_prefix}_train_scatter_yspace.png")
        _plot_scatter_with_bounds_yspace(
            x=x,
            y_space=y_space,
            a0=a0,
            y_space_min_data=y_space_min_data,
            out_png=out_plot,
            title_prefix="[TRAIN]",
        )

    return {
        "train_npz": train_npz,
        "out_dir": out_dir,
        "params_json": out_json,
        "params_npz": out_npz,
        "train_plot": out_plot,
        "params": params,
    }


def eval_earlystop_on_npz(
    *,
    params_json: str,
    eval_npz: str,
    out_dir: str,
    x_key: str = "x_q_only",
    y_key: str = "y",
    y_transform: str = "identity",
    save_prefix: str = "earlystop",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate early-stop parameters on eval NPZ. (PATCH mode)
    """
    _ensure_dir(out_dir)

    with open(params_json, "r") as f:
        params = json.load(f)

    a0 = float(params["a0"])
    y_space_min_data = float(params["y_space_min_data"])

    x, y_raw, meta = _load_xy_for_earlystop(eval_npz, x_key=x_key, y_key=y_key)
    y_space = _to_yspace_l2sq(y_raw, y_transform=y_transform)

    ub_patch, meta_patch = eval_origin_patch_yspace(a0, y_space, y_space_min_data)

    v_patch = _violation_rate(x, ub_patch)
    v_by = _violation_by_y_quantiles(y_space, x, ub_patch)

    report = {
        "params_json": params_json,
        "eval_npz": eval_npz,
        "x_key": x_key,
        "y_key": y_key,
        "y_transform": str(y_transform),
        "a0": float(a0),
        "y_space_min_data": float(y_space_min_data),
        "e": float(meta_patch["e"]),
        "eval_violation_patch": float(v_patch),
        "eval_violation_by_y_quantiles": v_by,
        "eval_meta": meta,
        "x_stats": _percentile_stats(x),
        "y_space_stats": _percentile_stats(y_space),
    }

    out_json = os.path.join(out_dir, f"{save_prefix}_eval_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    out_plot = None
    if plot:
        out_plot = os.path.join(out_dir, f"{save_prefix}_eval_scatter_yspace.png")
        _plot_scatter_with_bounds_yspace(
            x=x,
            y_space=y_space,
            a0=a0,
            y_space_min_data=y_space_min_data,
            out_png=out_plot,
            title_prefix="[EVAL]",
        )

    return {
        "eval_npz": eval_npz,
        "out_dir": out_dir,
        "eval_report_json": out_json,
        "eval_plot": out_plot,
        "report": report,
    }


# ----------------------------
# (B2) shift_b sweep (NEW)
# ----------------------------

def eval_shift_b_yspace(a0: float, b: float, y_space_query: np.ndarray) -> np.ndarray:
    """
    ub(y) = a0*y + b
    """
    yq = np.asarray(y_space_query, dtype=np.float32).reshape(-1)
    ub = (a0 * yq + float(b)).astype(np.float32)
    ub = np.maximum(ub, 0.0).astype(np.float32)
    return ub


def calc_b_for_target_violation(
    *,
    a0: float,
    x: np.ndarray,
    y_space: np.ndarray,
    target_violation: float,
) -> float:
    """
    Given fixed a0, choose b such that training violation ~= target_violation.

    residual = x - a0*y
    Want: P( x > a0*y + b ) = target
          P( residual > b ) = target
          b = quantile_{1-target}(residual)

    Clip b >= 0 for safety.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y_space, dtype=np.float64).reshape(-1)
    tv = float(target_violation)
    tv = min(max(tv, 0.0), 1.0)

    residual = x - (a0 * y)
    q_needed = 1.0 - tv
    b = float(np.quantile(residual, q_needed))
    if b < 0.0:
        b = 0.0
    return b


def _plot_scatter_with_shiftb_yspace(
    *,
    x: np.ndarray,
    y_space: np.ndarray,
    a0: float,
    b: float,
    out_png: str,
    max_points: int = 300_000,
    seed: int = 123,
    title_prefix: str = "",
):
    """
    Scatter plot in y_space (L2^2) with:
      - origin line: a0*y      (dashed)
      - shift_b line: a0*y + b (solid)
    """
    _ensure_dir(os.path.dirname(out_png) or ".")

    n = int(x.size)
    if n > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n, size=int(max_points), replace=False)
        xs = x[sel]
        ys = y_space[sel]
    else:
        xs = x
        ys = y_space

    plt.figure(figsize=(9, 6))
    plt.scatter(ys, xs, s=2, alpha=0.20, label="samples")

    y_max = float(np.max(y_space)) if y_space.size else 1.0
    y_line = np.linspace(0.0, y_max, 512, dtype=np.float32)

    ub_origin = (a0 * y_line).astype(np.float32)
    ub_shiftb = eval_shift_b_yspace(a0, b, y_line)

    plt.plot(y_line, ub_origin, linewidth=2.5, linestyle="--", label="origin: a0*y")
    plt.plot(y_line, ub_shiftb, linewidth=3.0, label=f"shift_b: a0*y + b")

    plt.title(f"{title_prefix} x vs y_space (L2^2) early-stop (shift_b)")
    plt.xlabel("y_space = L2^2")
    plt.ylabel("x")
    plt.grid(True, linewidth=0.3, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def train_export_shiftb_sweep_from_npz(
    *,
    train_npz: str,
    out_dir: str,
    q: float = 0.995,
    x_key: str = "x2",
    y_key: str = "y2",
    y_transform: Optional[str] = None,
    targets: List[float] = (0.02, 0.01, 0.005, 0.0025, 0.001),
    save_prefix: str = "earlystop_shiftb",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Train shift_b sweep on training NPZ.

    Steps:
      1) y_space = L2^2 space via y_transform
      2) fit a0 = quantile_q(x / y_space)
      3) for each target violation:
           b = quantile_{1-target}(x - a0*y_space)
      4) export params json + train scatter plot
    """
    _ensure_dir(out_dir)

    x, y_raw, meta = _load_xy_for_earlystop(train_npz, x_key=x_key, y_key=y_key)

    # auto decide y_transform from meta.metric if user didn't specify
    if y_transform is None:
        if meta is not None and "metric" in meta:
            m = str(meta["metric"]).lower()
            if m == "l2":
                y_transform = "square"
            elif m == "l2sq":
                y_transform = "identity"
            else:
                y_transform = "identity"
        else:
            y_transform = "identity"

    y_space = _to_yspace_l2sq(y_raw, y_transform=y_transform)
    a0 = fit_origin_upper_bound_yspace(y_space, x, q=q)

    sweep_items = []
    print("\n=== shift_b TRAIN sweep ===")
    for tv in list(targets):
        tv = float(tv)
        tag = f"t{tv:.6f}".replace(".", "p")

        b = calc_b_for_target_violation(a0=a0, x=x, y_space=y_space, target_violation=tv)
        ub = eval_shift_b_yspace(a0, b, y_space)

        v = _violation_rate(x, ub)
        v_by = _violation_by_y_quantiles(y_space, x, ub)

        params = {
            "method": "shift_b_in_yspace_L2sq",
            "q": float(q),
            "a0": float(a0),
            "b": float(b),
            "target_train_violation": float(tv),
            "train_npz": train_npz,
            "x_key": x_key,
            "y_key": y_key,
            "y_transform": str(y_transform),
            "train_violation_shiftb": float(v),
            "train_violation_by_y_quantiles": v_by,
            "train_meta": meta,
            "x_stats": _percentile_stats(x),
            "y_space_stats": _percentile_stats(y_space),
        }

        out_params_json = os.path.join(out_dir, f"{save_prefix}_{tag}_params.json")
        with open(out_params_json, "w") as f:
            json.dump(params, f, indent=2)

        out_plot = None
        if plot:
            out_plot = os.path.join(out_dir, f"{save_prefix}_{tag}_train_scatter_yspace.png")
            _plot_scatter_with_shiftb_yspace(
                x=x,
                y_space=y_space,
                a0=a0,
                b=b,
                out_png=out_plot,
                title_prefix=f"[TRAIN] target={tv}  a0={a0:.6g}  b={b:.6g}",
            )

        print(f"[target={tv:<8.6f}] b={b:.6g} | train_violation={v:.6f}")

        sweep_items.append({
            "target": tv,
            "b": float(b),
            "params_json": out_params_json,
            "train_plot": out_plot,
            "train_violation": float(v),
            "train_violation_by_y_quantiles": v_by,
        })

    return {
        "train_npz": train_npz,
        "out_dir": out_dir,
        "a0": float(a0),
        "y_transform": str(y_transform),
        "sweep": sweep_items,
    }


def eval_shiftb_on_npz(
    *,
    params_json: str,
    eval_npz: str,
    out_dir: str,
    x_key: str = "x_q_only",
    y_key: str = "y",
    y_transform: str = "identity",
    save_prefix: str = "earlystop_shiftb",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a single shift_b params_json on eval NPZ.
    """
    _ensure_dir(out_dir)

    with open(params_json, "r") as f:
        params = json.load(f)

    a0 = float(params["a0"])
    b = float(params["b"])
    target = float(params.get("target_train_violation", -1.0))

    x, y_raw, meta = _load_xy_for_earlystop(eval_npz, x_key=x_key, y_key=y_key)
    y_space = _to_yspace_l2sq(y_raw, y_transform=y_transform)

    ub = eval_shift_b_yspace(a0, b, y_space)

    v = _violation_rate(x, ub)
    v_by = _violation_by_y_quantiles(y_space, x, ub)

    tag = f"t{target:.6f}".replace(".", "p") if target >= 0 else "unknown"
    out_report = os.path.join(out_dir, f"{save_prefix}_{tag}_eval_report.json")

    report = {
        "params_json": params_json,
        "eval_npz": eval_npz,
        "x_key": x_key,
        "y_key": y_key,
        "y_transform": str(y_transform),
        "a0": float(a0),
        "b": float(b),
        "target_train_violation": float(target),
        "eval_violation_shiftb": float(v),
        "eval_violation_by_y_quantiles": v_by,
        "eval_meta": meta,
        "x_stats": _percentile_stats(x),
        "y_space_stats": _percentile_stats(y_space),
    }

    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    out_plot = None
    if plot:
        out_plot = os.path.join(out_dir, f"{save_prefix}_{tag}_eval_scatter_yspace.png")
        _plot_scatter_with_shiftb_yspace(
            x=x,
            y_space=y_space,
            a0=a0,
            b=b,
            out_png=out_plot,
            title_prefix=f"[EVAL] target={target}  a0={a0:.6g}  b={b:.6g}",
        )

    return {
        "eval_npz": eval_npz,
        "out_dir": out_dir,
        "eval_report_json": out_report,
        "eval_plot": out_plot,
        "report": report,
    }


def sweep_shiftb_train_and_eval(
    *,
    train_npz: str,
    eval_npz: str,
    out_dir: str,
    q: float = 0.995,
    train_x_key: str = "x2",
    train_y_key: str = "y2",
    train_y_transform: Optional[str] = None,
    eval_x_key: str = "x_q_only",
    eval_y_key: str = "y",
    eval_y_transform: str = "identity",
    targets: List[float] = (0.02, 0.01, 0.005, 0.0025, 0.001),
    save_prefix: str = "earlystop_shiftb",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Full shift_b sweep:
      - train sweep -> multiple params_json
      - eval each params_json
      - print summary
    """
    tr = train_export_shiftb_sweep_from_npz(
        train_npz=train_npz,
        out_dir=out_dir,
        q=q,
        x_key=train_x_key,
        y_key=train_y_key,
        y_transform=train_y_transform,
        targets=list(targets),
        save_prefix=save_prefix,
        plot=plot,
    )

    print("\n=== shift_b EVAL sweep ===")
    eval_rows = []
    for item in tr["sweep"]:
        ev = eval_shiftb_on_npz(
            params_json=item["params_json"],
            eval_npz=eval_npz,
            out_dir=out_dir,
            x_key=eval_x_key,
            y_key=eval_y_key,
            y_transform=eval_y_transform,
            save_prefix=save_prefix,
            plot=plot,
        )
        v_eval = float(ev["report"]["eval_violation_shiftb"])
        print(f"[target={item['target']:<8.6f}] b={item['b']:.6g} | eval_violation={v_eval:.6f}")

        eval_rows.append({
            "target": item["target"],
            "b": item["b"],
            "train_violation": item["train_violation"],
            "eval_violation": v_eval,
            "eval_report_json": ev["eval_report_json"],
            "eval_plot": ev["eval_plot"],
        })

    return {"train": tr, "eval_sweep": eval_rows}

# ============================================================
# Part C: GT analysis plots (flip-rank / QC-Hamming / query margin distribution)
# ============================================================

def _safe_get(data: Dict[str, Any], key: str):
    return data[key] if key in data else None


def _plot_bar_save(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
):
    _ensure_dir(os.path.dirname(out_path) or ".")
    plt.figure(figsize=(9, 5))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.3, alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_step_hist_save(
    hist_counts: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
):
    """
    hist_counts: [B+1], integer counts for bins 0..B
    """
    _ensure_dir(os.path.dirname(out_path) or ".")
    h = np.asarray(hist_counts).reshape(-1).astype(np.float64)
    xs = np.arange(h.size, dtype=np.int32)  # 0..B
    plt.figure(figsize=(9, 5))
    plt.step(xs, h, where="mid")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_margin_band_save(
    mean: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
):
    """
    mean/p10/p50/p90: [B] sorted margin by rank (small->large)
    """
    _ensure_dir(os.path.dirname(out_path) or ".")
    mean = np.asarray(mean).reshape(-1).astype(np.float32)
    p10 = np.asarray(p10).reshape(-1).astype(np.float32)
    p50 = np.asarray(p50).reshape(-1).astype(np.float32)
    p90 = np.asarray(p90).reshape(-1).astype(np.float32)
    B = int(mean.size)
    xs = np.arange(B, dtype=np.int32)

    plt.figure(figsize=(9, 5))
    plt.fill_between(xs, p10, p90, alpha=0.25, label="p10..p90")
    plt.plot(xs, p50, linewidth=2.5, label="p50")
    plt.plot(xs, mean, linewidth=2.0, linestyle="--", label="mean")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.3, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def visualize_eval_gt_extra_stats(
    eval_npz: str,
    *,
    out_dir: str,
    prefix: str = "eval",
    save_sample_heatmap: bool = False,
) -> Dict[str, Any]:
    """
    Draw GT extra plots if present in eval NPZ:
      - qc_hamming_hist
      - gt_flip_rank_counts / probs
      - query margin sorted stats curves

    This expects you already exported these keys in eval npz:
      qc_hamming_hist:        int64 [B+1]
      gt_flip_rank_counts:    int64 [B]
      gt_flip_rank_probs:     float32 [B]
      q_margin_sorted_mean:   float32 [B]
      q_margin_sorted_p10:    float32 [B]
      q_margin_sorted_p50:    float32 [B]
      q_margin_sorted_p90:    float32 [B]
      q_margin_sorted_sample: float32 [Qs,B]  (optional)
    """
    _ensure_dir(out_dir)
    data = _load_npz(eval_npz)

    plots = {}
    meta = _maybe_parse_meta(data.get("meta", None))

    # ------------------------------------------------
    # (C1) QC-Hamming histogram
    # ------------------------------------------------
    qc_hist = _safe_get(data, "qc_hamming_hist")
    if qc_hist is not None:
        out_png = os.path.join(out_dir, f"{prefix}_qc_hamming_hist.png")
        _plot_step_hist_save(
            qc_hist,
            title=f"{prefix}: QC-Hamming histogram (GT pairs)",
            xlabel="Hamming distance (query_code XOR gt_candidate_code)",
            ylabel="count",
            out_path=out_png,
        )
        plots["qc_hamming_hist"] = {"path": out_png, "bins": int(np.asarray(qc_hist).size)}

    # ------------------------------------------------
    # (C2) GT flip-rank distribution
    # ------------------------------------------------
    flip_counts = _safe_get(data, "gt_flip_rank_counts")
    flip_probs = _safe_get(data, "gt_flip_rank_probs")

    if flip_counts is not None:
        flip_counts = np.asarray(flip_counts).reshape(-1)
        xs = np.arange(flip_counts.size, dtype=np.int32)
        out_png = os.path.join(out_dir, f"{prefix}_gt_flip_rank_counts.png")
        _plot_bar_save(
            xs, flip_counts,
            title=f"{prefix}: GT flip-rank counts (by query margin-rank)",
            xlabel="rank (0 = smallest margin bit)",
            ylabel="count",
            out_path=out_png,
        )
        plots["gt_flip_rank_counts"] = {"path": out_png, "B": int(flip_counts.size)}

    if flip_probs is not None:
        flip_probs = np.asarray(flip_probs).reshape(-1)
        xs = np.arange(flip_probs.size, dtype=np.int32)
        out_png = os.path.join(out_dir, f"{prefix}_gt_flip_rank_probs.png")
        _plot_bar_save(
            xs, flip_probs,
            title=f"{prefix}: GT flip-rank probabilities (normalized)",
            xlabel="rank (0 = smallest margin bit)",
            ylabel="probability",
            out_path=out_png,
        )
        plots["gt_flip_rank_probs"] = {"path": out_png, "B": int(flip_probs.size)}

    # ------------------------------------------------
    # (C3) Query margin sorted distribution (small->large)
    # ------------------------------------------------
    m_mean = _safe_get(data, "q_margin_sorted_mean")
    m_p10 = _safe_get(data, "q_margin_sorted_p10")
    m_p50 = _safe_get(data, "q_margin_sorted_p50")
    m_p90 = _safe_get(data, "q_margin_sorted_p90")

    if (m_mean is not None) and (m_p10 is not None) and (m_p50 is not None) and (m_p90 is not None):
        out_png = os.path.join(out_dir, f"{prefix}_query_margin_sorted_band.png")
        _plot_margin_band_save(
            m_mean, m_p10, m_p50, m_p90,
            title=f"{prefix}: Query margin distribution (sorted bits: small->large)",
            xlabel="bit-rank (0=closest to threshold, most fragile)",
            ylabel="|enc - thr|",
            out_path=out_png,
        )
        plots["query_margin_sorted_band"] = {"path": out_png, "B": int(np.asarray(m_mean).size)}

    # optional: heatmap-like image from q_margin_sorted_sample
    if save_sample_heatmap:
        m_samp = _safe_get(data, "q_margin_sorted_sample")
        if m_samp is not None:
            out_png = os.path.join(out_dir, f"{prefix}_query_margin_sorted_sample_heat.png")
            mat = np.asarray(m_samp).astype(np.float32, copy=False)
            plt.figure(figsize=(9, 5))
            plt.imshow(mat, aspect="auto")
            plt.colorbar()
            plt.title(f"{prefix}: Sample query sorted margins heatmap (Qs x B)")
            plt.xlabel("bit-rank (0=small margin)")
            plt.ylabel("sampled queries")
            plt.tight_layout()
            _ensure_dir(os.path.dirname(out_png) or ".")
            plt.savefig(out_png, dpi=220)
            plt.close()
            plots["query_margin_sorted_sample_heat"] = {"path": out_png, "shape": [int(mat.shape[0]), int(mat.shape[1])]}

    return {
        "eval_npz": eval_npz,
        "out_dir": out_dir,
        "meta": meta,
        "plots": plots,
    }

# ============================================================
# One-call wrapper: scatter + earlystop export+eval
# ============================================================

def visualize_train_eval_and_earlystop(
    *,
    train_npz: Optional[str],
    eval_npz: Optional[str],
    out_dir: str = "./pair_viz",
    max_points: int = 200_000,
    seed: int = 123,
    alpha: float = 0.12,
    s: float = 2.0,
    # earlystop controls
    earlystop_enable: bool = True,
    earlystop_mode: str = "patch",     # "patch" | "shift_b"
    earlystop_q: float = 0.995,

    earlystop_train_x_key: str = "x2",
    earlystop_train_y_key: str = "y2",
    earlystop_train_y_transform: Optional[str] = None,  # None => auto from train meta.metric

    earlystop_eval_x_key: str = "x_q_only",
    earlystop_eval_y_key: str = "y",
    earlystop_eval_y_transform: str = "identity",       # GT bigann/deep1b => identity

    earlystop_prefix: str = "earlystop",
    earlystop_plot: bool = True,

    # shift_b sweep controls (NEW)
    shiftb_targets: List[float] = (0.02, 0.01, 0.005, 0.0025, 0.001),
    shiftb_prefix: str = "earlystop_shiftb",
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) scatter plots for train/eval
      2) earlystop model:
         - mode="patch"   : original origin+flat patch
         - mode="shift_b" : sweep b values and evaluate on eval set

    Returns a dict containing file paths + stats.
    """
    _ensure_dir(out_dir)
    result: Dict[str, Any] = {"out_dir": out_dir}

    if train_npz is not None:
        result["train_scatter"] = visualize_training_npz(
            train_npz,
            out_dir=out_dir,
            prefix="train",
            max_points=max_points,
            seed=seed,
            alpha=alpha,
            s=s,
        )

    if eval_npz is not None:
        result["eval_scatter"] = visualize_eval_npz(
            eval_npz,
            out_dir=out_dir,
            prefix="eval",
            max_points=max_points,
            seed=seed + 10,
            alpha=alpha,
            s=s,
        )
        # NEW: extra GT plots (flip-rank / QC-Hamming / query margin distribution)
        result["eval_gt_extra"] = visualize_eval_gt_extra_stats(
            eval_npz,
            out_dir=out_dir,
            prefix="eval",
            save_sample_heatmap=False,  # True if you want the heatmap too
        )
        

    if earlystop_enable and train_npz is not None:
        mode = str(earlystop_mode).lower().strip()

        if mode == "patch":
            es_train = train_export_earlystop_from_npz(
                train_npz=train_npz,
                out_dir=out_dir,
                q=earlystop_q,
                x_key=earlystop_train_x_key,
                y_key=earlystop_train_y_key,
                y_transform=earlystop_train_y_transform,
                save_prefix=earlystop_prefix,
                plot=earlystop_plot,
            )
            result["earlystop_train"] = es_train

            if eval_npz is not None:
                es_eval = eval_earlystop_on_npz(
                    params_json=es_train["params_json"],
                    eval_npz=eval_npz,
                    out_dir=out_dir,
                    x_key=earlystop_eval_x_key,
                    y_key=earlystop_eval_y_key,
                    y_transform=earlystop_eval_y_transform,
                    save_prefix=earlystop_prefix,
                    plot=earlystop_plot,
                )
                result["earlystop_eval"] = es_eval

        elif mode == "shift_b":
            if eval_npz is None:
                raise ValueError("earlystop_mode='shift_b' requires eval_npz to compare sweep results.")

            sweep_out = sweep_shiftb_train_and_eval(
                train_npz=train_npz,
                eval_npz=eval_npz,
                out_dir=out_dir,
                q=earlystop_q,
                train_x_key=earlystop_train_x_key,
                train_y_key=earlystop_train_y_key,
                train_y_transform=earlystop_train_y_transform,
                eval_x_key=earlystop_eval_x_key,
                eval_y_key=earlystop_eval_y_key,
                eval_y_transform=earlystop_eval_y_transform,
                targets=list(shiftb_targets),
                save_prefix=shiftb_prefix,
                plot=earlystop_plot,
            )
            result["shiftb_sweep"] = sweep_out

        else:
            raise ValueError(f"Unknown earlystop_mode='{earlystop_mode}'. Use 'patch' or 'shift_b'.")

    return result


if __name__ == "__main__":
    print("[demo] import this module and call visualize_train_eval_and_earlystop(...)")
