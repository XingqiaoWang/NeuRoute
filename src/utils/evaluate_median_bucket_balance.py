#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bucket balance evaluation + CSR next-input builder (Numba-parallel, streaming).

This file provides TWO main entry points:

  1) evaluate_median_bucket_balance(enc_path, ...)
     - Compute multiple candidate median thresholds (groups/windows)
     - Pack bucket codes for each group
     - Compute bucket-balance stats and pick the "most balanced" group
     - (Optional) build next-step CSR inputs (offsets + indices) using the best threshold

  2) evaluate_bucket_balance_with_threshold(enc_path, threshold, ...)
     - Same packing + stats pipeline, but using a user-provided threshold vector
     - (Optional) build next-step CSR inputs

Key fix in this version:
  ✅ Unified Numba thread configuration (no more "threads must be between 1 and 1" surprises)
     - Only configure threads ONCE per entry function.
     - Sub-functions DO NOT call set_num_threads() unless explicitly enabled.

If you see:
  ValueError: The number of threads must be between 1 and 1
it means:
  numba.config.NUMBA_NUM_THREADS == 1 in your current process
so you cannot set >1 threads at runtime.

To enable more threads, set env BEFORE starting python, e.g.:
  export NUMBA_NUM_THREADS=24
  export NUMBA_THREADING_LAYER=omp   # or tbb if installed
"""

from __future__ import annotations

import os
import time
import json
from typing import Optional, Tuple

import numpy as np
import numba
from numba import njit, prange
from numba import set_num_threads, get_num_threads


# =============================================================================
# Numba thread configuration (UNIFIED)
# =============================================================================

def configure_numba_threads(requested: int | None, *, verbose: bool = True) -> int:
    """
    Safely configure Numba parallel threads for the current process.

    Why you may see "between 1 and 1":
      numba.config.NUMBA_NUM_THREADS is 1, so Numba refuses set_num_threads(n>1).

    Args:
      requested: desired thread count. If None or <=0, use max_allowed.
      verbose: print diagnostic.

    Returns:
      actual thread count after attempting to set.
    """
    max_allowed = int(getattr(numba.config, "NUMBA_NUM_THREADS", 1))
    if max_allowed < 1:
        max_allowed = 1

    req = 0 if requested is None else int(requested)
    use = max_allowed if req <= 0 else min(req, max_allowed)
    use = max(1, use)

    before = int(get_num_threads())
    try:
        if before != use:
            set_num_threads(use)
    except Exception as e:
        if verbose:
            print(f"[numba] set_num_threads({use}) failed: {e}")
        use = int(get_num_threads())

    after = int(get_num_threads())

    if verbose:
        print(f"[numba] requested={requested} max_allowed={max_allowed} before={before} after={after}")
        if req > max_allowed:
            print(
                "[numba] NOTE: runtime max_allowed is small (often 1).\n"
                "        To use more threads, set env var BEFORE starting python, e.g.:\n"
                "          export NUMBA_NUM_THREADS=24\n"
                "          export NUMBA_THREADING_LAYER=omp\n"
            )

    return after


# =============================================================================
# Numba-accelerated bit packing
# =============================================================================

@njit(parallel=True, fastmath=True)
def pack_codes_numba(xchunk: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """
    Pack bucket codes for one threshold vector.

    Args:
        xchunk: (rows, B) float32
        thr:    (B,) float32

    Returns:
        codes:  (rows,) uint32  (bucket id in [0, 2^B))
    """
    rows, B = xchunk.shape
    codes = np.zeros(rows, dtype=np.uint32)
    for i in prange(rows):
        c = np.uint32(0)
        for k in range(B):
            if xchunk[i, k] >= thr[k]:
                c |= (np.uint32(1) << np.uint32(k))
        codes[i] = c
    return codes


def summarize_counts_dense(counts: np.ndarray, topk: int = 10) -> dict:
    """
    Summarize bucket size distribution.

    counts: array length nlist (bucket sizes)
    """
    counts_u64 = counts.astype(np.uint64, copy=False)
    nlist = counts_u64.size

    nonzero = counts_u64[counts_u64 > 0]
    nonempty = int(nonzero.size)
    empty = int(nlist - nonempty)

    if nonempty == 0:
        return {
            "nlist": int(nlist),
            "nonempty": 0,
            "empty": int(nlist),
            "empty_ratio": 1.0,
            "max_bucket": 0,
            "p50": 0, "p90": 0, "p99": 0, "p999": 0, "p9999": 0,
            "topk": [],
        }

    p50  = int(np.percentile(nonzero, 50))
    p90  = int(np.percentile(nonzero, 90))
    p99  = int(np.percentile(nonzero, 99))
    p999 = int(np.percentile(nonzero, 99.9))
    p9999= int(np.percentile(nonzero, 99.99))
    mx   = int(nonzero.max())

    k = min(int(topk), nonempty)
    idx = np.argpartition(counts_u64, -k)[-k:]
    idx = idx[np.argsort(counts_u64[idx])[::-1]]
    top = [(int(i), int(counts_u64[i])) for i in idx if counts_u64[i] > 0]

    return {
        "nlist": int(nlist),
        "nonempty": nonempty,
        "empty": empty,
        "empty_ratio": float(empty / nlist),
        "max_bucket": mx,
        "p50": p50, "p90": p90, "p99": p99, "p999": p999, "p9999": p9999,
        "topk": top,
    }


# =============================================================================
# Choose the most balanced median threshold (best group)
# =============================================================================

def balance_key(rep: dict) -> Tuple:
    """
    Sorting key: smaller is better.
    """
    return (
        int(rep.get("max_bucket", 0)),
        int(rep.get("p999", 0)),
        int(rep.get("p99", 0)),
        float(rep.get("empty_ratio", 1.0)),
        -int(rep.get("nonempty", 0)),
    )


def pick_most_balanced(reports: list) -> int:
    if not reports:
        raise ValueError("reports is empty; cannot pick best group.")
    best_idx = 0
    best_key = balance_key(reports[0])
    for i in range(1, len(reports)):
        k = balance_key(reports[i])
        if k < best_key:
            best_key = k
            best_idx = i
    return best_idx


# =============================================================================
# CSR next-step inputs (offsets + indices)
# =============================================================================

# @njit
# def _scatter_indices_from_codes(
#     codes_u32: np.ndarray,
#     offsets_u64: np.ndarray,
#     out_indices_u32: np.ndarray,
#     start_row: int
# ) -> None:
#     """
#     Given:
#       codes_u32[i] = bucket id of row (start_row + i)
#       offsets_u64: prefix sums, len nlist+1

#     Produce:
#       out_indices_u32[pos] = original row id (uint32)
#     """
#     cursor = offsets_u64[:-1].copy()  # uint64

#     for i in range(codes_u32.size):
#         b = int(codes_u32[i])
#         pos = cursor[b]
#         out_indices_u32[pos] = np.uint32(start_row + i)
#         cursor[b] = pos + 1


# =============================================================================
# CSR next-step inputs (offsets + indices)
# =============================================================================
from numba import njit
import numpy as np

@njit
def _scatter_indices_from_codes(
    codes_u32: np.ndarray,
    cursor_u64: np.ndarray,          # ✅ NEW: global cursor (len=nlist), in-place updated
    out_indices_u32: np.ndarray,
    start_row: int
) -> None:
    """
    Given:
      codes_u32[i] = bucket id of row (start_row + i)
      cursor_u64[b] = next write position for bucket b (initialized from offsets)

    Produce:
      out_indices_u32[pos] = original row id (uint32)
      and advance cursor_u64[b].
    """
    for i in range(codes_u32.size):
        b = int(codes_u32[i])
        pos = cursor_u64[b]
        out_indices_u32[pos] = np.uint32(start_row + i)
        cursor_u64[b] = pos + 1


def _write_raw_bin_atomic(path: str, arr: np.ndarray, io_buffer_mb: int = 16) -> None:
    """
    Atomic write: write to .tmp then replace.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    buf_bytes = 1024 * 1024 * int(io_buffer_mb)
    with open(tmp, "wb", buffering=buf_bytes) as f:
        f.write(np.ascontiguousarray(arr).view(np.uint8))
    os.replace(tmp, path)


def build_csr_inputs_from_best_codes(
    enc: np.ndarray,                 # mmap [N,B] float32-like
    best_thr: np.ndarray,            # [B] float32
    *,
    B: int,
    N: int,
    chunk_rows: int,
    out_dir: str,
    indices_path: Optional[str] = None,
    offsets_path: Optional[str] = None,
    bucket_sizes_path: Optional[str] = None,
    numba_threads: int = 24,
    warmup_rows: int = 200_000,
    io_buffer_mb: int = 16,
    verbose: bool = True,

    # IMPORTANT: do NOT re-configure threads here unless you explicitly want to.
    configure_threads: bool = False,
) -> dict:
    """
    Two-pass streaming CSR construction:
      Pass1: counts (uint64[nlist]) via codes + bincount
      Build offsets (uint64[nlist+1])
      Pass2: scatter original ids into indices file using offsets+cursor

    Outputs:
      - indices.uint32.bin (raw)
      - offsets.uint64.npy (default) or raw .bin
      - bucket_sizes.uint64.bin (raw)
    """
    if configure_threads:
        _ = configure_numba_threads(numba_threads, verbose=verbose)

    nlist = 1 << int(B)
    os.makedirs(out_dir, exist_ok=True)

    if indices_path is None:
        indices_path = os.path.join(out_dir, "indices.uint32.bin")
    if offsets_path is None:
        offsets_path = os.path.join(out_dir, "offsets.uint64.npy")
    if bucket_sizes_path is None:
        bucket_sizes_path = os.path.join(out_dir, "bucket_sizes.uint64.bin")

    # ---- warmup JIT for pack_codes_numba (avoid compile inside timing) ----
    w = min(int(warmup_rows), int(N))
    if w > 0:
        xw = np.asarray(enc[0:w, :], dtype=np.float32)
        _ = pack_codes_numba(xw, best_thr)

    # ---- Pass 1: counts ----
    if verbose:
        print(f"[csr] pass1: counting buckets nlist={nlist:,} ...")

    t1 = time.perf_counter()
    counts_u64 = np.zeros(nlist, dtype=np.uint64)

    for s in range(0, int(N), int(chunk_rows)):
        e = min(s + int(chunk_rows), int(N))
        xchunk = np.asarray(enc[s:e, :], dtype=np.float32)
        codes = pack_codes_numba(xchunk, best_thr)  # uint32
        bc = np.bincount(codes, minlength=nlist).astype(np.uint64, copy=False)
        counts_u64 += bc

        if verbose:
            dt = time.perf_counter() - t1
            done = e
            print(f"[csr] pass1 {done:,}/{N:,} | elapsed {dt:.1f}s | {(done/max(dt,1e-9)):.2f} vec/s")

    offsets_u64 = np.empty(nlist + 1, dtype=np.uint64)
    offsets_u64[0] = 0
    offsets_u64[1:] = np.cumsum(counts_u64, dtype=np.uint64)

    if int(offsets_u64[-1]) != int(N):
        raise RuntimeError(f"[csr] offsets[-1]={int(offsets_u64[-1])} != N={int(N)}")

    # Write offsets + bucket sizes (atomic)
    if offsets_path.endswith(".npy"):
        tmp = offsets_path + ".tmp.npy"
        np.save(tmp, offsets_u64, allow_pickle=False)
        os.replace(tmp, offsets_path)
    else:
        _write_raw_bin_atomic(offsets_path, offsets_u64, io_buffer_mb=io_buffer_mb)

    _write_raw_bin_atomic(bucket_sizes_path, counts_u64, io_buffer_mb=io_buffer_mb)

    t1_done = time.perf_counter() - t1
    if verbose:
        print(f"[csr] pass1 done in {t1_done:.2f}s | max_bucket={int(counts_u64.max(initial=0)):,}")

    # # ---- Pass 2: scatter indices ----
    # if verbose:
    #     print(f"[csr] pass2: writing indices -> {indices_path}")

    # if os.path.exists(indices_path):
    #     os.remove(indices_path)

    # indices_mm = np.memmap(indices_path, mode="w+", dtype=np.uint32, shape=(int(N),))

    # t2 = time.perf_counter()
    # for s in range(0, int(N), int(chunk_rows)):
    #     e = min(s + int(chunk_rows), int(N))
    #     xchunk = np.asarray(enc[s:e, :], dtype=np.float32)
    #     codes = pack_codes_numba(xchunk, best_thr)
    #     _scatter_indices_from_codes(codes, offsets_u64, indices_mm, start_row=s)

    #     if verbose:
    #         dt = time.perf_counter() - t2
    #         done = e
    #         print(f"[csr] pass2 {done:,}/{N:,} | elapsed {dt:.1f}s | {(done/max(dt,1e-9)):.2f} vec/s")

    # indices_mm.flush()
    # del indices_mm

    # t2_done = time.perf_counter() - t2
    # if verbose:
    #     print(f"[csr] pass2 done in {t2_done:.2f}s")
    
    # ---- Pass 2: scatter indices ----
    if verbose:
        print(f"[csr] pass2: writing indices -> {indices_path}")

    if os.path.exists(indices_path):
        os.remove(indices_path)

    indices_mm = np.memmap(indices_path, mode="w+", dtype=np.uint32, shape=(int(N),))

    # ✅ NEW: 全局 cursor，只初始化一次
    cursor_u64 = offsets_u64[:-1].copy()

    t2 = time.perf_counter()
    for s in range(0, int(N), int(chunk_rows)):
        e = min(s + int(chunk_rows), int(N))
        xchunk = np.asarray(enc[s:e, :], dtype=np.float32)
        codes = pack_codes_numba(xchunk, best_thr)

        # ✅ NEW: 传 cursor_u64（不再传 offsets_u64）
        _scatter_indices_from_codes(codes, cursor_u64, indices_mm, start_row=s)

        if verbose:
            dt = time.perf_counter() - t2
            done = e
            print(f"[csr] pass2 {done:,}/{N:,} | elapsed {dt:.1f}s | {(done/max(dt,1e-9)):.2f} vec/s")

    indices_mm.flush()
    del indices_mm

    t2_done = time.perf_counter() - t2
    if verbose:
        print(f"[csr] pass2 done in {t2_done:.2f}s")


    return {
        "indices_u32_bin": indices_path,
        "offsets_u64": offsets_path,
        "bucket_sizes_u64_bin": bucket_sizes_path,
        "timing": {
            "pass1_count_sec": float(t1_done),
            "pass2_scatter_sec": float(t2_done),
        },
    }


# =============================================================================
# Evaluate: median threshold selection
# =============================================================================

def evaluate_median_bucket_balance(
    enc_path: str,
    *,
    num_groups: int = 5,
    window_rows: int = 10_000_000,
    chunk_rows: int = 8_000_000,
    topk: int = 20,
    codes_dir: str = "/dev/shm",
    codes_path: Optional[str] = None,
    keep_codes: bool = False,
    numba_threads: int = 24,
    warmup_rows: int = 200_000,
    out_json_path: Optional[str] = None,
    verbose: bool = True,

    # Build CSR next inputs (optional)
    build_next_inputs: bool = True,
    next_out_dir: str = "/dev/shm/csr_inputs",
    next_indices_path: Optional[str] = None,
    next_offsets_path: Optional[str] = None,
    next_bucket_sizes_path: Optional[str] = None,
) -> dict:
    """
    Compute multiple median thresholds, evaluate bucket balance, pick best group,
    and optionally build CSR inputs for the next stage.
    """
    t_total0 = time.perf_counter()

    # ✅ Unified Numba threads configuration (ONLY HERE)
    actual_threads = configure_numba_threads(numba_threads, verbose=verbose)

    enc = np.load(enc_path, mmap_mode="r")
    if enc.ndim != 2:
        raise ValueError(f"Expected 2D encoder vectors, got shape={enc.shape}")

    N, B = enc.shape
    if not (1 <= int(B) <= 24):
        raise ValueError(f"This script supports B<=24 only. Got B={B}.")
    nlist = 1 << int(B)

    if verbose:
        print(f"[eval] enc={enc_path}")
        print(f"[eval] N={int(N):,}, B={int(B)}, nlist=2^B={int(nlist):,}")
        print(f"[eval] num_groups={num_groups}, window_rows={window_rows:,}, chunk_rows={chunk_rows:,}")
        print(f"[eval] numba_threads(requested)={numba_threads} actual={actual_threads}")
        print(f"[eval] codes_dir={codes_dir}")

    # -------------------------
    # Stage A: thresholds
    # -------------------------
    t_thr0 = time.perf_counter()
    thr = np.empty((int(num_groups), int(B)), dtype=np.float32)

    for g in range(int(num_groups)):
        s = g * int(window_rows)
        e = min(s + int(window_rows), int(N))
        if e <= s:
            raise ValueError(f"Not enough rows for group {g}: s={s}, e={e}, N={N}")
        xw = np.asarray(enc[s:e, :], dtype=np.float32)
        thr[g] = np.median(xw, axis=0).astype(np.float32)

    t_thr = time.perf_counter() - t_thr0
    if verbose:
        print(f"[thr] computed thresholds: shape={thr.shape} in {t_thr:.2f}s")

    # -------------------------
    # Stage B: codes memmap [G,N]
    # -------------------------
    os.makedirs(codes_dir, exist_ok=True)
    if codes_path is None:
        codes_path = os.path.join(codes_dir, f"bucket_codes_G{num_groups}_N{N}_B{B}_{os.getpid()}.npy")

    t_codes_init0 = time.perf_counter()
    codes_mm = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint32, shape=(int(num_groups), int(N)))
    codes_mm.flush()
    t_codes_init = time.perf_counter() - t_codes_init0

    if verbose:
        size_gb = (int(num_groups) * int(N) * 4) / 1e9
        print(f"[codes] init memmap: {codes_path}  ~{size_gb:.2f} GB  ({t_codes_init:.2f}s)")

    # -------------------------
    # Stage C: warmup JIT
    # -------------------------
    t_warm0 = time.perf_counter()
    w = min(int(warmup_rows), int(N))
    if w > 0:
        xw = np.asarray(enc[0:w, :], dtype=np.float32)
        _ = pack_codes_numba(xw, thr[0])
    t_warm = time.perf_counter() - t_warm0

    if verbose:
        print(f"[jit] warmup_rows={w:,} in {t_warm:.2f}s (includes compile on first run)")

    # -------------------------
    # Stage D: pack codes for all groups
    # -------------------------
    t_pack0 = time.perf_counter()
    for s in range(0, int(N), int(chunk_rows)):
        e = min(s + int(chunk_rows), int(N))
        xchunk = np.asarray(enc[s:e, :], dtype=np.float32)
        for g in range(int(num_groups)):
            codes_mm[g, s:e] = pack_codes_numba(xchunk, thr[g])

        if verbose:
            dt = time.perf_counter() - t_pack0
            print(f"[pack] {e:,}/{N:,} rows | elapsed {dt:.1f}s | {(e/max(dt,1e-9)):.2f} vec/s")

    codes_mm.flush()
    t_pack = time.perf_counter() - t_pack0
    if verbose:
        print(f"[pack] finished writing codes in {t_pack:.2f}s")

    # -------------------------
    # Stage E: balance stats for each group
    # -------------------------
    t_cnt0 = time.perf_counter()
    reports = []
    for g in range(int(num_groups)):
        cg = np.asarray(codes_mm[g, :])
        bc = np.bincount(cg, minlength=int(nlist))
        rep = summarize_counts_dense(bc, topk=int(topk))
        rep["group"] = int(g)
        rep["balance_key"] = tuple(balance_key(rep))
        reports.append(rep)
        if verbose:
            print(f"[count] g={g} done | max={rep['max_bucket']:,} | nonempty={rep['nonempty']:,}")
    t_cnt = time.perf_counter() - t_cnt0

    # pick best
    best_idx = pick_most_balanced(reports)
    best_g = int(reports[best_idx]["group"])
    best_thr = thr[best_g].copy()
    best_rep = reports[best_idx]
    best_key = best_rep["balance_key"]

    if verbose:
        print("\n=== Best (most balanced) median threshold ===")
        print(
            f"best_group={best_g} | "
            f"max={best_rep['max_bucket']:,} p99={best_rep['p99']:,} p999={best_rep['p999']:,} "
            f"nonempty={best_rep['nonempty']:,} empty_ratio={best_rep['empty_ratio']:.6f}"
        )
        print(f"best_balance_key={best_key}")

    # -------------------------
    # Stage F: optional CSR next inputs (best threshold only)
    # -------------------------
    next_inputs = None
    t_next0 = time.perf_counter()
    if build_next_inputs:
        next_inputs = build_csr_inputs_from_best_codes(
            enc=enc,
            best_thr=best_thr,
            B=int(B),
            N=int(N),
            chunk_rows=int(chunk_rows),
            out_dir=next_out_dir,
            indices_path=next_indices_path,
            offsets_path=(next_offsets_path or os.path.join(next_out_dir, "offsets.uint64.npy")),
            bucket_sizes_path=(next_bucket_sizes_path or os.path.join(next_out_dir, "bucket_sizes.uint64.bin")),
            numba_threads=numba_threads,
            warmup_rows=warmup_rows,
            io_buffer_mb=16,
            verbose=verbose,
            configure_threads=False,  # threads already configured in entry function
        )
    t_next = time.perf_counter() - t_next0

    # -------------------------
    # Final report payload
    # -------------------------
    result = {
        "enc_path": enc_path,
        "codes_path": codes_path,
        "N": int(N),
        "B": int(B),
        "nlist": int(nlist),
        "num_groups": int(num_groups),
        "window_rows": int(window_rows),
        "chunk_rows": int(chunk_rows),

        # Keep requested + actual to help debugging
        "numba_threads_requested": int(numba_threads),
        "numba_threads_actual": int(actual_threads),

        "timing": {
            "thresholds_sec": float(t_thr),
            "codes_init_sec": float(t_codes_init),
            "jit_warmup_sec": float(t_warm),
            "pack_codes_sec": float(t_pack),
            "bincount_sec": float(t_cnt),
            "build_next_inputs_sec": float(t_next),
            "total_sec": float(time.perf_counter() - t_total0),
        },
        "thresholds": thr.tolist(),
        "reports": reports,
        "best_group": best_g,
        "best_threshold": best_thr.tolist(),
        "best_report": best_rep,
        "best_balance_key": list(best_key),
        "next_inputs": next_inputs,
    }

    if out_json_path is not None:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        tmp = out_json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, out_json_path)
        if verbose:
            print(f"[eval] wrote report json: {out_json_path}")

    print("\n=== Bucket balance comparison (max / p99 / p999 / nonempty / empty_ratio) ===")
    for rep in reports:
        print(
            f"g={rep['group']:2d}  max={rep['max_bucket']:,}  "
            f"p99={rep['p99']:,}  p999={rep['p999']:,}  "
            f"nonempty={rep['nonempty']:,}  empty_ratio={rep['empty_ratio']:.6f}"
        )

    if next_inputs is not None:
        print("\n=== Next-step inputs (CSR) ===")
        print(f"indices_u32_bin: {next_inputs['indices_u32_bin']}")
        print(f"offsets_u64:      {next_inputs['offsets_u64']}")
        print(f"bucket_sizes_bin: {next_inputs['bucket_sizes_u64_bin']}")

    # cleanup codes file if requested
    if not keep_codes:
        try:
            del codes_mm
        except Exception:
            pass
        try:
            os.remove(codes_path)
            if verbose:
                print(f"[cleanup] removed codes file: {codes_path}")
        except Exception as ex:
            if verbose:
                print(f"[cleanup] failed to remove codes file: {codes_path} ({ex})")

    return result


# =============================================================================
# Evaluate: fixed threshold
# =============================================================================

def _percentile_from_counts_dense(bc: np.ndarray, pct: float) -> int:
    """
    Percentile of bucket sizes across ALL buckets (including empty).
    bc: array length nlist, each entry is bucket size.
    pct: 50/99/99.9 etc
    """
    if bc.size == 0:
        return 0
    pct = float(pct)
    if pct <= 0:
        return int(bc.min())
    if pct >= 100:
        return int(bc.max())

    k = int(round((pct / 100.0) * (bc.size - 1)))
    return int(np.partition(bc, k)[k])


def evaluate_bucket_balance_with_threshold(
    enc_path: str,
    *,
    threshold: np.ndarray,
    chunk_rows: int = 8_000_000,
    topk: int = 20,
    codes_dir: str = "/dev/shm",
    codes_path: Optional[str] = None,
    keep_codes: bool = False,
    numba_threads: int = 24,
    warmup_rows: int = 200_000,
    out_json_path: Optional[str] = None,
    verbose: bool = True,

    # Optional CSR next inputs
    build_next_inputs: bool = True,
    next_out_dir: str = "/dev/shm/csr_inputs",
    next_indices_path: Optional[str] = None,
    next_offsets_path: Optional[str] = None,
    next_bucket_sizes_path: Optional[str] = None,
) -> dict:
    """
    Fixed-threshold evaluation:
      - load enc
      - pack codes using provided threshold
      - compute balance stats
      - optional CSR next inputs
    """
    t_total0 = time.perf_counter()

    # ✅ Unified Numba threads configuration (ONLY HERE)
    actual_threads = configure_numba_threads(numba_threads, verbose=verbose)

    enc = np.load(enc_path, mmap_mode="r")
    if enc.ndim != 2:
        raise ValueError(f"Expected 2D encoder vectors, got shape={enc.shape}")

    N, B = enc.shape
    if not (1 <= int(B) <= 24):
        raise ValueError(f"This script supports B<=24 only. Got B={B}.")
    nlist = 1 << int(B)

    thr = np.asarray(threshold, dtype=np.float32).reshape(-1)
    if thr.shape[0] != int(B):
        raise ValueError(f"threshold length mismatch: got len={thr.shape[0]}, expected B={B}")

    if verbose:
        print(f"[eval-fixed] enc={enc_path}")
        print(f"[eval-fixed] N={int(N):,}, B={int(B)}, nlist=2^B={int(nlist):,}")
        print(f"[eval-fixed] numba_threads(requested)={numba_threads} actual={actual_threads}")
        print(f"[eval-fixed] thr[:8]={thr[:8]}")

    # -------------------------
    # Stage A: codes memmap
    # -------------------------
    os.makedirs(codes_dir, exist_ok=True)
    if codes_path is None:
        codes_path = os.path.join(codes_dir, f"bucket_codes_FIXED_N{N}_B{B}_{os.getpid()}.npy")

    t_codes_init0 = time.perf_counter()
    codes_mm = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint32, shape=(int(N),))
    codes_mm.flush()
    t_codes_init = time.perf_counter() - t_codes_init0
    if verbose:
        size_gb = (int(N) * 4) / 1e9
        print(f"[codes] init memmap: {codes_path}  ~{size_gb:.2f} GB  ({t_codes_init:.2f}s)")

    # -------------------------
    # Stage B: warmup
    # -------------------------
    t_warm0 = time.perf_counter()
    w = min(int(warmup_rows), int(N))
    if w > 0:
        xw = np.asarray(enc[0:w, :], dtype=np.float32)
        _ = pack_codes_numba(xw, thr)
    t_warm = time.perf_counter() - t_warm0
    if verbose:
        print(f"[jit] warmup_rows={w:,} in {t_warm:.2f}s")

    # -------------------------
    # Stage C: pack codes
    # -------------------------
    t_pack0 = time.perf_counter()
    for s in range(0, int(N), int(chunk_rows)):
        e = min(s + int(chunk_rows), int(N))
        xchunk = np.asarray(enc[s:e, :], dtype=np.float32)
        codes_mm[s:e] = pack_codes_numba(xchunk, thr)

        if verbose:
            dt = time.perf_counter() - t_pack0
            print(f"[pack] {e:,}/{N:,} rows | elapsed {dt:.1f}s | {(e/max(dt,1e-9)):.2f} vec/s")

    codes_mm.flush()
    t_pack = time.perf_counter() - t_pack0
    if verbose:
        print(f"[pack] finished writing codes in {t_pack:.2f}s")

    # -------------------------
    # Stage D: stats
    # -------------------------
    t_cnt0 = time.perf_counter()
    cg = np.asarray(codes_mm[:])
    bc = np.bincount(cg, minlength=int(nlist))
    rep = summarize_counts_dense(bc, topk=int(topk))
    rep["p50"] = _percentile_from_counts_dense(bc, 50.0)
    t_cnt = time.perf_counter() - t_cnt0

    if verbose:
        print(
            f"[count] done | max={rep['max_bucket']:,} | p50={rep['p50']:,} "
            f"| p99={rep['p99']:,} | p999={rep['p999']:,} | nonempty={rep['nonempty']:,}"
        )

    # -------------------------
    # Stage E: optional CSR next inputs
    # -------------------------
    next_inputs = None
    t_next0 = time.perf_counter()
    if build_next_inputs:
        next_inputs = build_csr_inputs_from_best_codes(
            enc=enc,
            best_thr=thr,
            B=int(B),
            N=int(N),
            chunk_rows=int(chunk_rows),
            out_dir=next_out_dir,
            indices_path=next_indices_path,
            offsets_path=(next_offsets_path or os.path.join(next_out_dir, "offsets.uint64.npy")),
            bucket_sizes_path=(next_bucket_sizes_path or os.path.join(next_out_dir, "bucket_sizes.uint64.bin")),
            numba_threads=numba_threads,
            warmup_rows=warmup_rows,
            io_buffer_mb=16,
            verbose=verbose,
            configure_threads=False,  # threads already configured in entry function
        )
    t_next = time.perf_counter() - t_next0

    # -------------------------
    # Final payload
    # -------------------------
    result = {
        "enc_path": enc_path,
        "codes_path": codes_path,
        "N": int(N),
        "B": int(B),
        "nlist": int(nlist),
        "chunk_rows": int(chunk_rows),

        "numba_threads_requested": int(numba_threads),
        "numba_threads_actual": int(actual_threads),

        "timing": {
            "codes_init_sec": float(t_codes_init),
            "jit_warmup_sec": float(t_warm),
            "pack_codes_sec": float(t_pack),
            "bincount_sec": float(t_cnt),
            "build_next_inputs_sec": float(t_next),
            "total_sec": float(time.perf_counter() - t_total0),
        },
        "threshold": thr.tolist(),
        "report": rep,
        "next_inputs": next_inputs,
    }

    if out_json_path is not None:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        tmp = out_json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2)
        os.replace(tmp, out_json_path)
        if verbose:
            print(f"[eval-fixed] wrote report json: {out_json_path}")

    print("\n=== Bucket balance comparison (max / p50 / p99 / p999 / nonempty / empty_ratio) ===")
    print(
        f"g= 0  max={rep['max_bucket']:,}  p50={rep['p50']:,}  "
        f"p99={rep['p99']:,}  p999={rep['p999']:,}  "
        f"nonempty={rep['nonempty']:,}  empty_ratio={rep['empty_ratio']:.6f}"
    )

    if next_inputs is not None:
        print("\n=== Next-step inputs (CSR) ===")
        print(f"indices_u32_bin: {next_inputs['indices_u32_bin']}")
        print(f"offsets_u64:      {next_inputs['offsets_u64']}")
        print(f"bucket_sizes_bin: {next_inputs['bucket_sizes_u64_bin']}")

    if not keep_codes:
        try:
            del codes_mm
        except Exception:
            pass
        try:
            os.remove(codes_path)
            if verbose:
                print(f"[cleanup] removed codes file: {codes_path}")
        except Exception as ex:
            if verbose:
                print(f"[cleanup] failed to remove codes file: {codes_path} ({ex})")

    return result


# =============================================================================
# CLI demo
# =============================================================================

def main() -> None:
    ENC_PATH = "/dev/shm/encoded_1b_22_f32.npy"

    result = evaluate_median_bucket_balance(
        ENC_PATH,
        num_groups=5,
        window_rows=10_000_000,
        chunk_rows=8_000_000,
        topk=20,
        codes_dir="/dev/shm",
        codes_path="/dev/shm/bucket_codes_tmp.npy",
        keep_codes=False,
        numba_threads=24,          # will be clamped to runtime max_allowed if needed
        warmup_rows=200_000,
        out_json_path=None,
        verbose=True,

        build_next_inputs=True,
        next_out_dir="/dev/shm/csr_inputs_best",
        next_indices_path="/dev/shm/csr_inputs_best/indices.uint32.bin",
        next_offsets_path="/dev/shm/csr_inputs_best/offsets.uint64.npy",
        next_bucket_sizes_path="/dev/shm/csr_inputs_best/bucket_sizes.uint64.bin",
    )

    best_g = result["best_group"]
    best_thr = np.asarray(result["best_threshold"], dtype=np.float32)
    print("\n=== Returned best median (thr) ===")
    print(f"best_group={best_g}, thr.shape={best_thr.shape}, thr[:8]={best_thr[:8]}")


if __name__ == "__main__":
    main()
