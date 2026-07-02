#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic 2D matrix loaders for pipelines.

This module supports BOTH:
  (A) memmap (default): avoid full RAM copy, good for /dev/shm and large npy/bin
  (B) full RAM load: force materialize into RAM & contiguous array

Features:
  - Supports arbitrary user-provided .npy matrices (any dtype), expects shape [N, d].
  - Supports BigANN uint8 official formats:
      u8bin_header, bvecs
  - Supports Deep1B float32 official formats:
      fbin_header (sometimes .fin), fvecs

Compatibility:
  - Keeps old function names used by existing pipelines:
      load_npy_matrix_to_ram
      load_bigann_u8_official_to_ram
      load_deep1b_f32_official_to_ram
      load_u8_matrix_to_ram_auto
      load_f32_matrix_to_ram_auto
      load_any_npy_to_ram

Notes on contiguity:
  - .npy memmap is typically C-contiguous by design.
  - bvecs/fvecs parsing via memmap often produces strided views (NOT C-contiguous).
    If your downstream requires contiguous, set make_contiguous=True (may copy).
"""

from __future__ import annotations

import os
import argparse
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _touch_2d(arr: np.ndarray) -> None:
    """Touch a couple of elements to ensure mapping / paging works."""
    if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
        _ = arr[0, 0]
        _ = arr[-1, -1]
    elif arr.size > 0:
        _ = arr.flat[0]
        _ = arr.flat[-1]


def _ensure_contiguous_if_needed(arr: np.ndarray, make_contiguous: bool) -> np.ndarray:
    """Force C-contiguous if requested (may copy)."""
    if make_contiguous and not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr)
    return arr


# =============================================================================
# Generic .npy loader
# =============================================================================

def load_npy_matrix(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_dtype: Optional[np.dtype] = None,
    cast_dtype: Optional[np.dtype] = None,
    require_2d: bool = True,
    make_contiguous: bool = True,
    prefer_memmap: bool = True,
    mmap_mode: str = "r",
) -> Tuple[np.ndarray, int, int, str]:
    """
    Load an arbitrary .npy matrix.

    Default behavior: memmap (prefer_memmap=True) => np.load(..., mmap_mode="r")
    If prefer_memmap=False => full RAM load.

    Returns:
      arr, N, d, mode_str where mode_str in {"npy_memmap","npy_ram", "..._cast"}
    """
    if not path.endswith(".npy"):
        raise ValueError(f"Not a .npy file: {path}")

    if prefer_memmap:
        arr = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        mode = "npy_memmap"
    else:
        arr = np.load(path, mmap_mode=None, allow_pickle=False)
        mode = "npy_ram"

    if require_2d and arr.ndim != 2:
        raise ValueError(f".npy must be 2D [N,d]. got shape={arr.shape}, ndim={arr.ndim}")

    if expected_dtype is not None and arr.dtype != expected_dtype:
        raise ValueError(f".npy dtype mismatch: {arr.dtype} != expected_dtype={expected_dtype}")

    # Casting will materialize a new array if dtype differs (may defeat memmap).
    if cast_dtype is not None and arr.dtype != cast_dtype:
        arr = arr.astype(cast_dtype, copy=False)
        mode = mode + "_cast"

    N = int(arr.shape[0])
    d = int(arr.shape[1]) if arr.ndim == 2 else 0

    if expected_N is not None and N != int(expected_N):
        raise ValueError(f".npy N mismatch: N={N} != expected_N={expected_N}")
    if expected_d is not None and d != int(expected_d):
        raise ValueError(f".npy d mismatch: d={d} != expected_d={expected_d}")

    arr = _ensure_contiguous_if_needed(arr, make_contiguous=make_contiguous)
    _touch_2d(arr)
    return arr, N, d, mode


def load_npy_matrix_to_ram(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_dtype: Optional[np.dtype] = None,
    cast_dtype: Optional[np.dtype] = None,
    require_2d: bool = True,
    make_contiguous: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """
    Backward-compatible wrapper: force full RAM load.
    (Old pipelines expect this name.)
    """
    arr, N, d, _ = load_npy_matrix(
        path,
        expected_N=expected_N,
        expected_d=expected_d,
        expected_dtype=expected_dtype,
        cast_dtype=cast_dtype,
        require_2d=require_2d,
        make_contiguous=make_contiguous,
        prefer_memmap=False,
    )
    return arr, N, d


# =============================================================================
# BigANN uint8 official loaders
# =============================================================================

def load_bigann_u8_official(
    path: str,
    *,
    d_hint: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_N: Optional[int] = None,
    verify_samples: int = 64,
    prefer_memmap: bool = True,
    make_contiguous: bool = False,  # memmap bvecs view is usually strided; default keep view
) -> Tuple[np.ndarray, int, int, str]:
    """
    Load BigANN uint8 base vectors.

    Auto-detect:
      A) u8bin_header: [u32 N][u32 d] + uint8 matrix (N*d bytes)
      B) bvecs:        N * ([i32 d] + uint8[d])

    prefer_memmap=True:
      - u8bin_header: true memmap into shape (N,d)
      - bvecs: memmap raw bytes, return view raw[:,4:] (likely non-contiguous)
    """
    sz = os.path.getsize(path)
    if sz < 8:
        raise ValueError(f"File too small: {path}")

    # ---- Try u8bin_header ----
    head = np.fromfile(path, dtype=np.uint32, count=2)
    if head.size == 2:
        N0, d0 = int(head[0]), int(head[1])
        if N0 > 0 and d0 > 0:
            expected_bytes = 8 + N0 * d0
            if expected_bytes == sz:
                if expected_d is not None and d0 != int(expected_d):
                    raise ValueError(f"u8bin_header d mismatch: file d={d0} vs expected_d={expected_d}")
                if expected_N is not None and N0 != int(expected_N):
                    raise ValueError(f"u8bin_header N mismatch: file N={N0} vs expected_N={expected_N}")

                if prefer_memmap:
                    vecs = np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(N0, d0), order="C")
                    fmt = "u8bin_header_memmap"
                else:
                    data = np.fromfile(path, dtype=np.uint8, offset=8, count=N0 * d0)
                    if data.size != N0 * d0:
                        raise ValueError(f"u8bin_header short read: got {data.size}, expected {N0*d0}")
                    vecs = data.reshape(N0, d0)
                    fmt = "u8bin_header_ram"

                vecs = _ensure_contiguous_if_needed(vecs, make_contiguous=make_contiguous)
                _verify_u8(vecs, N0, d0, require_contiguous=make_contiguous)
                return vecs, N0, d0, fmt

    # ---- Try bvecs ----
    if d_hint is None:
        d_first = np.fromfile(path, dtype=np.int32, count=1)
        if d_first.size != 1:
            raise ValueError(f"Cannot read first int32 d from {path}")
        d = int(d_first[0])
    else:
        d = int(d_hint)

    if d <= 0 or d > 65535:
        raise ValueError(f"Invalid d={d} for bvecs parse: {path}")
    if expected_d is not None and d != int(expected_d):
        raise ValueError(f"bvecs d mismatch: parsed d={d} vs expected_d={expected_d}")

    rec = 4 + d
    if sz % rec != 0:
        raise ValueError(
            f"Cannot parse {path} as u8bin_header or bvecs. "
            f"file_size={sz}, rec_size(4+d)={rec} (d={d}). Try correct d_hint."
        )
    N = sz // rec
    if expected_N is not None and int(N) != int(expected_N):
        raise ValueError(f"bvecs N mismatch: file N={N} vs expected_N={expected_N}")

    if prefer_memmap:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        fmt = "bvecs_memmap"
    else:
        raw = np.fromfile(path, dtype=np.uint8)
        fmt = "bvecs_ram"

    if raw.size != N * rec:
        raise ValueError(f"bvecs short read: got {raw.size}, expected {N*rec}")
    raw = raw.reshape(int(N), int(rec))

    d_check0 = raw[:1, :4].view(np.int32)[0, 0]
    if int(d_check0) != d:
        raise ValueError(f"bvecs sanity failed: first d={int(d_check0)} != expected d={d}")

    S = min(int(verify_samples), int(N))
    if S > 1:
        idx = np.linspace(0, int(N) - 1, num=S, dtype=np.int64)
        d_fields = raw[idx, :4].reshape(-1, 4).view(np.int32).reshape(-1)
        bad = np.nonzero(d_fields != d)[0]
        if bad.size > 0:
            k = min(8, bad.size)
            ex = [(int(idx[bad[i]]), int(d_fields[bad[i]])) for i in range(k)]
            raise ValueError(f"bvecs sanity failed: found rows with d!=expected. examples={ex}")

    vecs = raw[:, 4:]  # view; likely strided (non-contiguous)
    vecs = _ensure_contiguous_if_needed(vecs, make_contiguous=make_contiguous)
    _verify_u8(vecs, int(N), int(d), require_contiguous=make_contiguous)
    return vecs, int(N), int(d), fmt


def load_bigann_u8_official_to_ram(
    path: str,
    *,
    d_hint: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_N: Optional[int] = None,
    verify_samples: int = 64,
) -> Tuple[np.ndarray, int, int, str]:
    """Backward-compatible wrapper: force RAM + contiguous."""
    return load_bigann_u8_official(
        path,
        d_hint=d_hint,
        expected_d=expected_d,
        expected_N=expected_N,
        verify_samples=verify_samples,
        prefer_memmap=False,
        make_contiguous=True,
    )


def _verify_u8(arr: np.ndarray, N: int, d: int, *, require_contiguous: bool) -> None:
    if arr.dtype != np.uint8:
        raise ValueError(f"uint8 dtype mismatch: {arr.dtype}")
    if arr.ndim != 2 or arr.shape != (int(N), int(d)):
        raise ValueError(f"uint8 shape mismatch: {arr.shape} != {(int(N), int(d))}")
    if require_contiguous and (not arr.flags["C_CONTIGUOUS"]):
        raise ValueError("uint8 array is not C-contiguous (enable make_contiguous=True to force copy)")
    _touch_2d(arr)


# =============================================================================
# Deep1B float32 official loaders
# =============================================================================

def load_deep1b_f32_official(
    path: str,
    *,
    d_hint: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_N: Optional[int] = None,
    verify_samples: int = 64,
    prefer_memmap: bool = True,
    make_contiguous: bool = False,  # fvecs memmap view is usually strided; default keep view
) -> Tuple[np.ndarray, int, int, str]:
    """
    Load Deep1B float32 base vectors.

    Auto-detect:
      A) fbin_header: [u32 N][u32 d] + float32 matrix (N*d*4 bytes)
      B) fvecs:       N * ([i32 d] + float32[d])

    prefer_memmap=True:
      - fbin_header: true memmap into shape (N,d)
      - fvecs: memmap as int32 matrix (N,1+d) then view float32 (aligned)
    """
    sz = os.path.getsize(path)
    if sz < 8:
        raise ValueError(f"File too small: {path}")

    # ---- Try fbin_header ----
    head = np.fromfile(path, dtype=np.uint32, count=2)
    if head.size == 2:
        N0, d0 = int(head[0]), int(head[1])
        if N0 > 0 and d0 > 0:
            expected_bytes = 8 + (N0 * d0 * 4)
            if expected_bytes == sz:
                if expected_d is not None and d0 != int(expected_d):
                    raise ValueError(f"fbin_header d mismatch: file d={d0} vs expected_d={expected_d}")
                if expected_N is not None and N0 != int(expected_N):
                    raise ValueError(f"fbin_header N mismatch: file N={N0} vs expected_N={expected_N}")

                if prefer_memmap:
                    vecs = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(N0, d0), order="C")
                    fmt = "fbin_header_memmap"
                else:
                    data = np.fromfile(path, dtype=np.float32, offset=8, count=N0 * d0)
                    if data.size != N0 * d0:
                        raise ValueError(f"fbin_header short read: got {data.size}, expected {N0*d0}")
                    vecs = data.reshape(N0, d0)
                    fmt = "fbin_header_ram"

                vecs = _ensure_contiguous_if_needed(vecs, make_contiguous=make_contiguous)
                _verify_f32(vecs, N0, d0, require_contiguous=make_contiguous)
                return vecs, N0, d0, fmt

    # ---- Try fvecs ----
    if d_hint is None:
        d_first = np.fromfile(path, dtype=np.int32, count=1)
        if d_first.size != 1:
            raise ValueError(f"Cannot read first int32 d from {path}")
        d = int(d_first[0])
    else:
        d = int(d_hint)

    if d <= 0 or d > 1_000_000:
        raise ValueError(f"Invalid d={d} for fvecs parse: {path}")
    if expected_d is not None and d != int(expected_d):
        raise ValueError(f"fvecs d mismatch: parsed d={d} vs expected_d={expected_d}")

    rec_bytes = 4 + 4 * d
    if sz % rec_bytes != 0:
        raise ValueError(
            f"Cannot parse {path} as fbin_header or fvecs. "
            f"file_size={sz}, rec_size(4+4*d)={rec_bytes} (d={d}). Try correct d_hint."
        )
    N = sz // rec_bytes
    if expected_N is not None and int(N) != int(expected_N):
        raise ValueError(f"fvecs N mismatch: file N={N} vs expected_N={expected_N}")

    if prefer_memmap:
        mm_i32 = np.memmap(path, dtype=np.int32, mode="r", shape=(int(N), int(1 + d)), order="C")
        if int(mm_i32[0, 0]) != d:
            raise ValueError(f"fvecs sanity failed: first d={int(mm_i32[0,0])} != expected d={d}")

        S = min(int(verify_samples), int(N))
        if S > 1:
            idx = np.linspace(0, int(N) - 1, num=S, dtype=np.int64)
            d_fields = np.asarray(mm_i32[idx, 0], dtype=np.int32)
            bad = np.nonzero(d_fields != d)[0]
            if bad.size > 0:
                k = min(8, bad.size)
                ex = [(int(idx[bad[i]]), int(d_fields[bad[i]])) for i in range(k)]
                raise ValueError(f"fvecs sanity failed: found rows with d!=expected. examples={ex}")

        vecs = mm_i32[:, 1:].view(np.float32)  # (N,d) view, aligned
        fmt = "fvecs_memmap"
    else:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size != int(N) * rec_bytes:
            raise ValueError(f"fvecs short read: got {raw.size}, expected {int(N)*rec_bytes}")
        raw = raw.reshape(int(N), int(rec_bytes))

        d_check0 = raw[:1, :4].view(np.int32)[0, 0]
        if int(d_check0) != d:
            raise ValueError(f"fvecs sanity failed: first d={int(d_check0)} != expected d={d}")

        vecs = raw[:, 4:].view(np.float32).reshape(int(N), int(d))
        fmt = "fvecs_ram"

    vecs = _ensure_contiguous_if_needed(vecs, make_contiguous=make_contiguous)
    _verify_f32(vecs, int(N), int(d), require_contiguous=make_contiguous)
    return vecs, int(N), int(d), fmt


def load_deep1b_f32_official_to_ram(
    path: str,
    *,
    d_hint: Optional[int] = None,
    expected_d: Optional[int] = None,
    expected_N: Optional[int] = None,
    verify_samples: int = 64,
) -> Tuple[np.ndarray, int, int, str]:
    """Backward-compatible wrapper: force RAM + contiguous."""
    return load_deep1b_f32_official(
        path,
        d_hint=d_hint,
        expected_d=expected_d,
        expected_N=expected_N,
        verify_samples=verify_samples,
        prefer_memmap=False,
        make_contiguous=True,
    )


def _verify_f32(arr: np.ndarray, N: int, d: int, *, require_contiguous: bool) -> None:
    if arr.dtype != np.float32:
        raise ValueError(f"float32 dtype mismatch: {arr.dtype}")
    if arr.ndim != 2 or arr.shape != (int(N), int(d)):
        raise ValueError(f"float32 shape mismatch: {arr.shape} != {(int(N), int(d))}")
    if require_contiguous and (not arr.flags["C_CONTIGUOUS"]):
        raise ValueError("float32 array is not C-contiguous (enable make_contiguous=True to force copy)")
    _touch_2d(arr)


# =============================================================================
# Unified "auto" entry points (prefer memmap by default)
# =============================================================================

def load_u8_matrix_to_ram_auto(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    d_hint: Optional[int] = None,
    prefer_memmap: bool = True,
    make_contiguous: bool = False,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Auto loader for uint8 matrices.
      - If path endswith .npy: load arbitrary user npy (uint8).
      - Else: parse official BigANN uint8 formats.
    """
    if path.endswith(".npy"):
        arr, N, d, fmt = load_npy_matrix(
            path,
            expected_N=expected_N,
            expected_d=expected_d,
            expected_dtype=np.uint8,
            cast_dtype=None,
            prefer_memmap=prefer_memmap,
            make_contiguous=make_contiguous,
        )
        return arr, N, d, fmt
    return load_bigann_u8_official(
        path,
        d_hint=d_hint,
        expected_d=expected_d,
        expected_N=expected_N,
        prefer_memmap=prefer_memmap,
        make_contiguous=make_contiguous,
    )


def load_f32_matrix_auto(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    d_hint: Optional[int] = None,
    prefer_memmap: bool = True,
    make_contiguous: bool = False,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Auto loader for float32 matrices.
      - If path endswith .npy: load arbitrary user npy (float32).
      - Else: parse official Deep1B float32 formats.
    """
    if path.endswith(".npy"):
        arr, N, d, fmt = load_npy_matrix(
            path,
            expected_N=expected_N,
            expected_d=expected_d,
            expected_dtype=np.float32,
            cast_dtype=None,
            prefer_memmap=prefer_memmap,
            make_contiguous=make_contiguous,
        )
        return arr, N, d, fmt
    return load_deep1b_f32_official(
        path,
        d_hint=d_hint,
        expected_d=expected_d,
        expected_N=expected_N,
        prefer_memmap=prefer_memmap,
        make_contiguous=make_contiguous,
    )


def load_f32_matrix_to_ram_auto(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    d_hint: Optional[int] = None,
    prefer_memmap: bool = True,
    make_contiguous: bool = False,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Backward-compatible alias: old pipelines call load_f32_matrix_to_ram_auto().
    It now supports memmap by default (prefer_memmap=True).
    """
    return load_f32_matrix_auto(
        path,
        expected_N=expected_N,
        expected_d=expected_d,
        d_hint=d_hint,
        prefer_memmap=prefer_memmap,
        make_contiguous=make_contiguous,
    )


def load_f32_matrix_to_ram_auto_force_ram(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    d_hint: Optional[int] = None,
    make_contiguous: bool = True,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Optional helper: if you explicitly want full RAM materialization.
    """
    return load_f32_matrix_auto(
        path,
        expected_N=expected_N,
        expected_d=expected_d,
        d_hint=d_hint,
        prefer_memmap=False,
        make_contiguous=make_contiguous,
    )


def load_any_npy(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    cast_dtype: Optional[str] = None,
    prefer_memmap: bool = True,
    make_contiguous: bool = True,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Convenience loader for arbitrary customer .npy matrices (any dtype).
    Optionally cast to a dtype by name, e.g. "float32", "uint8".

    Note: casting forces materialization (copy) if dtype differs.
    """
    cd = np.dtype(cast_dtype) if cast_dtype is not None else None
    arr, N, d, fmt = load_npy_matrix(
        path,
        expected_N=expected_N,
        expected_d=expected_d,
        expected_dtype=None,
        cast_dtype=cd,
        prefer_memmap=prefer_memmap,
        make_contiguous=make_contiguous,
    )
    return arr, N, d, fmt


def load_any_npy_to_ram(
    path: str,
    *,
    expected_N: Optional[int] = None,
    expected_d: Optional[int] = None,
    cast_dtype: Optional[str] = None,
) -> Tuple[np.ndarray, int, int, str]:
    """
    Backward-compatible wrapper: force RAM + contiguous.
    This name is used by some pipelines (e.g., early_stop_function_evaluation).
    """
    return load_any_npy(
        path,
        expected_N=expected_N,
        expected_d=expected_d,
        cast_dtype=cast_dtype,
        prefer_memmap=False,
        make_contiguous=True,
    )


# =============================================================================
# CLI main
# =============================================================================

def _print_quick_stats(arr: np.ndarray, N: int, d: int, fmt: str, path: str, sample_rows: int) -> None:
    print("\n=== Load OK ===")
    print(f"path: {path}")
    print(f"fmt : {fmt}")
    print(f"shape: ({N}, {d})")
    print(f"dtype: {arr.dtype}  contiguous={arr.flags['C_CONTIGUOUS']}")
    try:
        print(f"nbytes: {arr.nbytes/1e9:.3f} GB")
    except Exception:
        pass

    S = min(int(sample_rows), int(N))
    if S <= 0:
        return
    idx = np.linspace(0, int(N) - 1, num=S, dtype=np.int64)
    samp = arr[idx]

    print("\n=== Sample stats ===")
    if samp.size > 0:
        if np.issubdtype(arr.dtype, np.integer):
            print(f"min={int(samp.min())}  max={int(samp.max())}  mean={float(samp.mean()):.6g}")
        else:
            print(f"min={float(samp.min()):.6g}  max={float(samp.max()):.6g}  mean={float(samp.mean()):.6g}")

    show_rows = min(2, S)
    show_dims = min(16, d)
    print("\n=== First few sampled vectors (first dims) ===")
    for i in range(show_rows):
        r = int(idx[i])
        print(f"row[{r}] first{show_dims}: {samp[i, :show_dims].tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Load base vectors with optional memmap (default ON).")
    ap.add_argument("--path", required=True, type=str, help="Input file path.")
    ap.add_argument("--mode", required=True, choices=["u8_auto", "f32_auto", "npy_any"],
                    help="u8_auto: uint8 .npy or BigANN official. f32_auto: float32 .npy or Deep1B official. npy_any: any .npy dtype.")
    ap.add_argument("--expected-d", type=int, default=None, help="Assert expected dimension d.")
    ap.add_argument("--expected-N", type=int, default=None, help="Assert expected row count N.")
    ap.add_argument("--d-hint", type=int, default=None, help="Hint d for *vecs formats.")
    ap.add_argument("--cast-dtype", type=str, default=None, help='For npy_any: cast dtype name, e.g. "float32", "uint8".')
    ap.add_argument("--print-samples", type=int, default=8, help="How many sampled rows to print stats for.")

    ap.add_argument("--memmap", type=int, default=1, help="1=prefer memmap (default), 0=force RAM load.")
    ap.add_argument("--contig", type=int, default=0, help="1=force C-contiguous (may copy), 0=keep views (default).")

    args = ap.parse_args()

    prefer_memmap = bool(int(args.memmap))
    make_contiguous = bool(int(args.contig))

    if args.mode == "u8_auto":
        arr, N, d, fmt = load_u8_matrix_to_ram_auto(
            args.path,
            expected_N=args.expected_N,
            expected_d=args.expected_d,
            d_hint=args.d_hint,
            prefer_memmap=prefer_memmap,
            make_contiguous=make_contiguous,
        )
    elif args.mode == "f32_auto":
        arr, N, d, fmt = load_f32_matrix_to_ram_auto(
            args.path,
            expected_N=args.expected_N,
            expected_d=args.expected_d,
            d_hint=args.d_hint,
            prefer_memmap=prefer_memmap,
            make_contiguous=make_contiguous,
        )
    else:
        arr, N, d, fmt = load_any_npy(
            args.path,
            expected_N=args.expected_N,
            expected_d=args.expected_d,
            cast_dtype=args.cast_dtype,
            prefer_memmap=prefer_memmap,
            make_contiguous=True if make_contiguous else True,  # npy_any usually wants contiguous
        )

    _print_quick_stats(arr, N, d, fmt, args.path, sample_rows=args.print_samples)


if __name__ == "__main__":
    main()
