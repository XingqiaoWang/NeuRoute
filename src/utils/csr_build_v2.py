#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# IMPORTANT: import your verified loader module here.
#   - load_u8_matrix_to_ram_auto: supports .npy(uint8) or BigANN official (u8bin_header/bvecs)
#   - load_f32_matrix_to_ram_auto: supports .npy(float32) or Deep1B official (fbin_header/fvecs/.fin)
#   - load_any_npy_to_ram: supports arbitrary customer .npy (any dtype), optional cast
# ------------------------------------------------------------
from .load_any_base import (
    load_u8_matrix_to_ram_auto,
    load_f32_matrix_to_ram_auto,
    load_any_npy_to_ram,
)


# =========================
# Config (for pipelines)
# =========================

@dataclass
class CSRBuildConfig:
    # Inputs
    base_dir: str
    vectors_path: str
    d: int
    indices_u32_path: str
    offsets_path: Optional[str] = None

    # NEW: CSR kind
    # - 'bucket': build bucket-level CSR using indices_u32_path + offsets_path
    # - 'cluster': build cluster-level CSR (subCSR) using sub_ids/sub_offsets in next_out_dir
    csr_kind: str = 'bucket'

    # NEW (cluster CSR only): subCSR inputs
    sub_ids_path: Optional[str] = None        # e.g. next_out_dir/sub_ids.i64.bin
    sub_offsets_path: Optional[str] = None    # e.g. next_out_dir/sub_offsets.u64.bin
    # Optional: if you already generated sub_codes externally, provide it here to avoid gather
    sub_codes_path: Optional[str] = None      # e.g. next_out_dir/sub_codes.f32.bin

    # Only used for raw row-major binary (NOT official u8bin/fbin with headers).
    N_for_raw_vectors: Optional[int] = None

    # How to interpret vectors_path
    # - "auto": decide by file extension + header/size checks in loader
    # - "npy_any": load arbitrary customer .npy (any dtype), optionally cast (see npy_cast_dtype)
    # - "u8_auto": prefer uint8 loader (.npy uint8 or official u8 formats)
    # - "f32_auto": prefer float32 loader (.npy float32 or official f32 formats)
    # - "raw_f32bin": raw float32 row-major contiguous .bin, requires N_for_raw_vectors
    vectors_format: str = "auto"

    # For vectors_format="npy_any": optionally cast the loaded .npy to this dtype
    # Example: "float32" or "uint8". If None -> no cast.
    npy_cast_dtype: Optional[str] = None

    # Output codes dtype (for C++ runtime).
    # If using BigANN base.1B.u8bin and your runtime can consume uint8 codes, set np.uint8.
    # Otherwise, set np.float32.
    out_codes_dtype: np.dtype = np.float32

    # Performance
    pos_block: int = 2_000_000
    io_buffer_mb: int = 16

    # Derived metadata
    big_bucket_threshold: int = 100_000
    bucket_sizes_dtype: np.dtype = np.uint32
    big_bucket_ids_dtype: np.dtype = np.int32

    # Outputs (optional overrides)
    out_codes_path: Optional[str] = None
    out_ids_path: Optional[str] = None
    out_offsets_bin: Optional[str] = None
    out_bucket_sizes_bin: Optional[str] = None
    out_big_bucket_ids_bin: Optional[str] = None
    out_manifest: Optional[str] = None


# =========================
# Helpers (atomic write)
# =========================

def _atomic_replace(tmp_path: str, final_path: str) -> None:
    os.replace(tmp_path, final_path)


def _copyfile_atomic(src: str, dst: str, io_buffer_mb: int = 16) -> None:
    """Copy a file to dst atomically (tmp + replace)."""
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    tmp = dst + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    buf_bytes = 1024 * 1024 * int(io_buffer_mb)
    with open(src, 'rb') as f_in, open(tmp, 'wb', buffering=buf_bytes) as f_out:
        while True:
            chunk = f_in.read(buf_bytes)
            if not chunk:
                break
            f_out.write(chunk)
    _atomic_replace(tmp, dst)


def _write_array_atomic(path: str, arr: np.ndarray, io_buffer_mb: int = 16) -> None:
    """Write a contiguous array as raw bytes to disk atomically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    buf_bytes = 1024 * 1024 * int(io_buffer_mb)
    with open(tmp, "wb", buffering=buf_bytes) as f:
        f.write(np.ascontiguousarray(arr).view(np.uint8))
    _atomic_replace(tmp, path)


def _dtype_to_manifest_str(dt: np.dtype) -> str:
    dt = np.dtype(dt)
    mapping = {
        np.dtype(np.float32): "float32",
        np.dtype(np.uint8): "uint8",
        np.dtype(np.int8): "int8",
        np.dtype(np.uint16): "uint16",
        np.dtype(np.int16): "int16",
        np.dtype(np.uint32): "uint32",
        np.dtype(np.int32): "int32",
        np.dtype(np.uint64): "uint64",
        np.dtype(np.int64): "int64",
    }
    return mapping.get(dt, dt.name)


# =========================
# Loaders (delegated to matrix_loaders.py)
# =========================

def load_vectors_to_ram(cfg: CSRBuildConfig) -> Tuple[np.ndarray, int, int, str]:
    """
    Load base vectors fully into RAM (NO memmap), using your verified loader module.

    Returns:
      vecs: ndarray [N, d]
      N, d
      fmt: a string describing which format path was used
    """
    path = cfg.vectors_path
    d_expected = int(cfg.d)
    fmt = cfg.vectors_format.lower().strip()

    print(f"[vectors] load -> RAM: {path}")
    print(f"[vectors] vectors_format={cfg.vectors_format}")

    # 1) Arbitrary customer .npy (any dtype)
    if fmt == "npy_any":
        vecs, N, d, used = load_any_npy_to_ram(
            path,
            expected_d=d_expected,
            expected_N=cfg.N_for_raw_vectors,  # optional check
            cast_dtype=cfg.npy_cast_dtype,
        )
        return vecs, N, d, used

    # 2) Prefer uint8 loader (BigANN official / uint8 .npy)
    if fmt == "u8_auto":
        vecs, N, d, used = load_u8_matrix_to_ram_auto(
            path,
            expected_d=d_expected,
            expected_N=cfg.N_for_raw_vectors,  # optional check
            d_hint=d_expected,
        )
        return vecs, N, d, used

    # 3) Prefer float32 loader (Deep1B official / float32 .npy)
    if fmt == "f32_auto":
        vecs, N, d, used = load_f32_matrix_to_ram_auto(
            path,
            expected_d=d_expected,
            expected_N=cfg.N_for_raw_vectors,  # optional check
            d_hint=d_expected,
        )
        return vecs, N, d, used

    # 4) Raw float32 row-major contiguous .bin (NO header)
    if fmt == "raw_f32bin":
        if cfg.N_for_raw_vectors is None:
            raise ValueError("vectors_format='raw_f32bin' requires N_for_raw_vectors.")
        N = int(cfg.N_for_raw_vectors)
        d = d_expected
        expected_bytes = N * d * np.dtype(np.float32).itemsize
        actual_bytes = os.path.getsize(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"raw_f32bin size mismatch: got {actual_bytes}, expected {expected_bytes} (=N*d*4)."
            )
        vecs = np.fromfile(path, dtype=np.float32).reshape(N, d)
        vecs = np.ascontiguousarray(vecs)
        return vecs, N, d, "raw_f32bin"

    # 5) Auto: decide by extension first, then fall back.
    if fmt == "auto":
        # Most robust behavior:
        #   - .npy: treat as customer array; do NOT force dtype here.
        #   - .u8bin / .bvecs: uint8 loader
        #   - .fin / .fbin / .fvecs: float32 official loader
        #   - .bin: assume raw_f32bin (requires N_for_raw_vectors)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".npy":
            vecs, N, d, used = load_any_npy_to_ram(
                path,
                expected_d=d_expected,
                expected_N=cfg.N_for_raw_vectors,
                cast_dtype=cfg.npy_cast_dtype,  # optional
            )
            return vecs, N, d, used

        if ext in (".u8bin", ".bvecs"):
            vecs, N, d, used = load_u8_matrix_to_ram_auto(
                path,
                expected_d=d_expected,
                expected_N=cfg.N_for_raw_vectors,
                d_hint=d_expected,
            )
            return vecs, N, d, used

        if ext in (".fin", ".fbin", ".fvecs"):
            vecs, N, d, used = load_f32_matrix_to_ram_auto(
                path,
                expected_d=d_expected,
                expected_N=cfg.N_for_raw_vectors,
                d_hint=d_expected,
            )
            return vecs, N, d, used

        if ext == ".bin":
            # raw f32 bin
            raw_cfg = CSRBuildConfig(**{**asdict(cfg), "vectors_format": "raw_f32bin"})
            return load_vectors_to_ram(raw_cfg)

        raise ValueError(f"[auto] Unsupported extension: {ext} for path={path}")

    raise ValueError(f"Unsupported vectors_format={cfg.vectors_format}")


def load_offsets(offsets_path: str, dtype=np.uint64) -> np.ndarray:
    """Load offsets as uint64 1D array. Supports .npy or raw .bin (little-endian uint64)."""
    print(f"[offsets] load: {offsets_path}")
    if offsets_path.endswith(".npy"):
        off = np.load(offsets_path, allow_pickle=False)
        if off.dtype != dtype or off.ndim != 1:
            raise ValueError(f"offsets .npy must be uint64 1D, got dtype={off.dtype}, shape={off.shape}")
        return off
    off = np.fromfile(offsets_path, dtype=dtype)
    if off.ndim != 1:
        raise ValueError("offsets must be 1D")
    return off


# =========================
# CSR codes + ids builder (NO memmap)
# =========================

def build_codes_and_ids_csr_inram(cfg: CSRBuildConfig, out_codes_path: str, out_ids_i64_path: str) -> int:
    """
    Build CSR-ordered codes + ids for C++:

      codes_out[pos] = vecs[ indices_u32[pos] ]   dtype=cfg.out_codes_dtype  [N, d]
      ids_out[pos]   = int64(indices_u32[pos])    int64                     [N]

    This version reads indices into RAM via np.fromfile (NO memmap).
    """
    t0 = time.time()

    vecs, N_vec, d_vec, vec_fmt = load_vectors_to_ram(cfg)
    if d_vec != int(cfg.d):
        raise ValueError(f"vectors d mismatch: loaded d={d_vec} vs cfg.d={cfg.d}")
    print(f"[vecs] fmt={vec_fmt} shape={vecs.shape} dtype={vecs.dtype} nbytes={vecs.nbytes/1e9:.2f} GB")

    # Load indices into RAM (NO memmap)
    print(f"[indices] load -> RAM: {cfg.indices_u32_path}")
    indices_u32 = np.fromfile(cfg.indices_u32_path, dtype=np.uint32)
    if indices_u32.ndim != 1:
        raise ValueError("indices must be 1D uint32")
    if indices_u32.size != N_vec:
        raise ValueError(f"indices size mismatch: {indices_u32.size} vs vectors N={N_vec}")
    print(f"[indices] size={indices_u32.size} dtype=uint32 nbytes={indices_u32.nbytes/1e9:.2f} GB")

    N = int(N_vec)
    d = int(cfg.d)

    os.makedirs(os.path.dirname(out_codes_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_ids_i64_path) or ".", exist_ok=True)

    codes_tmp = out_codes_path + ".tmp"
    ids_tmp = out_ids_i64_path + ".tmp"
    for p in (codes_tmp, ids_tmp):
        if os.path.exists(p):
            os.remove(p)

    bytes_per_vec = np.dtype(cfg.out_codes_dtype).itemsize * d
    print(f"[out codes] {out_codes_path} dtype={_dtype_to_manifest_str(cfg.out_codes_dtype)} ~{(N*bytes_per_vec)/1e9:.2f} GB")
    print(f"[out ids]   {out_ids_i64_path} dtype=int64 ~{(N*8)/1e9:.2f} GB")

    ids_i64_buf = np.empty(cfg.pos_block, dtype=np.int64)

    written = 0
    buf_bytes = 1024 * 1024 * int(cfg.io_buffer_mb)

    with open(codes_tmp, "wb", buffering=buf_bytes) as f_codes, open(ids_tmp, "wb", buffering=buf_bytes) as f_ids:
        pbar = tqdm(total=N, desc="CSR gather+write (codes+ids)", unit="vec")

        for p0 in range(0, N, cfg.pos_block):
            p1 = min(N, p0 + cfg.pos_block)
            take = p1 - p0

            ids_u32_block = indices_u32[p0:p1]
            ids_i64_buf[:take] = ids_u32_block.astype(np.int64, copy=True)

            block = vecs[ids_i64_buf[:take]]
            if block.dtype != cfg.out_codes_dtype:
                block = block.astype(cfg.out_codes_dtype, copy=False)
            block = np.ascontiguousarray(block)

            f_codes.write(block.view(np.uint8))
            f_ids.write(np.ascontiguousarray(ids_i64_buf[:take]).view(np.uint8))

            written += take
            pbar.update(take)

        pbar.close()

    _atomic_replace(codes_tmp, out_codes_path)
    _atomic_replace(ids_tmp, out_ids_i64_path)

    dt = time.time() - t0
    print(f"[done] wrote {written} rows in {dt:.2f}s -> {written/dt:.1f} vec/s")
    return N


def build_subcodes_and_subids_csr(
    cfg: CSRBuildConfig,
    *,
    sub_ids_i64_path: str,
    out_codes_path: str,
    out_ids_i64_path: str,
) -> int:
    """
    Build *cluster-level* CSR (subCSR-aligned) codes + ids for C++ runtime.

      sub_codes_out[pos] = vecs[ sub_ids[pos] ]   dtype=cfg.out_codes_dtype  [Nsub, d]
      sub_ids_out[pos]   = int64(sub_ids[pos])    int64                     [Nsub]

    Driven by sub_ids/sub_offsets produced by bucket_cluster_pipeline.

    Notes:
      - vecs are loaded into RAM (consistent with build_codes_and_ids_csr_inram).
      - sub_ids are memory-mapped to avoid a huge ids array in RAM.
      - ids output is written by atomic copy (sub_ids already int64).
    """
    t0 = time.time()

    vecs, N_vec, d_vec, vec_fmt = load_vectors_to_ram(cfg)
    if d_vec != int(cfg.d):
        raise ValueError(f"vectors d mismatch: loaded d={d_vec} vs cfg.d={cfg.d}")
    print(f"[vecs] fmt={vec_fmt} shape={vecs.shape} dtype={vecs.dtype} nbytes={vecs.nbytes/1e9:.2f} GB")

    sz = os.path.getsize(sub_ids_i64_path)
    if sz % 8 != 0:
        raise ValueError(f"sub_ids file size not multiple of 8: {sub_ids_i64_path} size={sz}")
    Nsub = int(sz // 8)
    print(f"[sub_ids] mmap: {sub_ids_i64_path} Nsub={Nsub} (~{sz/1e9:.2f} GB)")

    sub_ids = np.memmap(sub_ids_i64_path, mode='r', dtype=np.int64, shape=(Nsub,))

    if Nsub > 0:
        samp = np.asarray(sub_ids[: min(Nsub, 200_000)], dtype=np.int64)
        mn = int(samp.min())
        mx = int(samp.max())
        if mn < 0 or mx >= int(N_vec):
            raise ValueError(f"sub_ids out of range: min={mn} max={mx} N={N_vec}")

    os.makedirs(os.path.dirname(out_codes_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_ids_i64_path) or '.', exist_ok=True)

    # ids: copy (atomic)
    _copyfile_atomic(sub_ids_i64_path, out_ids_i64_path, io_buffer_mb=cfg.io_buffer_mb)

    codes_tmp = out_codes_path + '.tmp'
    if os.path.exists(codes_tmp):
        os.remove(codes_tmp)

    d = int(cfg.d)
    bytes_per_vec = np.dtype(cfg.out_codes_dtype).itemsize * d
    print(f"[out sub_codes] {out_codes_path} dtype={_dtype_to_manifest_str(cfg.out_codes_dtype)} ~{(Nsub*bytes_per_vec)/1e9:.2f} GB")

    buf_bytes = 1024 * 1024 * int(cfg.io_buffer_mb)
    written = 0

    with open(codes_tmp, 'wb', buffering=buf_bytes) as f_codes:
        pbar = tqdm(total=int(Nsub), desc='subCSR gather+write (sub_codes)', unit='vec')

        for p0 in range(0, int(Nsub), int(cfg.pos_block)):
            p1 = min(int(Nsub), p0 + int(cfg.pos_block))
            take = p1 - p0

            ids_block = np.asarray(sub_ids[p0:p1], dtype=np.int64)
            block = vecs[ids_block]
            if block.dtype != cfg.out_codes_dtype:
                block = block.astype(cfg.out_codes_dtype, copy=False)
            block = np.ascontiguousarray(block)

            f_codes.write(block.view(np.uint8))
            written += take
            pbar.update(take)

        pbar.close()

    _atomic_replace(codes_tmp, out_codes_path)

    dt = time.time() - t0
    print(f"[done] wrote {written} rows in {dt:.2f}s -> {written/dt:.1f} vec/s")
    return int(Nsub)


# =========================
# Offsets -> derived metadata
# =========================

def compute_bucket_sizes_from_offsets(offsets_u64: np.ndarray) -> np.ndarray:
    if offsets_u64.dtype != np.uint64 or offsets_u64.ndim != 1 or offsets_u64.size < 2:
        raise ValueError("offsets must be uint64 1D with len>=2")
    return offsets_u64[1:] - offsets_u64[:-1]


def write_offsets_raw_bin(offsets_u64: np.ndarray, out_offsets_bin_path: str, io_buffer_mb: int = 16) -> None:
    print(f"[offsets] write raw u64 bin: {out_offsets_bin_path} (len={offsets_u64.size})")
    _write_array_atomic(out_offsets_bin_path, offsets_u64, io_buffer_mb)
    print(f"[offsets] done: {out_offsets_bin_path}")


def write_bucket_sizes_bin(bucket_sizes_u64: np.ndarray, out_path: str, dtype: np.dtype, io_buffer_mb: int = 16) -> None:
    dtype = np.dtype(dtype)
    if dtype not in (np.uint32, np.uint64):
        raise ValueError("bucket_sizes_dtype must be uint32 or uint64")

    max_sz = int(bucket_sizes_u64.max(initial=0))
    if dtype == np.uint32 and max_sz > np.iinfo(np.uint32).max:
        raise ValueError(f"bucket_sizes max={max_sz} exceeds uint32 range")

    out_arr = bucket_sizes_u64.astype(dtype, copy=False)
    print(f"[bucket_sizes] write: {out_path} dtype={out_arr.dtype} nlist={out_arr.size} max={max_sz}")
    _write_array_atomic(out_path, out_arr, io_buffer_mb)
    print(f"[bucket_sizes] done: {out_path}")


def write_big_bucket_ids_bin(bucket_sizes_u64: np.ndarray, threshold: int, out_path: str, dtype: np.dtype, io_buffer_mb: int = 16) -> None:
    dtype = np.dtype(dtype)
    if dtype not in (np.int32, np.int64):
        raise ValueError("big_bucket_ids_dtype must be int32 or int64")
    if threshold < 0:
        raise ValueError("big_bucket_threshold must be >= 0")

    big_ids = np.nonzero(bucket_sizes_u64 >= np.uint64(threshold))[0].astype(np.int64, copy=False)
    big_ids.sort()

    if dtype == np.int32 and big_ids.size > 0 and int(big_ids.max()) > np.iinfo(np.int32).max:
        raise ValueError("big_bucket_ids exceed int32 range")

    out_arr = big_ids.astype(dtype, copy=False)
    print(f"[big_bucket_ids] threshold>={threshold} count={out_arr.size}/{bucket_sizes_u64.size} -> {out_path} dtype={out_arr.dtype}")
    _write_array_atomic(out_path, out_arr, io_buffer_mb)
    print(f"[big_bucket_ids] done: {out_path}")


# =========================
# Manifest
# =========================

def write_manifest(
    out_manifest_path: str,
    *,
    csr_kind: str = "bucket",
    d: int,
    N: int,
    nlist: Optional[int],
    codes_path: str,
    codes_dtype: np.dtype,
    ids_path: str,
    offsets_path: Optional[str],
    bucket_sizes_path: Optional[str],
    bucket_sizes_dtype: Optional[str],
    big_bucket_ids_path: Optional[str],
    big_bucket_ids_dtype: Optional[str],
    big_threshold: Optional[int],
    # Optional mapping artifacts for subCSR (bucket<->cluster + centroids)
    sidecars: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    manifest = {
        "format": "csr_inverted_lists_v1",
        "csr_kind": str(csr_kind),
        "endianness": "little",
        "d": int(d),
        "N": int(N),
        "nlist": (int(nlist) if nlist is not None else None),
        "codes": {
            "path": codes_path,
            "dtype": _dtype_to_manifest_str(codes_dtype),
            "layout": "row_major",
            "shape": [int(N), int(d)],
        },
        "ids": {"path": ids_path, "dtype": "int64", "layout": "row_major", "shape": [int(N)]},
        "offsets": (
            {"path": offsets_path, "dtype": "uint64", "shape": [int(nlist + 1)]}
            if offsets_path and nlist is not None else None
        ),
        "bucket_sizes": (
            {"path": bucket_sizes_path, "dtype": bucket_sizes_dtype, "shape": [int(nlist)]}
            if bucket_sizes_path and bucket_sizes_dtype and nlist is not None else None
        ),
        "big_buckets": (
            {"path": big_bucket_ids_path, "dtype": big_bucket_ids_dtype, "threshold_ge": int(big_threshold)}
            if big_bucket_ids_path and big_bucket_ids_dtype and big_threshold is not None else None
        ),
        "sidecars": sidecars or None,
    }

    os.makedirs(os.path.dirname(out_manifest_path) or ".", exist_ok=True)
    tmp = out_manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    _atomic_replace(tmp, out_manifest_path)
    print(f"[manifest] wrote: {out_manifest_path}")


# =========================
# High-level pipeline API
# =========================

def build_csr_artifacts(cfg: CSRBuildConfig) -> Dict[str, Any]:
    """Build CSR artifacts.

    Supports two modes via cfg.csr_kind:
      - bucket (default): bucket-CSR using indices_u32_path + offsets_path
      - cluster: cluster-CSR (subCSR) using sub_ids_path + sub_offsets_path

    For cluster mode, required:
      - cfg.sub_ids_path (int64 list, length Nsub)
      - cfg.sub_offsets_path (uint64 offsets, length n_sub+1)
    Optional:
      - cfg.sub_codes_path: if provided, sub_codes will be copied instead of gathered.

    Output naming:
      - bucket: codes.*.csr.bin / ids.int64.csr.bin / offsets.uint64.bin / bucket_sizes.bin / big_bucket_ids.bin
      - cluster: sub_codes.*.csr.bin / sub_ids.int64.csr.bin / sub_offsets.uint64.bin / sub_sizes.bin / big_sub_ids.bin
    """
    base_dir = cfg.base_dir
    os.makedirs(base_dir, exist_ok=True)

    csr_kind = (cfg.csr_kind or "bucket").lower().strip()
    is_cluster = (csr_kind == "cluster")

    # output paths
    if is_cluster:
        out_codes_path = cfg.out_codes_path or os.path.join(
            base_dir, f"sub_codes.{_dtype_to_manifest_str(cfg.out_codes_dtype)}.csr.bin"
        )
        out_ids_path = cfg.out_ids_path or os.path.join(base_dir, "sub_ids.int64.csr.bin")
        out_offsets_bin = cfg.out_offsets_bin or os.path.join(base_dir, "sub_offsets.uint64.bin")
        out_bucket_sizes_bin = cfg.out_bucket_sizes_bin or os.path.join(base_dir, "sub_sizes.bin")
        out_big_bucket_ids_bin = cfg.out_big_bucket_ids_bin or os.path.join(base_dir, "big_sub_ids.bin")
        out_manifest = cfg.out_manifest or os.path.join(base_dir, "subcsr_manifest.json")
    else:
        out_codes_path = cfg.out_codes_path or os.path.join(
            base_dir, f"codes.{_dtype_to_manifest_str(cfg.out_codes_dtype)}.csr.bin"
        )
        out_ids_path = cfg.out_ids_path or os.path.join(base_dir, "ids.int64.csr.bin")
        out_offsets_bin = cfg.out_offsets_bin or os.path.join(base_dir, "offsets.uint64.bin")
        out_bucket_sizes_bin = cfg.out_bucket_sizes_bin or os.path.join(base_dir, "bucket_sizes.bin")
        out_big_bucket_ids_bin = cfg.out_big_bucket_ids_bin or os.path.join(base_dir, "big_bucket_ids.bin")
        out_manifest = cfg.out_manifest or os.path.join(base_dir, "csr_manifest.json")

    # 1) CSR codes + ids
    if is_cluster:
        if not cfg.sub_ids_path or not os.path.exists(cfg.sub_ids_path):
            raise FileNotFoundError(f"[subcsr] missing sub_ids_path: {cfg.sub_ids_path}")
        if not cfg.sub_offsets_path or not os.path.exists(cfg.sub_offsets_path):
            raise FileNotFoundError(f"[subcsr] missing sub_offsets_path: {cfg.sub_offsets_path}")

        # ids: copy sub_ids to output ids path
        _copyfile_atomic(cfg.sub_ids_path, out_ids_path, io_buffer_mb=cfg.io_buffer_mb)

        # codes
        if cfg.sub_codes_path and os.path.exists(cfg.sub_codes_path):
            print(f"[subcsr] copy prebuilt sub_codes: {cfg.sub_codes_path} -> {out_codes_path}")
            _copyfile_atomic(cfg.sub_codes_path, out_codes_path, io_buffer_mb=cfg.io_buffer_mb)
            offsets_u64 = load_offsets(cfg.sub_offsets_path, dtype=np.uint64)
            N_final = int(offsets_u64[-1])
        else:
            N_final = build_subcodes_and_subids_csr(
                cfg,
                sub_ids_i64_path=cfg.sub_ids_path,
                out_codes_path=out_codes_path,
                out_ids_i64_path=out_ids_path,
            )

    else:
        N_final = build_codes_and_ids_csr_inram(cfg, out_codes_path=out_codes_path, out_ids_i64_path=out_ids_path)

    # 2) offsets + derived metadata
    offsets_bin_for_manifest = None
    bucket_sizes_for_manifest = None
    big_ids_for_manifest = None
    nlist = None

    if is_cluster:
        offsets_u64 = load_offsets(cfg.sub_offsets_path, dtype=np.uint64)
        nlist = int(offsets_u64.size - 1)  # n_sub

        if int(offsets_u64[-1]) != int(N_final):
            print(f"[warn][subcsr] sub_offsets[-1]={int(offsets_u64[-1])} != N={int(N_final)} (check inputs)")

        write_offsets_raw_bin(offsets_u64, out_offsets_bin, io_buffer_mb=cfg.io_buffer_mb)
        offsets_bin_for_manifest = out_offsets_bin

        sizes_u64 = compute_bucket_sizes_from_offsets(offsets_u64)
        write_bucket_sizes_bin(sizes_u64, out_bucket_sizes_bin, cfg.bucket_sizes_dtype, io_buffer_mb=cfg.io_buffer_mb)
        bucket_sizes_for_manifest = out_bucket_sizes_bin

        # big bucket ids are optional; allow disabling in "cluster-only" / minimal pipelines.
        big_ids_for_manifest = None
        if out_big_bucket_ids_bin is not None and cfg.big_bucket_threshold > 0 and cfg.big_bucket_ids_dtype is not None:
            write_big_bucket_ids_bin(
                sizes_u64,
                cfg.big_bucket_threshold,
                out_big_bucket_ids_bin,
                cfg.big_bucket_ids_dtype,
                io_buffer_mb=cfg.io_buffer_mb,
            )
            big_ids_for_manifest = out_big_bucket_ids_bin


    else:
        if cfg.offsets_path is not None and os.path.exists(cfg.offsets_path):
            offsets_u64 = load_offsets(cfg.offsets_path, dtype=np.uint64)
            nlist = int(offsets_u64.size - 1)

            if int(offsets_u64[-1]) != int(N_final):
                print(f"[warn] offsets[-1]={int(offsets_u64[-1])} != N={int(N_final)} (check inputs)")

            write_offsets_raw_bin(offsets_u64, out_offsets_bin, io_buffer_mb=cfg.io_buffer_mb)
            offsets_bin_for_manifest = out_offsets_bin

            bucket_sizes_u64 = compute_bucket_sizes_from_offsets(offsets_u64)
            write_bucket_sizes_bin(bucket_sizes_u64, out_bucket_sizes_bin, cfg.bucket_sizes_dtype, io_buffer_mb=cfg.io_buffer_mb)
            bucket_sizes_for_manifest = out_bucket_sizes_bin

            # big bucket ids are optional; allow disabling by setting threshold<=0 or out_big_bucket_ids_bin=None
            big_ids_for_manifest = None
            if out_big_bucket_ids_bin is not None and cfg.big_bucket_threshold > 0 and cfg.big_bucket_ids_dtype is not None:
                write_big_bucket_ids_bin(
                    bucket_sizes_u64,
                    cfg.big_bucket_threshold,
                    out_big_bucket_ids_bin,
                    cfg.big_bucket_ids_dtype,
                    io_buffer_mb=cfg.io_buffer_mb,
                )
                big_ids_for_manifest = out_big_bucket_ids_bin

        else:
            print("[offsets] skipped (offsets_path missing) -> bucket_sizes/big_bucket_ids will NOT be produced")

    # 3) manifest
    # Optional: include subCSR sidecars (if present) for bucket<->cluster mapping.
    sidecars_for_manifest = None
    if is_cluster:
        cand = {
            # cluster -> bucket
            "sub_bucket_id": (os.path.join(base_dir, "sub_bucket_id.u32.bin"), "uint32"),
            # bucket -> cluster offset range (length = nbuckets+1)
            "bucket_sub_offsets": (os.path.join(base_dir, "bucket_sub_offsets.u64.bin"), "uint64"),
            # per-cluster centroid (layout: [nsub, d])
            "sub_centroids": (os.path.join(base_dir, "sub_centroids.f32.bin"), "float32"),
            # per-vector local cluster label (optional)
            "sub_cluster_id": (os.path.join(base_dir, "sub_cluster_id.u16.bin"), "uint16"),
        }
        sidecars = {}
        for k, (p, dt) in cand.items():
            if os.path.isfile(p):
                sidecars[k] = {"path": p, "dtype": dt}
        if sidecars:
            sidecars_for_manifest = sidecars

    write_manifest(
        out_manifest,
        csr_kind=("cluster" if is_cluster else "bucket"),
        d=cfg.d,
        N=int(N_final),
        nlist=nlist,
        codes_path=out_codes_path,
        codes_dtype=cfg.out_codes_dtype,
        ids_path=out_ids_path,
        offsets_path=offsets_bin_for_manifest,
        bucket_sizes_path=bucket_sizes_for_manifest,
        bucket_sizes_dtype=("uint32" if cfg.bucket_sizes_dtype == np.uint32 else "uint64") if bucket_sizes_for_manifest else None,
        big_bucket_ids_path=big_ids_for_manifest,
        big_bucket_ids_dtype=("int32" if cfg.big_bucket_ids_dtype == np.int32 else "int64") if big_ids_for_manifest else None,
        big_threshold=cfg.big_bucket_threshold if big_ids_for_manifest else None,
        sidecars=sidecars_for_manifest,
    )

    return {
        "config": asdict(cfg),
        "csr_kind": ("cluster" if is_cluster else "bucket"),
        "N": int(N_final),
        "nlist": (int(nlist) if nlist is not None else None),
        "codes_path": out_codes_path,
        "ids_path": out_ids_path,
        "offsets_bin_path": offsets_bin_for_manifest,
        "bucket_sizes_path": bucket_sizes_for_manifest,
        "big_bucket_ids_path": big_ids_for_manifest,
        "manifest_path": out_manifest,
    }



# =========================
# Demo main (safe for import)
# =========================

def main() -> None:
    # Example: BigANN official base.1B.u8bin (128 dim)
    cfg = CSRBuildConfig(
        base_dir="/dev/shm/csr_out_bigann_u8",
        vectors_path="/path/to/big-ann-benchmarks/data/bigann/base.1B.u8bin",
        vectors_format="u8_auto",        # or "auto"
        out_codes_dtype=np.uint8,        # write uint8 CSR codes
        d=128,                           # BigANN base is 128
        indices_u32_path="/dev/shm/csr_inputs_best/indices.uint32.bin",
        offsets_path="/dev/shm/csr_inputs_best/offsets.uint64.npy",
        N_for_raw_vectors=None,
        pos_block=2_000_000,
        io_buffer_mb=16,
    )

    out = build_csr_artifacts(cfg)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
