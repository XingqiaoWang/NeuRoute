#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PATCHED_V5_ALIGNED3: emits D_in/sub_centroids_d/sub_codes_d/cap_per_bin; removes sweep_max_score and avoids csr_d/D_code ambiguity
# PATCHED_V5_UPGRADED: fix scope/NameError; derive nlist inside cmd_make; always write cfg

from __future__ import annotations

PATCH_ID = 'STD_ALIAS_EXPLICIT_PATHS_VERIFIED_20260218_1'

"""
make_pipeline_cfg_from_experiment_json.py

Subcommands
-----------
1) make:
   - Read experiment JSON
   - Stage CSR files into shm (preserve basenames; NO renaming)
   - Stage query into shm as RAW PAYLOAD (no header) because pipeline doesn't support headers
     * supports .u8bin/.fbin/.npy/.bin (raw) by flags
   - Write threshold vector binary (thr_f32_path) into shm
   - Stage early-stop params json (earlystop_json) into shm if present in experiment
   - If a0/b not provided by CLI, read them from earlystop_json
   - Emit config JSON using the EXACT key names parsed by
     bench_full_dp_querybins_pipeline_real_v6_3_torchcfg.cpp load_cfg_from_json()

2) strip_u8bin:
3) strip_fbin:

Notes
-----
- u8bin header auto-detect:
    * 8-byte header (n,d) or (d,n) validated by file size
    * OR 4-byte header (d) with n inferred from file size
- fbin: 8-byte header (n,d), float32 payload
- npy: [nq,d], converted to raw row-major (float32 default)
"""

import os, shutil
import re

def _ensure_link_or_copy(src_path: str, dst_path: str, verbose: bool = False):
    """Ensure dst_path exists and refers to src_path (prefer symlink; fallback to copy)."""
    
    if os.path.exists(dst_path):
        return
    try:
        src_dir = os.path.dirname(os.path.abspath(src_path))
        dst_dir = os.path.dirname(os.path.abspath(dst_path))
        if src_dir == dst_dir:
            os.symlink(os.path.basename(src_path), dst_path)
        else:
            os.symlink(src_path, dst_path)
        if verbose:
            print(f"[link] {dst_path} -> {src_path}")
    except Exception:
        try:
            shutil.copy2(src_path, dst_path)
            if verbose:
                print(f"[copy] {src_path} -> {dst_path}")
        except Exception as e:
            print(f"[warn] failed to link/copy {dst_path}: {e}")

def _derive_nlist_from_bucket_sub_offsets(bucket_sub_offsets_u64: str) -> int:
    """Return nlist inferred from bucket_sub_offsets.u64.bin (len = nlist+1, dtype u64)."""
    import os
    b = os.path.getsize(bucket_sub_offsets_u64)
    if b <= 0 or (b % 8) != 0:
        raise RuntimeError(f"bad bucket_sub_offsets size: {bucket_sub_offsets_u64} bytes={b} (expect multiple of 8)")
    u64_len = b // 8
    if u64_len < 2:
        raise RuntimeError(f"bucket_sub_offsets too short: len={u64_len}")
    return int(u64_len - 1)

def _read_encoding_dim_from_exp_config(exp_config_path: str):
    """Read experiment config JSON and return encoding_dim if present, else None."""
    import json, os
    if not exp_config_path or not os.path.isfile(exp_config_path):
        return None
    try:
        with open(exp_config_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        # expected: {"config": {"encoding_dim": 22, ...}, ...}
        cfg = js.get("config", js)
        d = cfg.get("encoding_dim", None)
        if d is None:
            # sometimes nested differently
            d = js.get("encoding_dim", None)
        return int(d) if d is not None else None
    except Exception:
        return None


import argparse
import json
import shutil
import struct
import sys
from pathlib import Path
import pathlib
from typing import Any, Dict, Optional, Tuple


def _ensure_sub_ids_flat(subcsr_dir_local: str, verbose: bool = True) -> None:
    """Ensure /subcsr/sub_ids.i64.bin exists (bench legacy name).
    If only CSR-named ids exist (e.g., sub_ids.i64.csr.bin or ids_i64.csr.bin), create/copy to sub_ids.i64.bin.
    """
    cand = [
        "sub_ids.i64.bin",
        "sub_ids.i64.csr.bin",
        "ids_i64.csr.bin",
        "ids.i64.csr.bin",
        "ids.i64.bin",
    ]
    p = pathlib.Path(subcsr_dir_local)
    dst = p / "sub_ids.i64.bin"
    if dst.is_file():
        return
    src = None
    for name in cand[1:]:
        if (p / name).is_file():
            src = p / name
            break
    if src is None:
        # nothing we can do; leave as-is
        if verbose:
            print("[warn] sub_ids not found in subcsr dir; expected one of:", cand)
        return
    try:
        import shutil
        shutil.copy2(src, dst)
        if verbose:
            print(f"[subcsr][alias] {src.name} -> {dst.name} (bench legacy)")
    except Exception as e:
        print(f"[warn] failed to alias sub_ids: {src} -> {dst}: {e}")



def _parse_D_in_from_metric(metric: str) -> Optional[int]:
    """Extract trailing dimension from metric like 'l2_u8_128' or 'ip_f32_768'."""
    if not metric:
        return None
    m = re.search(r"(?:_|:)(\d+)$", metric)
    if not m:
        m = re.search(r"(\d+)$", metric)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None



# =============================
# Newer CSR+SubCSR staging notes (v3_with_subcsr)
# =============================
# Many recent runs produce a directory like:
#   results_xxx/
#     offsets.uint64.bin
#     indices.uint32.bin            (may be referenced via csr_manifest.json)
#     ids.int64.csr.bin
#     bucket_sizes.bin              (uint64 payload; sometimes without suffix)
#     subcsr_manifest.json          (points to subcsr artifacts)
#     sub_offsets.u64.bin        (sometimes in root)
#     sub_ids.i64.csr.bin         (sometimes in root)
#     sub_codes.u8.csr.bin|sub_codes.f32.csr.bin
#     ...
# And subcsr artifacts may be stored elsewhere and referenced by subcsr_manifest.json.
# The C++ runtime header csr_index_runtime_v3_with_subcsr.h expects a base dir:
#   <csr_base_dir>/offsets.uint64.bin
#   <csr_base_dir>/indices.uint32.bin
#   <csr_base_dir>/ids.int64.csr.bin
#   <csr_base_dir>/bucket_sizes.uint64.bin
# plus a subdir:
#   <csr_base_dir>/subcsr/
#     sub_offsets.u64.bin
#     sub_ids.i64.bin
#     sub_bucket_id.u32.bin
#     sub_cluster_id.u16.bin
#     sub_centroids.f32.bin
#     bucket_sub_offsets.u64.bin
# Optionally sub_codes is provided separately via config key sub_codes_path.


# -----------------------------
# utils
# -----------------------------
def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def die(msg: str, code: int = 2):
    eprint(f"[FATAL] {msg}")
    raise SystemExit(code)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _filesize(p: Path) -> int:
    return int(p.stat().st_size)


def copy_file_same_name(src: Path, dst_dir: Path, verbose: bool = True) -> Path:
    dst = dst_dir / src.name
    if verbose:
        eprint(f"[INFO] cp {src} -> {dst}")
    shutil.copy2(src, dst)
    return dst


def copy_file_as(src: Path, dst: Path, verbose: bool = True) -> Path:
    ensure_dir(dst.parent)
    if verbose:
        eprint(f"[INFO] cp {src} -> {dst}")
    shutil.copy2(src, dst)
    return dst


def _load_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.is_file():
            with open(p, "r") as f:
                return json.load(f)
    except Exception as e:
        eprint(f"[WARN] failed to read json: {p}: {e}")
    return None


def _find_str_paths_in_json(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_find_str_paths_in_json(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find_str_paths_in_json(v))
    return out


def _pick_path(paths: list[str], *, must_contain: str) -> Optional[Path]:
    mc = must_contain.lower()
    for s in paths:
        if not isinstance(s, str):
            continue
        if mc in s.lower():
            p = Path(s).expanduser()
            if p.is_file():
                return p.resolve()
    return None


def _infer_nlist_from_u64_offsets(path: Path) -> Optional[int]:
    """Infer nlist from an offsets uint64 file: len = nlist+1."""
    try:
        sz = _filesize(path)
        if sz % 8 != 0:
            return None
        n = sz // 8
        if n <= 0:
            return None
        return int(n - 1)
    except Exception:
        return None


def _extract_manifest_meta(man: Optional[Dict[str, Any]]) -> Dict[str, Optional[int]]:
    """Extract d/N/nlist from a subcsr/csr manifest if present."""
    out: Dict[str, Optional[int]] = {"d": None, "N": None, "nlist": None}
    if not isinstance(man, dict):
        return out
    for k in ("d", "N", "nlist"):
        v = man.get(k, None)
        if v is not None:
            try:
                out[k] = int(v)
            except Exception:
                pass
    # fallback: offsets.shape
    if out["nlist"] is None:
        offs = man.get("offsets", None)
        if isinstance(offs, dict):
            shp = offs.get("shape", None)
            if isinstance(shp, list) and len(shp) >= 1:
                try:
                    out["nlist"] = int(shp[0]) - 1
                except Exception:
                    pass
    # fallback: codes.shape
    if out["N"] is None:
        codes = man.get("codes", None)
        if isinstance(codes, dict):
            shp = codes.get("shape", None)
            if isinstance(shp, list) and len(shp) >= 1:
                try:
                    out["N"] = int(shp[0])
                except Exception:
                    pass
    return out


def stage_csr_index_v3_with_subcsr(
    results_dir: Path,
    shm_dir: Path,
    *,
    require_main_csr: bool = True,
    verbose: bool = True,
) -> Dict[str, Optional[Path]]:
    """Stage CSRIndexRuntime v3_with_subcsr layout into shm_dir.

    If require_main_csr=False, we will stage **only** SubCSR artifacts (cluster-only runs)
    and will NOT error if main CSR (offsets/indices/ids/bucket_sizes) is missing.
    """
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        die(f"--csr_results_dir not a directory: {results_dir}")

    ensure_dir(shm_dir)
    subcsr_out = shm_dir / "subcsr"
    ensure_dir(subcsr_out)

    csr_manifest = _load_json_if_exists(results_dir / "csr_manifest.json")
    subcsr_manifest = _load_json_if_exists(results_dir / "subcsr_manifest.json")
    csr_paths = _find_str_paths_in_json(csr_manifest) if csr_manifest else []
    subcsr_paths = _find_str_paths_in_json(subcsr_manifest) if subcsr_manifest else []

    # ---- main CSR ----
    src_offsets: Optional[Path] = None
    src_ids: Optional[Path] = None
    src_indices: Optional[Path] = None
    src_bucket_sizes: Optional[Path] = None

    if require_main_csr:
        src_offsets = (results_dir / "offsets.uint64.bin")
        if not src_offsets.is_file():
            src_offsets = _pick_path(csr_paths, must_contain="offsets")
        if src_offsets is None or not src_offsets.is_file():
            die(f"Missing offsets file (expected offsets.uint64.bin or manifest path) under {results_dir}")

        src_ids = (results_dir / "ids.int64.csr.bin")
        if not src_ids.is_file():
            src_ids = _pick_path(csr_paths, must_contain="ids.int64.csr")
        if src_ids is None or not src_ids.is_file():
            die(f"Missing ids csr file (expected ids.int64.csr.bin or manifest path) under {results_dir}")

        src_indices = (results_dir / "indices.uint32.bin")
        if not src_indices.is_file():
            src_indices = _pick_path(csr_paths, must_contain="indices.uint32")
        if src_indices is None or not src_indices.is_file():
            die(f"Missing indices file (expected indices.uint32.bin or manifest path) under {results_dir}")

        src_bucket_sizes = (results_dir / "bucket_sizes.uint64.bin")
        if not src_bucket_sizes.is_file():
            alt = results_dir / "bucket_sizes.bin"
            if alt.is_file():
                src_bucket_sizes = alt
        if src_bucket_sizes is None or not src_bucket_sizes.is_file():
            src_bucket_sizes = _pick_path(csr_paths, must_contain="bucket_sizes")
        if src_bucket_sizes is None or not src_bucket_sizes.is_file():
            die(f"Missing bucket_sizes file under {results_dir}")

        copy_file_as(src_offsets, shm_dir / "offsets.uint64.bin", verbose=verbose)
        copy_file_as(src_indices, shm_dir / "indices.uint32.bin", verbose=verbose)
        copy_file_as(src_ids, shm_dir / "ids.int64.csr.bin", verbose=verbose)
        copy_file_as(src_bucket_sizes, shm_dir / "bucket_sizes.uint64.bin", verbose=verbose)

    # ---- SubCSR ----
    def find_sub(name: str, contains: str) -> Path:
        p = results_dir / "subcsr" / name
        if p.is_file():
            return p
        p2 = results_dir / name
        if p2.is_file():
            return p2
        mp = _pick_path(subcsr_paths, must_contain=contains)
        if mp is not None and mp.is_file():
            return mp
        die(f"Missing subcsr file: want {name} (contains='{contains}')")

    # tolerate older naming variants
    src_sub_offsets = (results_dir / "subcsr" / "sub_offsets.u64.bin")
    if not src_sub_offsets.is_file():
        alt = results_dir / "sub_offsets.u64.bin"
        if alt.is_file():
            src_sub_offsets = alt
        else:
            src_sub_offsets = find_sub("sub_offsets.u64.bin", "sub_offsets")

    src_sub_ids = (results_dir / "subcsr" / "sub_ids.i64.csr.bin")
    if not src_sub_ids.is_file():
        alt = results_dir / "sub_ids.i64.csr.bin"
        if alt.is_file():
            src_sub_ids = alt
        else:
            src_sub_ids = find_sub("sub_ids.i64.csr.bin", "sub_ids")

    src_sub_bucket_id = find_sub("sub_bucket_id.u32.bin", "sub_bucket_id")
    src_sub_cluster_id = find_sub("sub_cluster_id.u16.bin", "sub_cluster_id")
    src_sub_centroids = find_sub("sub_centroids.f32.bin", "sub_centroids")
    src_bucket_sub_offsets = find_sub("bucket_sub_offsets.u64.bin", "bucket_sub_offsets")

    copy_file_as(src_sub_offsets, subcsr_out / "sub_offsets.u64.bin", verbose=verbose)
    copy_file_as(src_sub_ids, subcsr_out / "sub_ids.i64.csr.bin", verbose=verbose)
    copy_file_as(src_sub_bucket_id, subcsr_out / "sub_bucket_id.u32.bin", verbose=verbose)
    copy_file_as(src_sub_cluster_id, subcsr_out / "sub_cluster_id.u16.bin", verbose=verbose)
    copy_file_as(src_sub_centroids, subcsr_out / "sub_centroids.f32.bin", verbose=verbose)
    copy_file_as(src_bucket_sub_offsets, subcsr_out / "bucket_sub_offsets.u64.bin", verbose=verbose)

    # ---- bench/runtime compatibility: provide both short and long dtype suffixes, and optional non-CSR alias ----
    try:
        # offsets: short name is primary for bench; also provide long-name alias
        off_short = str(subcsr_out / "sub_offsets.u64.bin")
        off_long  = str(subcsr_out / "sub_offsets.uint64.bin")
        if os.path.exists(off_short):
            _ensure_link_or_copy(off_short, off_long, verbose=verbose)

        # ids: CSR form is primary; provide both i64/int64 CSR names, and a legacy non-CSR alias if needed
        ids_csr_short = str(subcsr_out / "sub_ids.i64.csr.bin")
        ids_csr_long  = str(subcsr_out / "sub_ids.int64.csr.bin")
        if os.path.exists(ids_csr_short):
            _ensure_link_or_copy(ids_csr_short, ids_csr_long, verbose=verbose)
            _ensure_link_or_copy(ids_csr_short, str(subcsr_out / "sub_ids.i64.bin"), verbose=verbose)
    except Exception as e:
        print(f"[warn] compat links/copies failed: {e}")

    # optional: sub_codes
    staged_sub_codes: Optional[Path] = None
    sub_codes: Optional[Path] = None
    for cand in [
        results_dir / "sub_codes.u8.csr.bin",
        results_dir / "sub_codes.f32.csr.bin",
        results_dir / "subcsr" / "sub_codes.u8.bin",
        results_dir / "subcsr" / "sub_codes.f32.bin",
    ]:
        if cand.is_file():
            sub_codes = cand
            break
    if sub_codes is None:
        sub_codes = _pick_path(subcsr_paths, must_contain="sub_codes")

    if sub_codes is not None and sub_codes.is_file():
        if "u8" in sub_codes.name.lower():
            staged_sub_codes = copy_file_as(sub_codes, subcsr_out / "sub_codes.u8.bin", verbose=verbose)
        else:
            staged_sub_codes = copy_file_as(sub_codes, subcsr_out / "sub_codes.f32.bin", verbose=verbose)

    return {
        "csr_base_dir": shm_dir,
        "subcsr_dir": subcsr_out,
        "sub_codes": staged_sub_codes,
        "csr_manifest": (results_dir / "csr_manifest.json") if (results_dir / "csr_manifest.json").is_file() else None,
        "subcsr_manifest": (results_dir / "subcsr_manifest.json") if (results_dir / "subcsr_manifest.json").is_file() else None,
    }


def stage_subcsr_only_from_dir(results_dir: Path, shm_dir: Path, verbose: bool = True) -> Dict[str, Optional[Path]]:
    """
    Stage *only* SubCSR artifacts into shm_dir/subcsr.

    Use when your results_dir contains only sub_* files (no main CSR offsets/ids/indices/bucket_sizes).
    This matches csr_index_runtime_v3_with_subcsr.h expectations:

      meta_dir:
        sub_bucket_id.u32.bin
        sub_cluster_id.u16.bin (or .u32.bin if configured)
        sub_centroids.f32.bin
        bucket_sub_offsets.u64.bin

      cluster_csr_dir:
        sub_offsets.u64.bin
        sub_ids.i64.csr.bin

    Optional dense codes (sub_codes_path) are NOT staged here, because the CSR-packed
    sub_codes.*.csr.bin is NOT the same as a dense [N,D] codes matrix.
    """
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        die(f"--csr_results_dir not a directory: {results_dir}")

    ensure_dir(shm_dir)
    subcsr_out = shm_dir / "subcsr"
    ensure_dir(subcsr_out)

    # Helper: look in root first, then in results_dir/subcsr/
    def pick(name: str) -> Path:
        p = results_dir / name
        if p.is_file():
            return p
        p2 = results_dir / "subcsr" / name
        if p2.is_file():
            return p2
        die(f"Missing required subcsr file: {name} (looked in {results_dir} and {results_dir/'subcsr'})")

    # required
    src_sub_offsets = pick("sub_offsets.u64.bin")
    src_sub_ids     = pick("sub_ids.i64.csr.bin")
    src_sub_bucket  = pick("sub_bucket_id.u32.bin")
    src_sub_cent    = pick("sub_centroids.f32.bin")
    src_bucket_sub  = pick("bucket_sub_offsets.u64.bin")

    # cluster id can be u16 or u32 depending on build; prefer u16 if present
    src_sub_cluster = None
    for cand in ["sub_cluster_id.u16.bin", "sub_cluster_id.u32.bin"]:
        p = (results_dir / cand)
        if not p.is_file():
            p = (results_dir / "subcsr" / cand)
        if p.is_file():
            src_sub_cluster = p
            break
    if src_sub_cluster is None:
        die("Missing required subcsr file: sub_cluster_id.u16.bin (or sub_cluster_id.u32.bin)")

    # copy with SAME basenames
    copy_file_as(src_sub_offsets, subcsr_out / "sub_offsets.u64.bin", verbose=verbose)
    copy_file_as(src_sub_ids,     subcsr_out / "sub_ids.i64.csr.bin", verbose=verbose)
    copy_file_as(src_sub_bucket,  subcsr_out / "sub_bucket_id.u32.bin", verbose=verbose)
    copy_file_as(src_sub_cluster, subcsr_out / src_sub_cluster.name, verbose=verbose)
    copy_file_as(src_sub_cent,    subcsr_out / "sub_centroids.f32.bin", verbose=verbose)
    copy_file_as(src_bucket_sub,  subcsr_out / "bucket_sub_offsets.u64.bin", verbose=verbose)

    return {"csr_base_dir": shm_dir, "subcsr_dir": subcsr_out, "sub_codes": None}


def stage_cluster_manifest_v1(manifest_json: Path, shm_dir: Path, *, verbose: bool = True) -> Dict[str, Optional[Path]]:
    """Stage *cluster* CSR manifest (csr_inverted_lists_v1, csr_kind=cluster) into shm_dir/subcsr.

    The bench_cluster_only runtime expects a *subcsr_dir* containing:
      - sub_offsets.u64.bin
      - sub_ids.i64.bin
      - sub_bucket_id.u32.bin
      - sub_cluster_id.(u16|u32).bin
      - sub_centroids.f32.bin
      - bucket_sub_offsets.u64.bin
    And an explicit sub_codes_path pointing to the dense codes payload (N * d bytes for u8; N * d * 4 for f32).

    Your manifest often uses names like:
      offsets.path = sub_offsets.u64.bin
      ids.path     = sub_ids.i64.csr.bin
      codes.path   = sub_codes.u8.csr.bin
      sidecars.*   = sub_bucket_id.u32.bin / sub_cluster_id.u16.bin / sub_centroids.f32.bin / bucket_sub_offsets.u64.bin
    We copy+rename into the runtime's expected basenames.
    """
    manifest_json = manifest_json.expanduser().resolve()
    if not manifest_json.is_file():
        die(f"cluster manifest json not found: {manifest_json}")

    with open(manifest_json, "r") as f:
        mani = json.load(f)

    # sanity
    if str(mani.get("format", "")) != "csr_inverted_lists_v1":
        eprint(f"[WARN] unexpected manifest format={mani.get('format')} (expected csr_inverted_lists_v1)")
    if str(mani.get("csr_kind", "")) != "cluster":
        eprint(f"[WARN] manifest csr_kind={mani.get('csr_kind')} (expected cluster)")

    subcsr_out = Path(shm_dir).expanduser().resolve() / "subcsr"
    ensure_dir(subcsr_out)

    def _get_path(node: Any, key: str) -> Path:
        if not isinstance(node, dict) or "path" not in node:
            die(f"manifest missing {key}.path")
        p = Path(str(node["path"])).expanduser().resolve()
        if not p.is_file():
            die(f"manifest {key}.path not found: {p}")
        return p

    # required core
    src_offsets = _get_path(mani.get("offsets"), "offsets")
    src_ids = _get_path(mani.get("ids"), "ids")
    src_codes = _get_path(mani.get("codes"), "codes")

    # required sidecars
    sidecars = mani.get("sidecars")
    if not isinstance(sidecars, dict):
        die("manifest missing sidecars dict")
    src_sub_bucket_id = _get_path(sidecars.get("sub_bucket_id"), "sidecars.sub_bucket_id")
    src_bucket_sub_offsets = _get_path(sidecars.get("bucket_sub_offsets"), "sidecars.bucket_sub_offsets")
    src_sub_centroids = _get_path(sidecars.get("sub_centroids"), "sidecars.sub_centroids")
    src_sub_cluster_id = _get_path(sidecars.get("sub_cluster_id"), "sidecars.sub_cluster_id")

    # copy+rename to runtime basenames
    copy_file_as(src_offsets, subcsr_out / "sub_offsets.u64.bin", verbose=verbose)
    copy_file_as(src_ids, subcsr_out / "sub_ids.i64.csr.bin", verbose=verbose)
    copy_file_as(src_sub_bucket_id, subcsr_out / "sub_bucket_id.u32.bin", verbose=verbose)
    # keep u16/u32 basename as-is; runtime picks via cfg.sub_cluster_u16
    if "u16" in src_sub_cluster_id.name.lower():
        copy_file_as(src_sub_cluster_id, subcsr_out / "sub_cluster_id.u16.bin", verbose=verbose)
    else:
        copy_file_as(src_sub_cluster_id, subcsr_out / "sub_cluster_id.u32.bin", verbose=verbose)
    copy_file_as(src_sub_centroids, subcsr_out / "sub_centroids.f32.bin", verbose=verbose)
    copy_file_as(src_bucket_sub_offsets, subcsr_out / "bucket_sub_offsets.u64.bin", verbose=verbose)

    # codes: keep original basename (csr.bin naming is OK); only path matters.
    staged_codes = copy_file_as(src_codes, subcsr_out / src_codes.name, verbose=verbose)

    # optional bucket_sizes (not required by bench_cluster_only, but cheap to stage)
    staged_sizes: Optional[Path] = None
    bs = mani.get("bucket_sizes")
    if isinstance(bs, dict) and bs.get("path"):
        p = Path(str(bs["path"])).expanduser().resolve()
        if p.is_file():
            staged_sizes = copy_file_as(p, subcsr_out / p.name, verbose=verbose)

    return {
        "subcsr_dir": subcsr_out,
        "sub_codes": staged_codes,
        "sub_sizes": staged_sizes,
        "manifest": manifest_json,
    }


def jget(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_path_in_tree(d: Any, predicate):
    """Return first node that satisfies predicate, DFS."""
    if predicate(d):
        return d
    if isinstance(d, dict):
        for v in d.values():
            out = find_path_in_tree(v, predicate)
            if out is not None:
                return out
    elif isinstance(d, list):
        for v in d:
            out = find_path_in_tree(v, predicate)
            if out is not None:
                return out
    return None


def _read_i32_pair(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as f:
        hdr = f.read(8)
    if len(hdr) != 8:
        die(f"File too small to read 8-byte header: {path}")
    a, b = struct.unpack("<ii", hdr)
    return int(a), int(b)


def _read_i32(path: Path) -> int:
    with open(path, "rb") as f:
        hdr = f.read(4)
    if len(hdr) != 4:
        die(f"File too small to read 4-byte header: {path}")
    (a,) = struct.unpack("<i", hdr)
    return int(a)


def copy_range_bytes(src: Path, dst: Path, *, src_off: int, nbytes: int, buf_mb: int = 16):
    buf = bytearray(max(1, buf_mb) * 1024 * 1024)
    chunk = len(buf)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fi.seek(src_off)
        left = int(nbytes)
        while left > 0:
            take = chunk if left > chunk else left
            data = fi.read(take)
            if not data:
                die(f"Unexpected EOF while copying bytes: src={src} off={src_off} left={left}")
            fo.write(data)
            left -= len(data)


# -----------------------------
# u8bin / fbin parsing + stripping
# -----------------------------
def parse_u8bin_auto(path: Path) -> Tuple[int, int, int]:
    """
    Returns (n, d, header_bytes).
    Tries:
      - 8 bytes: (n,d) validated by file size
      - 8 bytes: (d,n) validated by file size
      - 4 bytes: (d) with n inferred from file size
    """
    size = _filesize(path)
    if size < 4:
        die(f"u8bin too small: {path} size={size}")

    if size >= 8:
        a, b = _read_i32_pair(path)
        if a > 0 and b > 0:
            # try (n,d)
            exp1 = 8 + a * b
            if exp1 == size:
                return a, b, 8
            # try (d,n)
            exp2 = 8 + a * b
            if exp2 == size:
                # ambiguous: choose smaller as d
                if a <= b:
                    return b, a, 8
                return a, b, 8

    d = _read_i32(path)
    if d <= 0:
        die(f"Invalid d from 4-byte header: d={d} file={path}")
    payload = size - 4
    if payload % d != 0:
        die(
            f"u8bin 4-byte header detected but payload not divisible by d. "
            f"size={size}, d={d}, payload={payload}, file={path}"
        )
    n = payload // d
    if n <= 0:
        die(f"Invalid inferred n={n} for u8bin file={path}")
    return int(n), int(d), 4


def parse_fbin(path: Path) -> Tuple[int, int, int]:
    """
    fbin: 8-byte header (n,d) int32 + float32 payload
    """
    size = _filesize(path)
    if size < 8:
        die(f"fbin too small: {path} size={size}")
    n, d = _read_i32_pair(path)
    if n <= 0 or d <= 0:
        die(f"Invalid (n,d)=({n},{d}) in fbin header: {path}")
    exp = 8 + n * d * 4
    if exp != size:
        die(f"fbin size mismatch: header says (n={n},d={d}) => {exp} bytes, actual={size}: {path}")
    return int(n), int(d), 8


def strip_u8bin_to_raw(
    src: Path,
    dst: Path,
    *,
    nq: Optional[int] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    expect_d: Optional[int] = None,
    buf_mb: int = 16,
) -> Tuple[int, int]:
    n, d, hdr = parse_u8bin_auto(src)
    if expect_d is not None and d != int(expect_d):
        die(f"d mismatch: detected d={d}, expect_d={expect_d}, file={src}")

    if head is not None:
        nq = head
    if tail is not None and (nq is not None or head is not None):
        die("Use either --tail or --nq/--head, not both.")

    if tail is not None:
        t = int(tail)
        if t <= 0:
            die("--tail must be > 0")
        start_row = max(0, n - t)
        n_out = min(t, n)
    else:
        if nq is None:
            start_row = 0
            n_out = n
        else:
            q = int(nq)
            if q <= 0:
                die("--nq/--head must be > 0")
            start_row = 0
            n_out = min(q, n)

    src_off = hdr + start_row * d
    nbytes = n_out * d
    copy_range_bytes(src, dst, src_off=src_off, nbytes=nbytes, buf_mb=buf_mb)
    return int(n_out), int(d)


def strip_fbin_to_raw(
    src: Path,
    dst: Path,
    *,
    nq: Optional[int] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    expect_d: Optional[int] = None,
    buf_mb: int = 16,
) -> Tuple[int, int]:
    n, d, hdr = parse_fbin(src)
    if expect_d is not None and d != int(expect_d):
        die(f"d mismatch: detected d={d}, expect_d={expect_d}, file={src}")

    if head is not None:
        nq = head
    if tail is not None and (nq is not None or head is not None):
        die("Use either --tail or --nq/--head, not both.")

    if tail is not None:
        t = int(tail)
        if t <= 0:
            die("--tail must be > 0")
        start_row = max(0, n - t)
        n_out = min(t, n)
    else:
        if nq is None:
            start_row = 0
            n_out = n
        else:
            q = int(nq)
            if q <= 0:
                die("--nq/--head must be > 0")
            start_row = 0
            n_out = min(q, n)

    src_off = hdr + start_row * d * 4
    nbytes = n_out * d * 4
    copy_range_bytes(src, dst, src_off=src_off, nbytes=nbytes, buf_mb=buf_mb)
    return int(n_out), int(d)


# -----------------------------
# query staging (always headerless)
# -----------------------------
def stage_query_to_shm_as_raw(
    query_path: Path,
    shm_dir: Path,
    *,
    expected_d: Optional[int],
    out_name: Optional[str],
    npy_out_dtype: str,
    io_buf_mb: int,
) -> Tuple[Path, str, int, int]:
    """
    Returns (staged_path, dtype_str, nq, d)
    dtype_str: uint8|float32
    Always writes headerless raw payload (because pipeline doesn't support headers).
    """
    suf = query_path.suffix.lower()

    if suf == ".u8bin":
        dst = shm_dir / (out_name or "q.uint8.bin")
        nq, d = strip_u8bin_to_raw(query_path, dst, expect_d=expected_d, buf_mb=io_buf_mb)
        return dst, "uint8", nq, d

    if suf == ".fbin":
        dst = shm_dir / (out_name or "q.float32.bin")
        nq, d = strip_fbin_to_raw(query_path, dst, expect_d=expected_d, buf_mb=io_buf_mb)
        return dst, "float32", nq, d

    if suf == ".npy":
        import numpy as np
        dst = shm_dir / (out_name or f"q.{npy_out_dtype}.bin")
        arr = np.load(query_path, mmap_mode="r")
        if arr.ndim != 2:
            die(f"Query .npy must be 2D [nq,d], got shape={arr.shape}")
        nq, d = int(arr.shape[0]), int(arr.shape[1])
        if expected_d is not None and d != int(expected_d):
            die(f"Query d mismatch: query d={d}, expected_d={expected_d}, file={query_path}")

        if npy_out_dtype == "float32":
            out = np.asarray(arr, dtype=np.float32, order="C")
            dtype = "float32"
        elif npy_out_dtype == "uint8":
            if np.issubdtype(arr.dtype, np.floating):
                out = np.clip(arr, 0, 255).astype(np.uint8, copy=False)
            else:
                out = np.asarray(arr, dtype=np.uint8, order="C")
            dtype = "uint8"
        else:
            die(f"Unsupported npy_out_dtype={npy_out_dtype}")

        with open(dst, "wb") as f:
            f.write(out.tobytes(order="C"))
        return dst, dtype, nq, d

    die(f"Unsupported query format: {query_path} (supported: .u8bin .fbin .npy)")
    raise RuntimeError("unreachable")


# -----------------------------
# experiment extraction
# -----------------------------
def extract_csr_summary(exp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer: stages.csr_build.build.summary
    Fallback: any dict containing required CSR path keys.
    """
    s = jget(exp, "stages.csr_build.build.summary", None)
    if isinstance(s, dict) and all(k in s for k in ("codes_path", "ids_path", "offsets_bin_path", "bucket_sizes_path")):
        return s

    node = find_path_in_tree(
        exp,
        lambda x: isinstance(x, dict)
        and ("codes_path" in x)
        and ("ids_path" in x)
        and ("offsets_bin_path" in x)
        and ("bucket_sizes_path" in x),
    )
    if isinstance(node, dict):
        return node

    die("Could not find CSR summary with required paths in experiment JSON.")
    return {}


def extract_model_and_thresholds(exp: Dict[str, Any]) -> Dict[str, Any]:
    torchscript = jget(exp, "stages.model_selection.outputs.best_torchscript_path", None)
    if torchscript is None:
        torchscript = jget(exp, "stages.model_selection.outputs.best.torchscript_path", None)

    best_thr = jget(exp, "stages.model_selection.outputs.best.best_threshold", None)
    if best_thr is None:
        best_thr = jget(exp, "build_index.margin_position", None)

    return {"torchscript_path": torchscript, "threshold": best_thr}


def extract_vector_dim(exp: Dict[str, Any]) -> Optional[int]:
    d = jget(exp, "build_index.vector_dim", None)
    if d is None:
        d = jget(exp, "config.config.vector_dim", None)
    if d is None:
        d = jget(exp, "stages.csr_build.prepare.inputs.d", None)
    try:
        return int(d) if d is not None else None
    except Exception:
        return None


def extract_hidden_dim(exp: Dict[str, Any]) -> Optional[int]:
    b = jget(exp, "build_index.hidden_dim", None)
    if b is None:
        b = jget(exp, "config.config.encoding_dim", None)
    if b is None:
        b = jget(exp, "config.encoding_dim", None)
    try:
        return int(b) if b is not None else None
    except Exception:
        return None


def extract_earlystop_params_json(exp: Dict[str, Any]) -> Optional[str]:
    p = jget(exp, "stages.early_stop.eval.visualize.output_model.model", None)
    if isinstance(p, str) and p:
        return p

    node = find_path_in_tree(
        exp,
        lambda x: isinstance(x, dict)
        and ("output_model" in x)
        and isinstance(x.get("output_model"), dict)
        and ("model" in x["output_model"]),
    )
    if isinstance(node, dict):
        m = node.get("output_model", {}).get("model", None)
        if isinstance(m, str) and m:
            return m
    return None


def write_threshold_bin(thr: Any, out_path: Path, *, expected_len: Optional[int]) -> int:
    import numpy as np
    if thr is None:
        die("Threshold list not found in experiment JSON (need to write thr_f32_path).")
    if not isinstance(thr, list):
        die(f"Threshold is not a list. Got type={type(thr)}")
    arr = np.asarray(thr, dtype=np.float32)
    if arr.ndim != 1:
        die(f"Threshold must be 1D list. Got shape={arr.shape}")
    if expected_len is not None and int(arr.shape[0]) != int(expected_len):
        die(f"Threshold length mismatch: len(thr)={arr.shape[0]}, expected D={expected_len}")
    with open(out_path, "wb") as f:
        f.write(arr.tobytes(order="C"))
    return int(arr.shape[0])


def load_earlystop_a0b(path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Read a0/b from earlystop params json:
      { "a0": ..., "b": ... }
    """
    try:
        with open(path, "r") as f:
            d = json.load(f)
        a0 = d.get("a0", None)
        b = d.get("b", None)
        a0v = float(a0) if a0 is not None else None
        bv = float(b) if b is not None else None
        return a0v, bv
    except Exception as e:
        eprint(f"[WARN] Failed to read a0/b from earlystop_json={path}: {e}")
        return None, None


# -----------------------------
# make config (keys exactly as C++ parses)
# -----------------------------
def build_cpp_cfg(
    *,
    csr_base_dir: Optional[Path],
    query_path: Path,
    thr_f32_path: Path,
    logits_path: Optional[Path],
    torch_model_path: Optional[Path],
    metric: str,
    D: int,
    Q: int,
    repeats: int,
    omp_threads: int,
    csr_d: Optional[int],
    csr_N: Optional[int],
    csr_nlist: Optional[int],
    Lsel: int,
    Rmax: int,
    NBINS: int,
    bin_batch: int,
    enum_mode: str,
    x_budget_scalar: Optional[float],
    x_budget_f32_path: Optional[Path],
    a0: Optional[float],
    b: Optional[float],
    earlystop_json: Optional[Path],
    big_topm: Optional[int],
    phase2_vec_budget: Optional[int],
    gt_I_path: Optional[str],
    eval_K: Optional[int],
    qblock: Optional[int],
    big_qblock: Optional[int],
    sched_enum: Optional[str],
    chunk_enum: Optional[int],
    sched_big: Optional[str],
    chunk_big: Optional[int],
    sched_refine: Optional[str],
    chunk_refine: Optional[int],
    pretouch_csr: Optional[int],
    debug_big: Optional[int],
    torch_threads: Optional[int],
    torch_interop_threads: Optional[int],
    torch_try_set_interop: Optional[bool],
    torch_convert_threads: Optional[int],
    torch_u8_scale: Optional[float],
    torch_u8_zero_point: Optional[float],
    out_prefix: Optional[str],
    out_id_mode: Optional[str],
    out_k: Optional[int],
    # sweep passthrough (optional)
    sweep_Lsel: Optional[str],
    sweep_Rmax: Optional[str],
    sweep_x_budget: Optional[str],
    sweep_big_topm: Optional[str],
    sweep_phase2_vec_budget: Optional[str],
    sweep_csv_path: Optional[str],
    # ---- cluster-only / subcsr extras (optional) ----
    subcsr_meta_dir: Optional[Path] = None,
    subcsr_cluster_dir: Optional[Path] = None,
    sub_codes_path: Optional[Path] = None,
    D_code: Optional[int] = None,
    sub_codes_u8: Optional[bool] = None,
    sub_cluster_u16: Optional[bool] = None,
    gt_i64_bin: Optional[str] = None,
    gt_K: Optional[int] = None,
    heap_cap: Optional[int] = None,
    max_score: Optional[float] = None,
) -> Dict[str, Any]:
    # For cluster-only bench we can omit main CSR completely.
    # For full pipeline, keep providing csr_base_dir.
    if not query_path:
        die("query_path is required")
    if not thr_f32_path:
        die("thr_f32_path is required")

    if (torch_model_path is None or str(torch_model_path) == "") and (logits_path is None or str(logits_path) == ""):
        die("Must provide logits source: torch_model_path OR logits_path (one is required).")

    cfg: Dict[str, Any] = {
        "query_path": str(query_path),
        "thr_f32_path": str(thr_f32_path),

        "torch_model_path": str(torch_model_path) if torch_model_path else "",
        "logits_path": str(logits_path) if logits_path else "",

        "metric": metric,
        "D": int(D),
        "Q": int(Q),
        "repeats": int(repeats),
        "omp_threads": int(omp_threads),

        "Lsel": int(Lsel),
        "Rmax": int(Rmax),
        "NBINS": int(NBINS),
        "bin_batch": int(bin_batch),
        "enum_mode": enum_mode,
    }

    if csr_base_dir is not None:
        cfg["csr_base_dir"] = str(csr_base_dir)

    # cluster-only / subcsr keys (bench_cluster_only_* expects these)
    if subcsr_meta_dir is not None:
        cfg["subcsr_meta_dir"] = str(subcsr_meta_dir)
    if subcsr_cluster_dir is not None:
        cfg["subcsr_cluster_dir"] = str(subcsr_cluster_dir)


    # ---- explicit subcsr file paths (so C++ does not rely on fixed filenames) ----
    # We always *emit* these if subcsr_meta_dir/cluster_dir is provided, and we try to
    # pick an existing variant if multiple names are possible.
    def _pick_existing(cands):
        for p in cands:
            try:
                if p and Path(p).is_file():
                    return str(Path(p))
            except Exception:
                pass
        # fall back to the first candidate (even if it doesn't exist yet)
        return str(Path(cands[0])) if cands else ""

    if subcsr_meta_dir is not None:
        _m = Path(subcsr_meta_dir)
        # ids: prefer csr-suffixed name, then legacy
        cfg["sub_ids_path"] = _pick_existing([
            _m / "sub_ids.i64.csr.bin",
            _m / "sub_ids.int64.csr.bin",
            _m / "sub_ids.i64.bin",
            _m / "sub_ids.int64.bin",
        ])
        # offsets
        cfg["sub_offsets_path"] = _pick_existing([
            _m / "sub_offsets.u64.bin",
            _m / "sub_offsets.uint64.bin",
        ])
        # per-subvector metadata
        cfg["sub_bucket_id_path"] = _pick_existing([
            _m / "sub_bucket_id.u32.bin",
            _m / "sub_bucket_id.uint32.bin",
        ])
        cfg["sub_cluster_id_path"] = _pick_existing([
            _m / "sub_cluster_id.u16.bin",
            _m / "sub_cluster_id.uint16.bin",
        ])
        # bucket->sub offsets (len = nlist+1)
        cfg["bucket_sub_offsets_path"] = _pick_existing([
            _m / "bucket_sub_offsets.u64.bin",
            _m / "bucket_sub_offsets.uint64.bin",
        ])

    if subcsr_cluster_dir is not None:
        _c = Path(subcsr_cluster_dir)
        cfg["sub_centroids_path"] = _pick_existing([
            _c / "sub_centroids.f32.bin",
            _c / "sub_centroids.float32.bin",
        ])
        if sub_codes_path is not None:
            cfg["sub_codes_path"] = str(sub_codes_path)
        if sub_codes_u8 is not None:
            cfg["sub_codes_u8"] = bool(sub_codes_u8)
        if sub_cluster_u16 is not None:
            cfg["sub_cluster_u16"] = bool(sub_cluster_u16)
        if gt_i64_bin is not None:
            cfg["gt_i64_bin"] = str(gt_i64_bin)
        if gt_K is not None:
            cfg["gt_K"] = int(gt_K)
        if heap_cap is not None:
            cfg["heap_cap"] = int(heap_cap)
        if max_score is not None:
            cfg["max_score"] = float(max_score)
        if csr_N is not None:
            cfg["csr_N"] = int(csr_N)
        if csr_nlist is not None:
            cfg["csr_nlist"] = int(csr_nlist)

    if x_budget_scalar is not None:
        cfg["x_budget_scalar"] = float(x_budget_scalar)
    if x_budget_f32_path is not None:
        cfg["x_budget_f32_path"] = str(x_budget_f32_path)

    # a0/b: only emit if present (from CLI or earlystop_json)
    if a0 is not None:
        cfg["a0"] = float(a0)
    if b is not None:
        cfg["b"] = float(b)

    if earlystop_json is not None:
        cfg["earlystop_json"] = str(earlystop_json)

    if big_topm is not None:
        cfg["big_topm"] = int(big_topm)
    if phase2_vec_budget is not None:
        cfg["phase2_vec_budget"] = int(phase2_vec_budget)

    if gt_I_path:
        cfg["gt_I_path"] = str(gt_I_path)
    if eval_K is not None:
        cfg["eval_K"] = int(eval_K)

    if qblock is not None:
        cfg["qblock"] = int(qblock)
    if big_qblock is not None:
        cfg["big_qblock"] = int(big_qblock)

    if sched_enum is not None:
        cfg["sched_enum"] = str(sched_enum)
    if chunk_enum is not None:
        cfg["chunk_enum"] = int(chunk_enum)

    if sched_big is not None:
        cfg["sched_big"] = str(sched_big)
    if chunk_big is not None:
        cfg["chunk_big"] = int(chunk_big)

    if sched_refine is not None:
        cfg["sched_refine"] = str(sched_refine)
    if chunk_refine is not None:
        cfg["chunk_refine"] = int(chunk_refine)

    if pretouch_csr is not None:
        cfg["pretouch_csr"] = int(pretouch_csr)
    if debug_big is not None:
        cfg["debug_big"] = int(debug_big)

    if torch_threads is not None:
        cfg["torch_threads"] = int(torch_threads)
    if torch_interop_threads is not None:
        cfg["torch_interop_threads"] = int(torch_interop_threads)
    if torch_try_set_interop is not None:
        cfg["torch_try_set_interop"] = bool(torch_try_set_interop)
    if torch_convert_threads is not None:
        cfg["torch_convert_threads"] = int(torch_convert_threads)
    if torch_u8_scale is not None:
        cfg["torch_u8_scale"] = float(torch_u8_scale)
    if torch_u8_zero_point is not None:
        cfg["torch_u8_zero_point"] = float(torch_u8_zero_point)

    if out_prefix is not None:
        cfg["out_prefix"] = str(out_prefix)
    if out_id_mode is not None:
        cfg["out_id_mode"] = str(out_id_mode)
    if out_k is not None:
        cfg["out_k"] = int(out_k)

    # sweep passthrough
    if sweep_Lsel:
        cfg["sweep_Lsel"] = sweep_Lsel
    if sweep_Rmax:
        cfg["sweep_Rmax"] = sweep_Rmax
    if sweep_x_budget:
        cfg["sweep_x_budget"] = sweep_x_budget
    if sweep_big_topm:
        cfg["sweep_big_topm"] = sweep_big_topm
    if sweep_phase2_vec_budget:
        cfg["sweep_phase2_vec_budget"] = sweep_phase2_vec_budget
    if sweep_csv_path:
        cfg["sweep_csv_path"] = sweep_csv_path
    # ---- dimension keys (ALIGNED2) ----
    # D_in: original/input dimension used for refine (from metric suffix, e.g., l2_u8_128 -> 128)
    cfg["D_in"] = int(_parse_D_in_from_metric(metric) or 0)
    # D: logits / cluster-space dimension (already stored as cfg["D"])
    cfg["sub_centroids_d"] = int(cfg.get("D", 0))
    cfg["sub_codes_d"] = int(cfg["D_in"])
    # cap_per_bin is used by bench; keep NBINS separate
    cfg.setdefault("cap_per_bin", 256)
    # ---- id sidecar (bench legacy) ----
    # Some benches expect subcsr/sub_ids.i64.bin even if experiment outputs CSR-named ids.
    cfg["sub_ids_path"] = os.path.join(str(subcsr_meta_dir), "sub_ids.i64.bin")





    return cfg




# -----------------------------
# subcommand: make
# -----------------------------
def cmd_make(args: argparse.Namespace) -> None:
    exp_json_path = Path(args.exp_json).expanduser().resolve()
    if not exp_json_path.is_file():
        die(f"exp_json not found: {exp_json_path}")

    query_path = Path(args.query).expanduser().resolve()
    if not query_path.is_file():
        die(f"query not found: {query_path}")

    shm_dir = Path(args.shm_dir).expanduser().resolve()
    ensure_dir(shm_dir)

    out_json_path = Path(args.out_json).expanduser().resolve()
    ensure_dir(out_json_path.parent)
    if out_json_path.exists() and not args.overwrite:
        die(f"Refuse to overwrite: {out_json_path}")

    with open(exp_json_path, "r") as f:
        exp = json.load(f)

    csr: Dict[str, Any] = {}
    staged_subcsr_dir: Optional[Path] = None
    staged_sub_codes: Optional[Path] = None
    manifest_maybe: Optional[Path] = None

    # Prefer explicit cluster-manifest staging (cluster-only layout)
    if args.cluster_manifest_json:
        manifest_maybe = Path(args.cluster_manifest_json).expanduser().resolve()
        if args.stage_to_shm:
            eprint(f"[INFO] Staging CLUSTER manifest -> shm_dir={shm_dir}: {manifest_maybe}")
            staged = stage_cluster_manifest_v1(manifest_maybe, shm_dir, verbose=True)
            staged_subcsr_dir = staged.get("subcsr_dir", None)
            staged_sub_codes = staged.get("sub_codes", None)
            # Ensure legacy id sidecar name exists for benches that still expect sub_ids.i64.bin
            if staged_subcsr_dir is not None:
                _ensure_sub_ids_flat(str(staged_subcsr_dir), verbose=True)
        else:
            staged_subcsr_dir = shm_dir / "subcsr"

            _ensure_sub_ids_flat(str(staged_subcsr_dir), verbose=True)
    # Else: Prefer explicit results dir staging (newer v3_with_subcsr layout)
    elif args.csr_results_dir:
        resdir = Path(args.csr_results_dir).expanduser().resolve()
        if args.stage_to_shm:
            # Prefer staging by *cluster manifest* if present in the directory; it carries d/N/nlist.
            man_in_dir = resdir / "subcsr_manifest.json"
            if man_in_dir.is_file():
                manifest_maybe = man_in_dir
                eprint(f"[INFO] Staging CLUSTER manifest -> shm_dir={shm_dir}: {manifest_maybe}")
                staged = stage_cluster_manifest_v1(manifest_maybe, shm_dir, verbose=True)
            else:
                eprint(f"[INFO] Staging CSR/SubCSR (v3_with_subcsr) from results_dir={resdir} -> shm_dir={shm_dir}")
                # Decide whether results_dir contains full CSR+SubCSR or SubCSR-only.
                has_main = (resdir / "offsets.uint64.bin").is_file() or (resdir / "csr_manifest.json").is_file()
                if has_main:
                    staged = stage_csr_index_v3_with_subcsr(resdir, shm_dir, verbose=True)
                else:
                    eprint("[INFO] No main CSR artifacts detected; staging SubCSR-only.")
                    staged = stage_subcsr_only_from_dir(resdir, shm_dir, verbose=True)

            staged_subcsr_dir = staged.get("subcsr_dir", None)
            staged_sub_codes = staged.get("sub_codes", None)
            # Ensure legacy id sidecar name exists for benches that still expect sub_ids.i64.bin
            if staged_subcsr_dir is not None:
                _ensure_sub_ids_flat(str(staged_subcsr_dir), verbose=True)
        else:
            eprint("[WARN] --csr_results_dir provided but --stage_to_shm=0; cfg will point at shm_dir but files may be missing")
            staged_subcsr_dir = shm_dir / "subcsr"
            _ensure_sub_ids_flat(str(staged_subcsr_dir), verbose=True)
    else:
        # Backward-compatible: read CSR paths from experiment JSON
        csr = extract_csr_summary(exp)
    model = extract_model_and_thresholds(exp)

    vec_d = extract_vector_dim(exp)       # e.g. 1536
    hidden_d = extract_hidden_dim(exp)    # e.g. 17
    if hidden_d is None:
        die("Could not extract hidden_dim (D) from experiment JSON.")
    D = int(hidden_d)

    # If cluster manifest is provided, try to auto-fill D_code / dtype flags from manifest.
    # (User can still override by CLI.)
    mani_d_code: Optional[int] = None
    mani_codes_u8: Optional[bool] = None
    mani_cluster_u16: Optional[bool] = None
    if manifest_maybe is not None and manifest_maybe.is_file():
        try:
            with open(manifest_maybe, "r") as f:
                mani = json.load(f)
            if isinstance(mani, dict):
                if "d" in mani:
                    mani_d_code = int(mani["d"])
                c = mani.get("codes")
                if isinstance(c, dict) and c.get("dtype"):
                    dt = str(c.get("dtype")).lower()
                    if dt in ("uint8", "u8"):
                        mani_codes_u8 = True
                    elif dt in ("float32", "f32"):
                        mani_codes_u8 = False
                sc = mani.get("sidecars", {})
                if isinstance(sc, dict):
                    scc = sc.get("sub_cluster_id")
                    if isinstance(scc, dict) and scc.get("dtype"):
                        mani_cluster_u16 = (str(scc.get("dtype")).lower() in ("uint16", "u16"))
        except Exception as e:
            eprint(f"[WARN] failed to parse cluster manifest for defaults: {e}")

    # ---- CSR source files (legacy) ----
    # If we are in cluster-only mode (cluster manifest provided), skip staging main CSR.
    if (not args.csr_results_dir) and (not args.cluster_manifest_json):
        src_codes = Path(str(csr["codes_path"])).expanduser().resolve()
        src_ids = Path(str(csr["ids_path"])).expanduser().resolve()
        src_offsets = Path(str(csr["offsets_bin_path"])).expanduser().resolve()
        src_bucket_sizes = Path(str(csr["bucket_sizes_path"])).expanduser().resolve()

        src_big_bucket_ids: Optional[Path] = None
        if args.stage_big_bucket_ids and csr.get("big_bucket_ids_path"):
            p = Path(str(csr["big_bucket_ids_path"])).expanduser().resolve()
            if p.is_file():
                src_big_bucket_ids = p

        for p in (src_codes, src_ids, src_offsets, src_bucket_sizes):
            if not p.is_file():
                die(f"Missing CSR source file: {p}")

        # stage CSR (preserve basenames)
        if args.stage_to_shm:
            eprint(f"[INFO] Staging CSR files into: {shm_dir} (preserve basenames)")
            copy_file_same_name(src_codes, shm_dir)
            copy_file_same_name(src_ids, shm_dir)
            copy_file_same_name(src_offsets, shm_dir)
            copy_file_same_name(src_bucket_sizes, shm_dir)
            if args.stage_big_bucket_ids and src_big_bucket_ids is not None:
                copy_file_same_name(src_big_bucket_ids, shm_dir)

    # ---- stage query as RAW payload (no header) ----
    # NOTE: pipeline requires headerless, so .npy will be converted to raw .bin automatically.
    staged_q, q_dtype, q_nq, q_d = stage_query_to_shm_as_raw(
        query_path,
        shm_dir,
        expected_d=vec_d,
        out_name=args.query_out_name,
        npy_out_dtype=args.npy_out_dtype,
        io_buf_mb=args.io_buf_mb,
    )

    if vec_d is None:
        vec_d = int(q_d)

    # ---- write thr_f32_path (float32 * D) ----
    thr_out = shm_dir / (args.thr_out_name or "threshold_vec.f32.bin")
    thr_len = write_threshold_bin(model.get("threshold"), thr_out, expected_len=D)

    # ---- resolve torch model path ----
    torchscript_path = model.get("torchscript_path", None)
    if args.torch_model_path:
        torchscript_path = args.torch_model_path

    torch_model_path: Optional[Path] = None
    if torchscript_path:
        torch_model_path = Path(str(torchscript_path)).expanduser().resolve()
        if not torch_model_path.is_file():
            die(f"torch_model_path not found: {torch_model_path}")
        if args.stage_torch_model:
            torch_model_path = copy_file_same_name(torch_model_path, shm_dir)

    # ---- logits_path alternative ----
    logits_path: Optional[Path] = None
    if args.logits_path:
        logits_path = Path(args.logits_path).expanduser().resolve()
        if not logits_path.is_file():
            die(f"logits_path not found: {logits_path}")
        if args.stage_logits:
            logits_path = copy_file_same_name(logits_path, shm_dir)

    # ---- earlystop_json (optional) ----
    earlystop_json_path: Optional[Path] = None
    if args.earlystop_json:
        p = Path(args.earlystop_json).expanduser().resolve()
        if not p.is_file():
            die(f"--earlystop_json not found: {p}")
        earlystop_json_path = copy_file_same_name(p, shm_dir) if args.stage_to_shm else (shm_dir / p.name)
    else:
        p = extract_earlystop_params_json(exp)
        if p:
            pp = Path(p).expanduser().resolve()
            if pp.is_file():
                earlystop_json_path = copy_file_same_name(pp, shm_dir) if args.stage_to_shm else (shm_dir / pp.name)
            else:
                eprint(f"[WARN] earlystop json path found but file missing: {pp}")

    
    # ---- GT staging (optional) ----
    # cluster-only bench uses gt_i64_bin (raw int64 Q*K) + gt_K.
    staged_gt_i64: Optional[Path] = None
    if args.gt_i64_bin:
        p = Path(args.gt_i64_bin).expanduser().resolve()
        if not p.is_file():
            die(f"--gt_i64_bin not found: {p}")
        staged_gt_i64 = copy_file_same_name(p, shm_dir) if args.stage_to_shm else p

    # pipeline bench may use gt_I_path (often int64 .bin). Stage it too if provided.
    staged_gt_I: Optional[Path] = None
    if args.gt_I_path:
        p = Path(args.gt_I_path).expanduser().resolve()
        if not p.is_file():
            die(f"--gt_I_path not found: {p}")
        staged_gt_I = copy_file_same_name(p, shm_dir) if args.stage_to_shm else p
    # ---- CSR meta: MUST prefer csr_build.build.summary.{N,nlist} ----
    csr_N: Optional[int] = None
    csr_nlist: Optional[int] = None
    if isinstance(csr, dict):
        if "N" in csr:
            try:
                csr_N = int(csr["N"])
            except Exception:
                csr_N = None
        if "nlist" in csr:
            try:
                csr_nlist = int(csr["nlist"])
            except Exception:
                csr_nlist = None

    # extra fallback: sometimes N/nlist stored under csr["config"] or other node
    if csr_N is None:
        n2 = jget(exp, "stages.csr_build.build.summary.N", None)
        if n2 is not None:
            try:
                csr_N = int(n2)
            except Exception:
                csr_N = None
    if csr_nlist is None:
        nl2 = jget(exp, "stages.csr_build.build.summary.nlist", None)
        if nl2 is not None:
            try:
                csr_nlist = int(nl2)
            except Exception:
                csr_nlist = None

    csr_d: Optional[int] = int(vec_d) if vec_d is not None else None

    # If we have a cluster manifest, use it to populate d/N/nlist.
    if manifest_maybe is not None and Path(manifest_maybe).is_file():
        try:
            with open(Path(manifest_maybe), "r") as f:
                mani = json.load(f)
            mm = _extract_manifest_meta(mani)
            if csr_d is None and mm.get("d") is not None:
                csr_d = int(mm["d"])
            if csr_N is None and mm.get("N") is not None:
                csr_N = int(mm["N"])
            if csr_nlist is None and mm.get("nlist") is not None:
                csr_nlist = int(mm["nlist"])
        except Exception as e:
            eprint(f"[WARN] failed to read manifest meta from {manifest_maybe}: {e}")

    # Final fallback (cluster-only): infer nlist from staged subcsr offsets.
    if (csr_nlist is None or csr_nlist <= 0) and staged_subcsr_dir is not None:
        offp = Path(staged_subcsr_dir) / "sub_offsets.u64.bin"
        if offp.is_file():
            inf = _infer_nlist_from_u64_offsets(offp)
            if inf is not None and inf > 0:
                csr_nlist = int(inf)

    # ---- Q in config ----
    Q = int(args.Q) if args.Q is not None else int(q_nq)

    # ---- a0/b: if not provided by CLI, read from earlystop_json ----
    a0 = args.a0
    b = args.b
    if (a0 is None or b is None) and earlystop_json_path is not None and earlystop_json_path.is_file():
        ea0, eb = load_earlystop_a0b(earlystop_json_path)
        if a0 is None:
            a0 = ea0
        if b is None:
            b = eb

    # ---- build final C++ config ----
    # Final safety: ensure legacy sub_ids name exists after any staging path.
    if staged_subcsr_dir is not None:
        _ensure_sub_ids_flat(str(staged_subcsr_dir), verbose=True)

    cfg = build_cpp_cfg(
        # IMPORTANT:
        # - For cluster-only bench, DO NOT set csr_base_dir, otherwise the bench will auto-fill
        #   csr_{offsets,ids,bucket_sizes} paths and try to mmap main CSR.
        # - For full pipeline, keep csr_base_dir.
        csr_base_dir=(None if args.cluster_only else shm_dir),
        query_path=staged_q,
        thr_f32_path=thr_out,
        logits_path=logits_path,
        torch_model_path=torch_model_path,

        metric=args.metric,
        D=D,
        Q=Q,
        repeats=args.repeats,
        omp_threads=args.omp_threads,

        csr_d=csr_d,
        csr_N=csr_N,
        csr_nlist=csr_nlist,

        Lsel=args.Lsel,
        Rmax=args.Rmax,
        NBINS=args.NBINS,
        bin_batch=args.bin_batch,
        enum_mode=args.enum_mode,

        x_budget_scalar=args.x_budget_scalar,
        x_budget_f32_path=Path(args.x_budget_f32_path).expanduser().resolve() if args.x_budget_f32_path else None,
        a0=a0,
        b=b,
        earlystop_json=earlystop_json_path,

        big_topm=args.big_topm,
        phase2_vec_budget=args.phase2_vec_budget,
        gt_I_path=(str(staged_gt_I) if staged_gt_I else args.gt_I_path),
        eval_K=args.eval_K,

        qblock=args.qblock,
        big_qblock=args.big_qblock,
        sched_enum=args.sched_enum,
        chunk_enum=args.chunk_enum,
        sched_big=args.sched_big,
        chunk_big=args.chunk_big,
        sched_refine=args.sched_refine,
        chunk_refine=args.chunk_refine,

        pretouch_csr=args.pretouch_csr,
        debug_big=args.debug_big,

        torch_threads=args.torch_threads,
        torch_interop_threads=args.torch_interop_threads,
        torch_try_set_interop=args.torch_try_set_interop,
        torch_convert_threads=args.torch_convert_threads,
        torch_u8_scale=args.torch_u8_scale,
        torch_u8_zero_point=args.torch_u8_zero_point,

        out_prefix=args.out_prefix,
        out_id_mode=args.out_id_mode,
        out_k=args.out_k,

        sweep_Lsel=args.sweep_Lsel,
        sweep_Rmax=args.sweep_Rmax,
        sweep_x_budget=args.sweep_x_budget,
        sweep_big_topm=args.sweep_big_topm,
        sweep_phase2_vec_budget=args.sweep_phase2_vec_budget,
        sweep_csv_path=args.sweep_csv_path,

        # cluster-only / subcsr extras
        subcsr_meta_dir=staged_subcsr_dir,
        subcsr_cluster_dir=staged_subcsr_dir,
        sub_codes_path=staged_sub_codes,
        D_code=(args.D_code if args.D_code is not None else mani_d_code),
        sub_codes_u8=(
            (None if args.sub_codes_u8 is None else bool(int(args.sub_codes_u8)))
            if args.sub_codes_u8 is not None
            else mani_codes_u8
        ),
        sub_cluster_u16=(
            (None if args.sub_cluster_u16 is None else bool(int(args.sub_cluster_u16)))
            if args.sub_cluster_u16 is not None
            else mani_cluster_u16
        ),
        gt_i64_bin=(str(staged_gt_i64) if staged_gt_i64 else args.gt_i64_bin),
        gt_K=args.gt_K,
        heap_cap=args.heap_cap,
        max_score=args.max_score,
)

    # In cluster-only mode, make sure main CSR paths are empty (avoid accidental mmap).
    if args.cluster_only:
        cfg.pop("csr_base_dir", None)
        cfg["csr_offsets_path"] = ""
        cfg["csr_ids_path"] = ""
        cfg["csr_bucket_sizes_path"] = ""

    
# ---- bench alignment: derive nlist from bucket_sub_offsets (len = nlist+1) ----
    with open(out_json_path, "w") as f:
        json.dump(cfg, f, indent=2)
    eprint("========================================")
    eprint(f"[OK] wrote C++ config      : {out_json_path}")
    eprint(f"[OK] csr_base_dir          : {shm_dir}")
    eprint(f"[OK] query_path (raw)      : {staged_q}  (dtype={q_dtype}, nq={q_nq}, d={q_d})")
    eprint(f"[OK] thr_f32_path           : {thr_out} (len={thr_len}, bytes={_filesize(thr_out)})")
    eprint(f"[OK] csr meta               : d={csr_d} N={csr_N} nlist={csr_nlist}")
    eprint(f"[OK] a0/b                   : a0={a0} b={b}  (source: {'CLI' if (args.a0 is not None or args.b is not None) else 'earlystop_json' if earlystop_json_path else 'none'})")
    if torch_model_path:
        eprint(f"[OK] torch_model_path      : {torch_model_path}")
    if logits_path:
        eprint(f"[OK] logits_path           : {logits_path}")
    eprint(f"[OK] earlystop_json         : {earlystop_json_path}")
    eprint("========================================")


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- make ----
    apm = sub.add_parser("make", help="Make C++ config from experiment json + stage files to shm")
    apm.add_argument("--exp_json", required=True, type=str)
    apm.add_argument("--query", required=True, type=str)

    # If provided, prefer staging CSR/SubCSR from this results directory (v3_with_subcsr layout),
    # instead of relying on codes_path/ids_path/offsets_path from exp_json.
    apm.add_argument("--csr_results_dir", type=str, default=None,
                     help="Directory containing offsets/ids/indices/bucket_sizes + subcsr_manifest.json."
                     )

    # If provided, use this *cluster* manifest and stage ONLY the subcsr assets required by bench_cluster_only.
    # This is the recommended mode for your new pipeline (no main CSR).
    apm.add_argument("--cluster_manifest_json", type=str, default=None,
                     help="Cluster CSR manifest (csr_inverted_lists_v1, csr_kind=cluster).")

    apm.add_argument("--cluster_only", type=int, default=0,
                     help="1: emit cfg for cluster-only bench (omit main CSR). If --cluster_manifest_json is set, this should be 1.")

    apm.add_argument("--stage_to_shm", type=int, default=1)
    apm.add_argument("--shm_dir", type=str, default="/dev/shm/csr")

    apm.add_argument("--query_out_name", type=str, default=None)
    apm.add_argument("--npy_out_dtype", type=str, default="float32", choices=["float32", "uint8"])
    apm.add_argument("--io_buf_mb", type=int, default=16)

    apm.add_argument("--thr_out_name", type=str, default="threshold_vec.f32.bin")

    apm.add_argument("--stage_big_bucket_ids", type=int, default=0)

    apm.add_argument("--torch_model_path", type=str, default=None, help="Override torch_model_path (torchscript).")
    apm.add_argument("--logits_path", type=str, default=None, help="Alternative logits_path (instead of torch).")
    apm.add_argument("--stage_torch_model", type=int, default=1, help="Copy torch model into shm to speed IO.")
    apm.add_argument("--stage_logits", type=int, default=1, help="Copy logits_path into shm if provided.")

    apm.add_argument("--earlystop_json", type=str, default=None, help="Override earlystop params json path.")

    apm.add_argument("--out_json", required=True, type=str)
    apm.add_argument("--overwrite", type=int, default=1)

    apm.add_argument("--metric", type=str, default="l2_u8_128",
                     choices=["ip1536", "l2_u8_128", "l2_f32_96", "l2_f32_1536"])
    apm.add_argument("--Q", type=int, default=None)
    apm.add_argument("--repeats", type=int, default=1)
    apm.add_argument("--omp_threads", type=int, default=24)

    apm.add_argument("--Lsel", type=int, default=16)
    apm.add_argument("--Rmax", type=int, default=7)
    apm.add_argument("--NBINS", type=int, default=256)
    apm.add_argument("--bin_batch", type=int, default=8)
    apm.add_argument("--enum_mode", type=str, default="pruned_dfs", choices=["pruned_dfs"])

    apm.add_argument("--x_budget_scalar", type=float, default=0.08)
    apm.add_argument("--x_budget_f32_path", type=str, default=None)

    # IMPORTANT: do NOT set default a0/b; let them come from earlystop_json unless user overrides.
    apm.add_argument("--a0", type=float, default=None)
    apm.add_argument("--b", type=float, default=None)

    apm.add_argument("--big_topm", type=int, default=10)
    apm.add_argument("--phase2_vec_budget", type=int, default=1500000)

    apm.add_argument("--gt_I_path", type=str, default=None)
    apm.add_argument("--eval_K", type=int, default=None)

    # cluster-only bench extras (kept optional for backward compat)
    apm.add_argument("--gt_i64_bin", type=str, default=None, help="GT I int64 raw (Q*K) file")
    apm.add_argument("--gt_K", type=int, default=None, help="GT K (columns)")
    apm.add_argument("--D_code", type=int, default=None, help="Compressed code dim (e.g. 17/20/22)")
    apm.add_argument("--sub_codes_u8", type=int, default=None, help="1 if sub_codes are uint8, 0 if float32")
    apm.add_argument("--sub_cluster_u16", type=int, default=None, help="1 if sub_cluster_id uses uint16")
    apm.add_argument("--heap_cap", type=int, default=None, help="Per-query heap capacity (cluster candidates)")
    apm.add_argument("--max_score", type=float, default=None, help="Optional score threshold to stop/limit enum")

    apm.add_argument("--qblock", type=int, default=32)
    apm.add_argument("--big_qblock", type=int, default=32)

    apm.add_argument("--sched_enum", type=str, default="dynamic", choices=["static", "dynamic", "guided"])
    apm.add_argument("--chunk_enum", type=int, default=32)
    apm.add_argument("--sched_big", type=str, default="guided", choices=["static", "dynamic", "guided"])
    apm.add_argument("--chunk_big", type=int, default=32)
    apm.add_argument("--sched_refine", type=str, default="guided", choices=["static", "dynamic", "guided"])
    apm.add_argument("--chunk_refine", type=int, default=32)

    apm.add_argument("--pretouch_csr", type=int, default=0)
    apm.add_argument("--debug_big", type=int, default=0)

    apm.add_argument("--torch_threads", type=int, default=1)
    apm.add_argument("--torch_interop_threads", type=int, default=1)
    apm.add_argument("--torch_try_set_interop", type=int, default=0, help="0/1; written as bool in JSON")
    apm.add_argument("--torch_convert_threads", type=int, default=8)
    apm.add_argument("--torch_u8_scale", type=float, default=None)
    apm.add_argument("--torch_u8_zero_point", type=float, default=0.0)

    apm.add_argument("--out_prefix", type=str, default="")
    apm.add_argument("--out_id_mode", type=str, default="external", choices=["auto", "internal", "external"])
    apm.add_argument("--out_k", type=int, default=None)

    # sweep passthrough
    apm.add_argument("--sweep_Lsel", type=str, default=None)
    apm.add_argument("--sweep_Rmax", type=str, default=None)
    apm.add_argument("--sweep_x_budget", type=str, default=None)
    apm.add_argument("--sweep_big_topm", type=str, default=None)
    apm.add_argument("--sweep_phase2_vec_budget", type=str, default=None)
    apm.add_argument("--sweep_csv_path", type=str, default=None)
    apm.add_argument("--sweep_max_score", type=str, default=None)

    # ---- strip_u8bin ----
    apu = sub.add_parser("strip_u8bin", help="Strip u8bin header -> raw payload")
    apu.add_argument("--in_u8bin", required=True, type=str)
    apu.add_argument("--out_raw", required=True, type=str)
    apu.add_argument("--expect_d", type=int, default=None)
    apu.add_argument("--nq", type=int, default=None)
    apu.add_argument("--head", type=int, default=None)
    apu.add_argument("--tail", type=int, default=None)
    apu.add_argument("--io_buf_mb", type=int, default=16)

    # ---- strip_fbin ----
    apf = sub.add_parser("strip_fbin", help="Strip fbin header -> raw payload")
    apf.add_argument("--in_fbin", required=True, type=str)
    apf.add_argument("--out_raw", required=True, type=str)
    apf.add_argument("--expect_d", type=int, default=None)
    apf.add_argument("--nq", type=int, default=None)
    apf.add_argument("--head", type=int, default=None)
    apf.add_argument("--tail", type=int, default=None)
    apf.add_argument("--io_buf_mb", type=int, default=16)

    args = ap.parse_args()

    if args.cmd == "make":
        args.stage_to_shm = bool(int(args.stage_to_shm))
        args.stage_torch_model = bool(int(args.stage_torch_model))
        args.stage_logits = bool(int(args.stage_logits))
        args.stage_big_bucket_ids = bool(int(args.stage_big_bucket_ids))
        args.cluster_only = bool(int(args.cluster_only))
        args.torch_try_set_interop = bool(int(args.torch_try_set_interop)) if args.torch_try_set_interop is not None else False
        cmd_make(args)
        return

    if args.cmd == "strip_u8bin":
        src = Path(args.in_u8bin).expanduser().resolve()
        dst = Path(args.out_raw).expanduser().resolve()
        ensure_dir(dst.parent)
        if not src.is_file():
            die(f"input u8bin not found: {src}")
        nq, d = strip_u8bin_to_raw(
            src, dst,
            nq=args.nq, head=args.head, tail=args.tail,
            expect_d=args.expect_d,
            buf_mb=args.io_buf_mb,
        )
        eprint(f"[OK] wrote raw u8 payload: {dst}  (n={nq}, d={d}, bytes={_filesize(dst)})")
        return

    if args.cmd == "strip_fbin":
        src = Path(args.in_fbin).expanduser().resolve()
        dst = Path(args.out_raw).expanduser().resolve()
        ensure_dir(dst.parent)
        if not src.is_file():
            die(f"input fbin not found: {src}")
        nq, d = strip_fbin_to_raw(
            src, dst,
            nq=args.nq, head=args.head, tail=args.tail,
            expect_d=args.expect_d,
            buf_mb=args.io_buf_mb,
        )
        eprint(f"[OK] wrote raw f32 payload: {dst}  (n={nq}, d={d}, bytes={_filesize(dst)})")
        return

    die(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
