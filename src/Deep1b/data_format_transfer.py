#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import numpy as np


def fbin_to_npy(fbin_path: str, npy_path: str | None = None, mmap_out: bool = True) -> str:
    """
    Convert FAISS .fbin (float32) file to .npy.

    .fbin format:
      int32 n (num vectors)
      int32 d (dim)
      then n*d float32 values row-major

    Args:
        fbin_path: input .fbin file path
        npy_path : output .npy file path (default: same name + .npy)
        mmap_out : if True, write via np.memmap to avoid huge RAM usage

    Returns:
        output .npy file path
    """
    if npy_path is None:
        npy_path = os.path.splitext(fbin_path)[0] + ".npy"

    with open(fbin_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError("File too small: cannot read (n, d) header.")

        n, d = struct.unpack("ii", header)
        print(f"[fbin] n={n:,}  d={d}  (float32)")

        # total float32 count
        total = n * d
        data_offset = 8

        # Create output array (mmap or normal)
        if mmap_out:
            # memmap output file (biggest-safe way)
            out = np.lib.format.open_memmap(
                npy_path, mode="w+", dtype=np.float32, shape=(n, d)
            )

            # Memory-map the fbin payload directly (no RAM copy)
            payload = np.memmap(
                fbin_path, mode="r", dtype=np.float32, offset=data_offset, shape=(total,)
            )

            # Copy in chunks (prevents large transient memory usage)
            chunk_rows = max(1, min(n, 1_000_000))  # tune if needed
            for i in range(0, n, chunk_rows):
                j = min(n, i + chunk_rows)
                out[i:j, :] = payload[i * d : j * d].reshape(j - i, d)

                if (i // chunk_rows) % 10 == 0:
                    print(f"[write] {j:,}/{n:,} rows")

            out.flush()
        else:
            # Full load into RAM (only safe if file fits in memory)
            payload = np.fromfile(f, dtype=np.float32, count=total)
            if payload.size != total:
                raise RuntimeError("Unexpected EOF: file truncated?")
            out = payload.reshape(n, d)
            np.save(npy_path, out)

    print(f"[OK] saved to: {npy_path}")
    return npy_path


if __name__ == "__main__":
    fbin_to_npy("/path/to/big-ann-benchmarks/data/deep1b//base.100M.fbin", "/path/to/big-ann-benchmarks/data/deep1b/base.100M.npy", mmap_out=False)
