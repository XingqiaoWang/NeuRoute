#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import numpy as np
import argparse
import os


class U8BinLoader:
    def __init__(self, path: str):
        self.path = path
        with open(self.path, "rb") as f:
            # Read header (first 8 bytes: nb, dim)
            header = f.read(8)
            if len(header) != 8:
                raise RuntimeError("Invalid u8bin header.")
            self.nb, self.dim = struct.unpack("<ii", header)

            # Read the rest of the file directly into RAM
            data = np.frombuffer(f.read(), dtype=np.uint8)
            expected_size = self.nb * self.dim
            if data.size != expected_size:
                raise RuntimeError(
                    f"Data size mismatch: expected {expected_size}, got {data.size}"
                )

            # Reshape to (nb, dim)
            self.data = data.reshape(self.nb, self.dim)

    def __len__(self):
        return self.nb

    def get_rows(self, start: int, end: int) -> np.ndarray:
        return self.data[start:end]


def save_to_npy(data: np.ndarray, output_path: str, no_scale: bool):
    """
    Save the dataset to .npy, with optional scaling.
    """
    if no_scale:
        # Keep original values as float32
        out_data = data.astype(np.float32)
        print("[INFO] Saving without scaling. Values remain in original range [0, 255].")
    else:
        # Scale to [0, 1]
        out_data = data.astype(np.float32) / 255.0
        print("[INFO] Scaling values to [0, 1] before saving.")

    np.save(output_path, out_data)
    print(f"[INFO] Saved data to {output_path} with shape {out_data.shape} and dtype {out_data.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Transform .u8bin file to .npy with optional sampling or full export")
    parser.add_argument("--in", dest="input", required=True, help="Path to input .u8bin file")
    parser.add_argument("--out", dest="output", required=True, help="Path to output .npy file")
    parser.add_argument("--n", type=int, default=2_000_000, help="Number of vectors to sample (ignored if --full is set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--full", action="store_true", help="Process the entire dataset instead of sampling")
    parser.add_argument("--no-scale", action="store_true", help="Keep original values without scaling to [0,1]")
    args = parser.parse_args()

    # Load the entire file into RAM
    base = U8BinLoader(args.input)
    print(f"[INFO] Loaded {args.input}: shape = ({base.nb}, {base.dim}) into RAM")

    if args.full:
        # Process the entire dataset
        print("[INFO] Processing the full dataset...")
        save_to_npy(base.data, args.output, args.no_scale)
    else:
        # Subsample
        if args.n > base.nb:
            raise ValueError(f"Requested {args.n} vectors, but file only contains {base.nb}")

        print(f"[INFO] Sampling {args.n} vectors (seed={args.seed})...")
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(base.nb, size=args.n, replace=False)

        sampled = base.data[indices]
        save_to_npy(sampled, args.output, args.no_scale)


if __name__ == "__main__":
    main()
