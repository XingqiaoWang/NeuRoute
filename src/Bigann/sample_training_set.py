#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import struct
import numpy as np
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Subsample and scale .u8bin file to .npy")
    parser.add_argument("--in", dest="input", required=True, help="Path to input .u8bin file")
    parser.add_argument("--out", dest="output", required=True, help="Path to output .npy file")
    parser.add_argument("--n", type=int, default=2_000_000, help="Number of vectors to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load entire file into RAM
    base = U8BinLoader(args.input)
    print(f"Loaded {args.input}: shape = ({base.nb}, {base.dim}) into RAM")

    if args.n > base.nb:
        raise ValueError(f"Requested {args.n} vectors, but file only contains {base.nb}")

    # Random subsample
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(base.nb, size=args.n, replace=False)

    sampled = base.data[indices].astype(np.float32)  # convert to float32
    sampled /= 255.0  # scale to [0, 1]

    # Save to .npy
    np.save(args.output, sampled)
    print(f"Saved scaled subsample to {args.output} with shape {sampled.shape} and dtype {sampled.dtype}")


if __name__ == "__main__":
    main()
