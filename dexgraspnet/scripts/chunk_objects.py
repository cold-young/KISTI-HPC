#!/usr/bin/env python3
"""
Phase 3: Split object code list into N chunks for multi-GPU parallel processing.

Usage:
    python dexgraspnet/scripts/chunk_objects.py \
        --object_codes object_codes.txt \
        --n_chunks 8 \
        --output object_chunks.txt

Output format: one chunk per line, space-separated object codes.
"""

import argparse
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split object codes into GPU chunks")
    parser.add_argument("--object_codes", type=str, required=True,
                        help="Path to object_codes.txt (one code per line)")
    parser.add_argument("--n_chunks", type=int, default=8,
                        help="Number of chunks (typically = number of GPUs)")
    parser.add_argument("--output", type=str, default="object_chunks.txt",
                        help="Output file (one chunk per line, space-separated)")
    args = parser.parse_args()

    with open(args.object_codes) as f:
        codes = [line.strip() for line in f if line.strip()]

    n = len(codes)
    chunk_size = math.ceil(n / args.n_chunks)
    chunks = [codes[i:i + chunk_size] for i in range(0, n, chunk_size)]

    with open(args.output, "w") as f:
        for chunk in chunks:
            f.write(" ".join(chunk) + "\n")

    print(f"Split {n} objects into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} objects")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
