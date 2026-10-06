import numpy as np
import sys
import os
from pathlib import Path


def convert_npy_to_bin(npy_path):
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        raise FileNotFoundError(npy_path)

    out_path = Path(npy_path).with_suffix(".bin")
    print(f"Converting {npy_path} to {out_path}...")

    # 1. Load the file with pickle allowed
    raw_data = np.load(npy_path, allow_pickle=True)

    # 2. Safely unpack the pickled object into a strict 2D array
    if raw_data.dtype == object:
        # Sometimes NumPy pickles a list into a 0-dimensional array. We extract it using .item()
        if raw_data.shape == ():
            raw_data = raw_data.item()

        # Stack the internal lists/objects into a clean, 2D float32 array
        data = np.vstack(raw_data).astype(np.float32)
    else:
        # If it's already a standard array, just cast it
        data = raw_data.astype(np.float32)

    n, d = data.shape

    with open(out_path, "wb") as f:
        # Header: [N][D] as uint32
        f.write(np.array([n, d], dtype=np.uint32).tobytes())
        # Data
        f.write(data.tobytes())

    print(f"  Success: Created {out_path} (N={n}, D={d})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_centroids.py <file.npy>")
    else:
        convert_npy_to_bin(sys.argv[1])
