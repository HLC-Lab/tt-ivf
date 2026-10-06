import h5py
import numpy as np
import sys
import os
from pathlib import Path


def export_to_bin(hdf5_path):
    if not os.path.exists(hdf5_path):
        print(f"Error: {hdf5_path} not found.")
        raise FileNotFoundError(hdf5_path)

    name = os.path.splitext(os.path.basename(hdf5_path))[0]
    print(f"Exporting {hdf5_path} to binary...")

    with h5py.File(hdf5_path, "r") as f:
        # Standard ANN-Benchmarks HDF5 layout
        for key in ["train", "test", "neighbors", "distances"]:
            if key in f:
                if key == "neighbors":
                    data = np.array(f[key]).astype(np.uint32)
                else:
                    data = np.array(f[key]).astype(np.float32)

                if key == "test":
                    out_name = f"{name}_queries.bin"
                else:
                    out_name = f"{name}_{key}.bin"

                out_name = Path(hdf5_path).parent / out_name
                n, d = data.shape
                with open(out_name, "wb") as bin_f:
                    # Header: [N][D]
                    bin_f.write(np.array([n, d], dtype=np.uint32).tobytes())
                    # Data
                    bin_f.write(data.tobytes())
                print(f"  Created {out_name} (N={n}, D={d})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 export_hdf5.py <dataset_name.hdf5>")
    else:
        export_to_bin(sys.argv[1])
