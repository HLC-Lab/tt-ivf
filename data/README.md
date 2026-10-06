# Benchmark inputs

```text
data/
├── datasets/
│   ├── glove-100-angular.hdf5
│   ├── glove-100-angular_train.bin
│   ├── glove-100-angular_queries.bin
│   ├── glove-100-angular_neighbors.bin
│   └── glove-100-angular_distances.bin
└── centroids/
    ├── centroids-glove-100-angular-512.npy
    └── centroids-glove-100-angular-512.bin
```

All three TT benchmarks use this layout; `--dataset` remains the dataset name.
`ANN_DATA_DIR` selects another root containing `datasets/` and `centroids/`.
FAISS reads HDF5 here and writes trained centroids to `centroids/`. Conversion
helpers write beside their input files, even when run from another directory.
The shared scripts are `tools/export_hdf5.py` and `tools/convert_centroids.py`.
Inputs are distributed separately and ignored by Git.

Link existing files in with `python3 tools/link_data.py /path/to/old/files`;
it refuses to replace a different file already at the destination.
