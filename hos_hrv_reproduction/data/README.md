# Data directory

The original Manipal/Kasturba ECG recordings used in Chua et al. (2009) are not distributed with the paper and were not found in a public repository. Do not put identifiable clinical data in a public Git repository.

Use a manifest with one row per record:

```csv
path,label,fs,mains_hz,channel
data/raw/record_001.npy,NSR,320,50,0
```

`path` is relative to the project directory. `label` must be one of `NSR`, `PVC`, `CHB`, `SSS`, or `CHF` for the paper-like evaluation. `fs` defaults to 320 Hz, `mains_hz` defaults to 50 Hz, and `channel` defaults to 0.

The public PhysioNet downloader stores comparator datasets under `data/public/`; those datasets have different labels and are not an exact replication of the article.
