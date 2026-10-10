# Collaborator handoff

## Current status

- Python package, three pipeline commands, public example downloader, and numerical tests are implemented.
- Source paper is included with attribution. No original clinical dataset is included or accessible to the project.
- Synthetic checks validate parts of the implementation; real ECG detection accuracy and paper-level results have not been validated.
- Exact replication remains dependent on the original records and unspecified preprocessing, FFT/entropy, SVM gamma/scaling, and split details.

## Next work

1. Review equations against the source paper and verify entropy normalization, diagonal indices, windowing, and bispectral centroid definitions.
2. Evaluate the QRS detector against reviewed reference annotations on MIT-BIH. Report missed/extra beats and timing errors before using detector-derived HRV for classification.
3. Add explicit segment-selection and curated R-peak inputs to the CLI and manifest. The Python `process_record` helper already accepts `peaks_path` with `peaks_units="seconds"` (default) or `"samples"` at the original ECG rate.
4. Add patient-grouped evaluation and persisted train/test indices before processing multiple recordings or segments per subject.
5. Add reviewed public-cohort manifests. Do not invent CHB/SSS labels or count segments as new patients.
6. Add reproducible scientific figures and comparisons to paper Tables 2–4 once a cohort is defined.

## Where to start

- `hos_hrv_reproduction/docs/paper_audit.md`: source methods and unresolved choices.
- `hos_hrv_reproduction/src/hos_hrv_reproduction/config.py`: reported settings and assumptions.
- `signal.py` / `pipeline.py`: ECG-to-HRV processing.
- `bispectrum.py`: ensemble estimator and seven features.
- `classifier.py`: one-vs-rest SVM and metrics.
- `tests/`: data-independent regression tests.

Reproduce a failing case before modifying an algorithm. Record configuration, dataset version, subject IDs, seed, and split indices with every scientific experiment.
