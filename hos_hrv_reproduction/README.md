# HOS/HRV methods and a public-data PVC experiment

The current public-data experiment asks whether seven bispectral features of a
15-minute heart-rate series distinguish **LOW_PVC** (at most 1% `V` beats) from
**HIGH_PVC** (at least 5% `V` beats) in previously unseen subjects. These are
operational burden bands, not diagnoses. LOW_PVC does not mean a healthy subject.
The intermediate 1–5% band is excluded before modeling.

This is an adaptation of the signal-processing method in Chua et al. (2009), not
a reproduction of its five clinical groups or reported classification accuracy.
The older paper-style commands remain available for supplied original data.

## Run the versioned MIT-BIH experiment

Use Python 3.10 or newer. The following commands run **from the repository root**
(`HOS/`, one level above this README):

```shell
python -m pip install -e "./hos_hrv_reproduction[dev]"
python -m pytest hos_hrv_reproduction/tests -q
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-prepare --download --data-dir hos_hrv_reproduction/data/public/mitdb --output-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-extract --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-evaluate --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-report --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
```

Installation includes NumPy, SciPy, scikit-learn, WFDB and Matplotlib; `dev` adds
pytest. If working inside this package directory instead, install with
`python -m pip install -e ".[dev]"` and remove `hos_hrv_reproduction/` from the
command paths above. Installed aliases `mitdb-prepare`, `mitdb-extract`,
`mitdb-evaluate` and `mitdb-report` provide the same interfaces. `--help` describes
the arguments for each stage.

`mitdb-prepare --download` retrieves the fixed public MIT-BIH Arrhythmia 1.0.0
version, including all 48 signal/header/reference-annotation triplets (144 files,
about 90 MiB). It verifies the official SHA256 checksums and reuses verified
existing files. Omitting `--download` uses files already present in `--data-dir`.
Raw data and generated results are ignored by Git; do not commit patient signals.

Use a **new output directory for each experiment**. The preparation stage rejects
an existing prepared run, preserving its cohort and provenance. Extraction saves
its `PaperConfig` and input hashes; keep the saved settings and artifacts together.
Do not edit feature tables, labels or thresholds after inspecting model results.

## Frozen cohort and analysis

The full scientific protocol is in
[`docs/mitdb_pvc_protocol.md`](docs/mitdb_pvc_protocol.md).

- The `mitdb_pvc_v1` rules are frozen: two non-overlapping 15-minute segments per
  record, MLII selected by name, exclusion of paced records 102/104/107/217,
  common subject ID for records 201 and 202, beat/rhythm eligibility, and waveform
  quality checks. Every candidate remains in `cohort.csv` with inclusion status
  and explicit reasons; `manifest.csv` contains accepted segments.
- The **reference** branch uses annotated beat times to assess the HOS features.
  The **detector** branch detects R peaks on the full ECG record after filtering
  and resampling, then extracts the same segment features. PVC beats are retained.
  QRS matching is one-to-one within 150 ms and reports missed/extra beats and
  timing error.
- RR intervals become `HR=60/RR`, interpolated at 4 Hz without extrapolation.
  Bispectra average 1024-sample Hann-windowed blocks with 512-sample overlap;
  at least five blocks are required. Each segment yields one feature vector,
  never multiple independent patients.
- Seven HOS features (`P1, P2, H1, H2, H3, f1m, f2m`) are compared with eight
  simple interval/power-spectrum features and their combination. The baseline
  includes mean/SD of RR, RMSSD_RR, pRR50, log VLF/LF/HF power, and PSD entropy.
  These retain ectopy and are not normal-to-normal autonomic HRV indices.
- Majority prediction and three RBF-SVM feature sets share nested 3x3
  subject-disjoint folds. Standardization and tuning run inside training only;
  `C={1,10,100,190}`, `gamma={scale,0.01,0.1}`, balanced class weights and seed
  20261026 are fixed. Feasibility, not performance, determines split selection.
- The primary population is **all accepted QC segments**. Extraction failures in
  either branch remain `ABSTAIN`, count as incorrect end-to-end predictions, and
  stay in metric denominators. Shared-success comparisons are reported separately.
  Fewer than five accepted subjects in either class or infeasible grouped folds
  produces `insufficient_data`; thresholds are not relaxed to improve accuracy.
- Balanced accuracy, HIGH_PVC sensitivity/F1, LOW_PVC specificity and confusion
  matrices are saved with 95% intervals from 2000 attempted subject-cluster
  bootstrap draws. One-class draws are skipped; requested and usable counts are saved.
  Intervals are conditional on the fixed out-of-fold predictions and do not
  establish clinical validity or guarantee HOS improves upon simple features.

The default `mitdb_pvc_v1` protocol cannot be modified while retaining that ID.
For a methodological change or a software fixture, use a renamed protocol ID
and a new output directory. Such runs are flagged as modified experiments.

## Artifacts and implementation map

Each completed run includes:

| Artifact | Purpose |
|---|---|
| `protocol.json`, `paper_config.json` | Frozen cohort rules and explicit processing assumptions |
| `source_files.json`, `provenance.json` | Input hashes, dataset version, code/package versions |
| `cohort.csv`, `manifest.csv`, `audit.json` | All candidates, accepted segments, counts and exclusions |
| `features_reference.csv`, `features_detector.csv`, `extraction.json` | Seven HOS/eight baseline features, status/errors and extraction metadata |
| `qrs_metrics.csv` | Per-segment beat matching TP/FP/FN and timing errors |
| `splits.json`, `predictions.csv`, `evaluation.json` | Subject splits, every OOF prediction/abstention, metrics and comparisons |
| `report_fa.md`, `figures/`, `examples/` | Persian Markdown report and scientific figures/example arrays |

`mitdb_protocol.py` owns download verification and eligibility;
`mitdb_features.py` owns both beat-time paths and feature extraction;
`mitdb_evaluation.py` owns grouped splits/tuning/uncertainty;
`mitdb_report.py` renders the saved run without retraining. `mitdb_cli.py` exposes
the four commands. The `mitdb_common.py` helpers save artifact provenance.

Tests include exact 1%/5% boundary labels, rhythm intervals, shared 201/202
subject identity, beat-time conversion, bispectrum numerical checks, feature
allowlists, absence of subject overlap, deterministic splits, failure accounting,
Persian report generation and an offline synthetic end-to-end fixture. Synthetic
outputs validate software behavior and are not biomedical evidence.

No real-data performance is promised by this implementation. Inspect the generated
`audit.json` before interpreting a run, and use its actual `evaluation.json` and
`report_fa.md` rather than treating planned cohort counts as measured results.

## Original-paper implementation

This project is a from-scratch Python implementation of the pipeline in:

> Chua, K. C., Chandran, V., Acharya, R. U., and Lim, C. M. (2009), *Cardiac Health Diagnosis Using Higher Order Spectra and Support Vector Machine*, The Open Medical Informatics Journal, 3, 1-8. DOI: [10.2174/1874431100903010001](https://doi.org/10.2174/1874431100903010001).

The reference article is kept in the sibling [`Sources/`](../Sources/README.md) directory. The local HOSA toolbox guide is separate and is not published with this repository.

## Reproducibility status

The original paper's 330 ECG records are not attached to the article and no public download or repository for that exact collection was found. The paper says that the recordings were collected by the authors at Kasturba Medical College Hospital, Manipal, India, from more than 330 patients, digitized at 320 samples/s, and divided into NSR (183), PVC (37), CHB (42), SSS (43), and CHF (25). Those counts sum to 330.

Consequently, an exact rerun of the published confusion matrix is currently blocked. The implementation provides input support for original recordings if obtained from the authors or hospital, but still needs real-data validation and resolution of methodological ambiguities. Public PhysioNet records can validate the method on different cohorts and must not be presented as an exact replication.

The article also leaves several details unstated. They are exposed in `PaperConfig` rather than hidden in the code:

* the median-filter window used for baseline extraction;
* the exact Tompkins detector thresholds and refractory period;
* how RR intervals were interpolated to 4 Hz;
* FFT normalization and the treatment of zero bispectral bins;
* the SVM RBF gamma value;
* the random seed and the exact train/test indices.

The default settings follow the paper where it is explicit and use conservative, documented defaults elsewhere. Every assumption is recorded in `docs/paper_audit.md`.

## Layout

```text
hos_hrv_reproduction/
  data/                         # manifests and user-supplied recordings
  docs/paper_audit.md           # paper extraction and unresolved details
  scripts/run_pipeline.py       # ECG -> HRV -> bispectrum -> features
  scripts/download_physionet.py   # optional public comparator downloader
  src/hos_hrv_reproduction/     # implementation
  tests/                        # numerical smoke tests
  pyproject.toml
```

Raw ECG files are intentionally not copied into this project. Put them under `data/` and describe them in a CSV manifest with columns:

```csv
path,label,fs,mains_hz,channel
data/raw/record_001.npy,NSR,320,50,0
```

The loader accepts NumPy `.npy`, numeric comma-delimited `.csv`/`.txt`, tab-delimited `.tsv`, and WFDB record basenames or headers. WFDB is included in the package dependencies.

## Install

```powershell
git clone https://github.com/AliakbarEkrami/HOS.git
cd HOS/hos_hrv_reproduction
python -m pip install -e .
```

## Run feature extraction

```powershell
python scripts/run_pipeline.py extract `
  --manifest data/manifest.csv `
  --output results/features.csv
```

Each record is converted to the paper's 320-Hz ECG input when its native rate differs, filtered, passed through a Pan-Tompkins-style QRS detector, converted to heart rate, linearly resampled to 4 Hz, split into 1024-sample blocks with 512-sample overlap, and summarized by the seven paper features:

`P1, P2, H1, H2, H3, f1m, f2m`.

## Run feature statistics

```powershell
python scripts/run_pipeline.py stats `
  --features results/features.csv `
  --output results/statistics.json
```

This writes one-way ANOVA results and pairwise Welch t-tests. The paper reports ANOVA followed by a t-test but does not specify the t-test variant, so Welch's version is an explicit, documented assumption.

## Run the paper-like SVM experiment

```powershell
python scripts/run_pipeline.py evaluate `
  --features results/features.csv `
  --output results/evaluation.json `
  --repeats 5 `
  --test-size 100 `
  --seed 20261004
```

The paper reports an RBF SVM with `C=190`, approximately two-thirds training and one-third testing, repeated five times. The reported aggregate confusion matrix contains 500 test predictions, so this CLI exposes `--test-size 100` to mirror that total. The paper's seed and exact indices were not published.

## Public comparators

The closest openly downloadable resources are not the original cohort. This table ranks them for this project:

| Dataset | Public data | Recording specification | Useful overlap | Main mismatch |
|---|---|---|---|---|
| [QT Database](https://physionet.org/content/qtdb/1.0.0/) | Yes | 105 two-lead recordings, 15 minutes; sampling rate is record-specific | Closest duration; waveform annotations | Not the paper's five diagnostic classes; sampling is not uniformly 320 Hz |
| [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) | Yes | 48 two-lead recordings, 30 minutes, 360 Hz | Beat annotations and PVC/arrhythmia examples | Different cohort, duration, sampling rate, and labels |
| [St Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) | Yes | 75 twelve-lead recordings, 30 minutes, 257 Hz | Holter recordings, ectopy, conduction abnormalities | Only a small number of patients have sinus-node dysfunction or AV block; no CHF/NSR/PVC class design matching the paper |
| [MIT-BIH NSR](https://physionet.org/content/nsrdb/1.0.0/) | Yes | 18 long-term two-lead recordings, 128 Hz | Normal-sinus reference group | No disease classes and much longer recordings |
| [BIDMC CHF](https://physionet.org/content/chfdb/1.0.0/) | Yes | 15 two-lead recordings, about 20 hours, 250 Hz | CHF group | CHF only; no matching NSR/PVC/CHB/SSS cohort |
| [European ST-T](https://physionet.org/content/edb/1.0.0/) | Yes | 90 two-lead recordings, 2 hours, 250 Hz | Ambulatory ECG and annotated rhythm/conduction events | Designed for ST/T changes, not the paper's five classes |

All can be converted to the paper's processing input by selecting a 10–15 minute segment and resampling the ECG to 320 Hz, but resampling does not make them equivalent to the original cohort. The optional downloader retrieves small examples or all files from these public comparators; it does not claim they reproduce the paper's dataset.

```powershell
python scripts/download_physionet.py --database mitdb --output data/public/mitdb
# closest-duration option:
python scripts/download_physionet.py --database qtdb --output data/public/qtdb
```

## Existing software found during the audit

No public repository containing the article's complete data-to-results workflow was found. General-purpose implementations do exist, including the [HOSA MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/3013-hosa-higher-order-spectral-analysis-toolbox), [HOSA_Octave](https://github.com/synergetics/HOSA_Octave), [higher-spectrum](https://github.com/synergetics/spectrum), and [PyBispectra](https://github.com/braindatalab/PyBispectra). This project does not depend on those implementations for the paper-specific feature extraction; its estimator is written here so every choice can be inspected.

## Expected paper targets

Once the original records and split details are available, the validation target is the article's reported aggregate matrix (rows are actual classes, columns are predicted classes):

```text
          NSR  PVC  CHB  SSS  CHF
NSR       255   17   13    0    5
PVC        12   37    0    1    0
CHB         5    0   48    0    7
SSS         0    1    0   64    0
CHF         4    0    0    0   31
```

The resulting class accuracies reported by the paper are 87.93%, 74.00%, 80.00%, 98.46%, and 88.57% (mean 85.79%). The binary abnormal-vs-normal sensitivity/specificity are 90.00% and 87.93%.
