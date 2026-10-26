# HOS: ECG/HRV higher-order spectral analysis

Python implementation and reproducibility study of **Cardiac Health Diagnosis Using Higher Order Spectra and Support Vector Machine** by Chua et al. (2009), for biomedical signal processing coursework.

We implement the published ECG-to-HRV, bispectrum, seven-feature, and SVM pipeline with explicit assumptions, and a versioned **LOW_PVC versus HIGH_PVC** experiment on public MIT-BIH data. The original 330-record dataset is unavailable to this project. No published accuracy has been reproduced on the original cohort.

## Public-data PVC experiment

The question is whether seven HOS features distinguish 15-minute segments with
**at most 1% V beats** from segments with **at least 5% V beats** in new subjects.
These are experimental burden bands, not clinical diagnoses. The 1–5% gap is
excluded; LOW_PVC does not mean healthy. This does not recreate the paper's five
disease classes.

The frozen [mitdb_pvc_v1 protocol](hos_hrv_reproduction/docs/mitdb_pvc_protocol.md)
defines rhythm/purity and waveform quality checks, MLII selection, paced-record
exclusion, the shared patient for 201/202, reference/detector branches, eight
baseline features, seven HOS features, and their combination. Nested 3x3 subject
grouped SVM evaluation prevents subjects from appearing on both sides of a split.
2000 subject bootstrap draws quantify conditional OOF uncertainty. Failed
processing is retained and counts as an incorrect end-to-end prediction.

After installing below, run these commands **from the repository root**:

```shell
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-prepare --download --data-dir hos_hrv_reproduction/data/public/mitdb --output-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-extract --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-evaluate --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-report --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
```

The download uses the fixed public version 1.0.0 and verifies official SHA256
hashes. It downloads 144 signal/header/annotation files (about 90 MiB).
Data and generated runs are ignored by Git. Downloads resume by checking existing
hashes. For a second experiment, use a new output directory. A completed run
contains `report_fa.md`, figures, eligibility audit, both feature tables, QRS
metrics, frozen settings, input hashes, subject splits and OOF predictions.
Installed aliases `mitdb-prepare`, `mitdb-extract`, `mitdb-evaluate`, `mitdb-report`
provide the same commands. The original paper commands below remain available.

The [completed real-data run](hos_hrv_reproduction/docs/experiments/mitdb_pvc_v1/README.md)
includes a [Persian report](hos_hrv_reproduction/docs/experiments/mitdb_pvc_v1/report_fa.md),
figures and archived predictions/settings. It retained **38 segments from 22
subjects**. Reference-timing balanced accuracy was **89.74% for HOS** versus
**95.83% for baseline features**; detector-timing HOS reached **96.15%**. These
small-cohort results do not establish consistent HOS superiority. The archive
contains conditional bootstrap intervals and local detector errors, including a
segment with only 65.16% QRS sensitivity. All **103 offline tests** pass.

An illustrated Persian work report is available as a [six-page PDF](output/pdf/HOS_PVC_Report_FA.pdf),
with its [LaTeX source and figure assets](output/pdf/hos_pvc_report_fa/main.tex).

## Paper and data

The source paper is included at [Sources/hos_of_hrv.pdf](Sources/hos_of_hrv.pdf). Citation and its separate CC BY-NC 3.0 license are documented in [Sources/README.md](Sources/README.md).

The paper describes 10–15 minute ECG recordings digitized at 320 Hz from Kasturba Medical College Hospital, Manipal, India: NSR 183, PVC 37, CHB 42, SSS 43, CHF 25. We have not obtained these recordings. Public PhysioNet datasets can test the method but differ in patients and diagnostic labels; they cannot establish exact reproduction of the reported results.

## Implemented workflow

1. Read ECG, RR intervals in seconds, or uniformly sampled HRV from a manifest.
2. For ECG input, optionally resample to 320 Hz, filter and detect R peaks with an offline Pan-Tompkins-style detector.
3. Convert RR intervals to heart rate and interpolate at 4 Hz.
4. Estimate the bispectrum using 1024-sample blocks with 512-sample overlap.
5. Extract `P1`, `P2`, `H1`, `H2`, `H3`, `f1m`, and `f2m`.
6. Run one-way ANOVA, exploratory pairwise Welch tests, and repeated one-vs-rest RBF-SVM evaluation with `C=190`.

Defaults for unspecified paper details are exposed in `PaperConfig`. See the [paper audit](hos_hrv_reproduction/docs/paper_audit.md) for entropy normalization, FFT/window assumptions, detector differences, scaling, gamma, and split limitations. This is a research implementation, not a clinical diagnostic tool.

## Clone and install

Use Python 3.10 or newer. Run these commands from the cloned repository root:

```shell
git clone https://github.com/AliakbarEkrami/HOS.git
cd HOS
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```shell
# Linux/macOS
source .venv/bin/activate
```

Then install and test:

```shell
python -m pip install -r requirements.txt
python -m pytest hos_hrv_reproduction/tests -q
```

The root requirements file installs the local package and development dependencies. Runtime dependencies are NumPy, SciPy, scikit-learn, WFDB and Matplotlib; pytest is used for development. A package-only install is also available with `python -m pip install -e ./hos_hrv_reproduction`.

## Run experiments

Run pipeline commands from the package folder so the example manifest paths resolve correctly:

```shell
cd hos_hrv_reproduction
```

Create `data/manifest.csv` based on [data/manifest.example.csv](hos_hrv_reproduction/data/manifest.example.csv), with real files and verified labels. Each row represents one recording. Put raw signals in `data/raw/`; they are ignored by Git. Example:

```csv
record_id,subject_id,path,label,kind,fs,mains_hz,channel
record_001,subject_001,data/raw/record_001.npy,NSR,ecg,320,50,0
```

```shell
python scripts/run_pipeline.py extract --manifest data/manifest.csv --output results/features.csv
python scripts/run_pipeline.py stats --features results/features.csv --output results/statistics.json
python scripts/run_pipeline.py evaluate --features results/features.csv --output results/evaluation.json --repeats 5 --test-size 100
```

The `--test-size 100` default mirrors the 500 total test predictions implied by the paper's five-repeat matrix. It requires a sufficiently large dataset in every class. For a smaller five-class dataset, use `--test-size 0` to use the per-class one-third holdout pool. The paper's exact split indices are unknown.

## Public data options

The downloader defaults to one example record and supports `--all`:

```shell
python scripts/download_physionet.py --database qtdb --output data/public/qtdb
python scripts/download_physionet.py --database mitdb --output data/public/mitdb
```

Supported databases: `qtdb`, `mitdb`, `incartdb`, `nsrdb`, `chfdb`. QT has 15-minute recordings; MIT-BIH Arrhythmia is useful for annotated PVC and QRS experiments. NSR and CHF databases offer separate clinical groups. They do not collectively provide a validated substitute for all five original classes. Downloading does not automatically create diagnostic labels or a training manifest. See the [dataset comparison](hos_hrv_reproduction/README.md#public-comparators).

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow, tests, and scientific requirements, and [HANDOFF.md](HANDOFF.md) for current status and next tasks. Collaborators can clone this repository and submit pull requests. Direct push access requires the repository owner to invite their GitHub account.

```text
HOS/
  Sources/hos_of_hrv.pdf             Reference paper
  hos_hrv_reproduction/
    src/hos_hrv_reproduction/        Python implementation
    scripts/                        Extraction, statistics, evaluation, downloads
    data/                           Manifest template and local data instructions
    docs/paper_audit.md              Methods audit and ambiguities
    tests/                          Numerical regression tests
    pyproject.toml                  Package metadata and dependencies
  requirements.txt                  Root installation entry point
  CONTRIBUTING.md                    Collaboration workflow
  HANDOFF.md                         Status and next steps
```
