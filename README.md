# HOS: ECG/HRV higher-order spectral analysis

Python implementation and reproducibility study of **Cardiac Health Diagnosis Using Higher Order Spectra and Support Vector Machine** by Chua et al. (2009), for biomedical signal processing coursework.

We are rebuilding the published ECG-to-HRV, bispectrum, seven-feature, and SVM pipeline with explicit assumptions, then planning experiments on public ECG datasets. The original 330-record dataset is unavailable to this project. No published accuracy has been reproduced on the original cohort.

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

The root requirements file installs the local package and development dependencies. Runtime dependencies are NumPy, SciPy, scikit-learn, and WFDB; pytest is used for development. A package-only install is also available with `python -m pip install -e ./hos_hrv_reproduction`.

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
