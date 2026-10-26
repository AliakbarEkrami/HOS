# Collaborator handoff

## Current status

- The Python implementation now includes a frozen `mitdb_pvc_v1` public-data
  experiment alongside the original `extract`, `stats` and `evaluate` commands.
- Its research target is **LOW_PVC** (at most 1% V beats) versus **HIGH_PVC**
  (at least 5% V beats) in 15-minute MIT-BIH segments from unseen subjects.
  These are operational labels, not clinical diagnoses; LOW_PVC does not mean
  healthy. It does not reproduce the original paper's five disease groups.
- Both annotated/reference and ECG-detector branches produce seven HOS and eight
  baseline features. Grouped nested evaluation, abstention/failure accounting,
  bootstrap intervals and a Persian report are implemented. QRS errors are
  evaluated independently against reference annotations.
- The source paper is included with attribution. The original 330-record cohort
  is unavailable to the project, so its exact results have not been reproduced.
- Software fixtures and numerical tests validate behavior. A scientific result
  must come from a completed, hashed real-data run; no expected accuracy or
  planned cohort count should be reported as an observed result.

## Reproduce the public-data experiment

Use Python 3.10+ and run these commands from the repository root (`HOS/`):

```shell
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python -m pytest hos_hrv_reproduction/tests -q
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-prepare --download --data-dir hos_hrv_reproduction/data/public/mitdb --output-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-extract --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-evaluate --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
python hos_hrv_reproduction/scripts/run_pipeline.py mitdb-report --run-dir hos_hrv_reproduction/results/mitdb_pvc_v1
```

The fixed version 1.0.0 download uses official SHA256 hashes and retains the 144
signal/header/reference-annotation files (about 90 MiB). It checks existing files
so a partial download can resume. Data and generated results are ignored by Git.
Do not publish raw waveforms with a code contribution.

Prepare into a **new run directory** for every experiment; existing prepared runs
are protected. The default protocol ID locks the scientific rules. A deliberate
change requires a renamed protocol ID, saved settings and a separate run.

Check `audit.json` first: it contains measured class/subject counts and reasons
for exclusion. Fewer than five subjects per class or infeasible grouped folds
leaves evaluation at `insufficient_data`. Do not relax the rules after viewing
test performance. The primary metric population contains all QC-accepted segments;
failures in either branch remain ABSTAIN and count as incorrect, with coverage
shown separately. Shared-success-only comparisons are explicitly secondary.

## Run artifacts and scientific checks

- `protocol.json`, `paper_config.json`, `source_files.json`, `provenance.json`:
  exact rules, processing assumptions, SHA256 hashes, code/package versions.
- `cohort.csv`, `manifest.csv`, `audit.json`: every candidate, accepted segments,
  beat/rhythm/quality decisions, exclusions and actual class/subject sizes.
- `features_reference.csv`, `features_detector.csv`, `extraction.json`,
  `qrs_metrics.csv`: features/status/errors from both branches, QRS TP/FP/FN,
  sensitivity/PPV and one-to-one 150 ms timing matches.
- `splits.json`, `predictions.csv`, `evaluation.json`: subject-disjoint outer and
  inner folds, every OOF prediction/ABSTAIN, parameters, bootstrap intervals and
  paired comparisons. Subject identity is shared by records 201 and 202.
- `report_fa.md`, `figures/`, `examples/`: Persian Markdown interpretation, signal
  examples, HOS distributions, performance and confusion plots. The report only
  reads saved results and never retrains models.

Review R-peak errors before attributing detector-branch performance to HOS.
Ensure splits have no shared subject, features contain no PVC count/burden or ID,
and failures were not silently dropped. Preserve intervals, sample size and
coverage when describing results. Intervals are conditional on fixed OOF fits.
Differences from simple features may be small or absent and remain valid findings.
Bootstrap reports retain requested and usable draws; draws containing only one
class are skipped, so 2000 attempted draws need not yield 2000 usable samples.

## Where to start

- `hos_hrv_reproduction/docs/mitdb_pvc_protocol.md`: locked experiment and limits.
- `hos_hrv_reproduction/docs/paper_audit.md`: source methods and unresolved paper choices.
- Package `README.md`: root-relative commands and artifact descriptions.
- `mitdb_protocol.py`: verified downloader, burden/rhythm rules, MLII/quality checks.
- `mitdb_features.py`: reference/detector branches, baseline features, QRS matching.
- `mitdb_evaluation.py`: nested grouped tuning, uncertainty, failure accounting.
- `mitdb_report.py`: saved-artifact Persian report and scientific figures.
- `mitdb_cli.py`, `mitdb_common.py`: interfaces, JSON/CSV and provenance helpers.
- `config.py`, `signal.py`, `bispectrum.py`: shared numerical implementation.
- `classifier.py`, `pipeline.py`: retained original-paper-style workflow.
- `tests/`: boundary, leakage, numerical, artifact and offline integration tests.

## Appropriate next work

1. Run the full public cohort and inspect exact eligibility/quality exclusions and
   per-record QRS errors; inspect examples, not just aggregate metrics.
2. Independently review estimator normalization, diagonal/index/centroid units,
   six-block small-ensemble uncertainty and preprocessing assumptions against
   the paper. Record any methodological variant under a new protocol ID.
3. Interpret subject-bootstrap uncertainty and HOS-versus-baseline comparisons
   without claiming clinical performance or exact reproduction.
4. If the original data become available, keep their consent/access conditions,
   labels, split details and paper experiment separate from the public PVC study.

Reproduce a failing case before modifying an algorithm. Run the full test suite
after changes, include meaningful new regression coverage, and use a branch/pull
request as described in CONTRIBUTING.md. Do not count segments or bootstrap draws
as new patients, invent CHB/SSS labels, or compare this binary task's accuracy to
the original five-class paper as if they were the same experiment.
