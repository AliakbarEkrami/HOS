# Observed MIT-BIH PVC experiment: mitdb_pvc_v1

[Persian report and figures](report_fa.md). This is a real public-data run, not the synthetic software fixture.

## Population and validation

All 144 input files matched the official version 1.0.0 SHA256 sums. Of 96 candidate segments, 38 passed the frozen annotation and waveform rules: 26 LOW_PVC segments from 15 subjects and 12 HIGH_PVC segments from 7 subjects (22 distinct subjects total). Both timing branches extracted features for all 38 segments.

Nested 3x3 subject-grouped validation uses seed 20261026, with train-only standardization and tuning. The 304 OOF predictions cover eight source/model combinations. All 2000 subject-bootstrap draws contained both classes. Intervals condition on these fitted OOF models; they do not repeat the entire training process.

| Timing source | Model | Balanced accuracy | 95% CI | HIGH sensitivity | LOW specificity | HIGH F1 |
| --- | --- | --- | --- | --- | --- | --- |
| reference | majority | 50.00% | 50.00% to 50.00% | 0.00% | 100.00% | 0.00% |
| reference | baseline | 95.83% | 83.33% to 100.00% | 91.67% | 100.00% | 95.65% |
| reference | hos | 89.74% | 78.92% to 100.00% | 83.33% | 96.15% | 86.96% |
| reference | combined | 93.91% | 81.41% to 100.00% | 91.67% | 96.15% | 91.67% |
| detector | majority | 50.00% | 50.00% to 50.00% | 0.00% | 100.00% | 0.00% |
| detector | baseline | 87.82% | 72.59% to 98.28% | 83.33% | 92.31% | 83.33% |
| detector | hos | 96.15% | 89.47% to 100.00% | 100.00% | 92.31% | 92.31% |
| detector | combined | 96.15% | 89.47% to 100.00% | 100.00% | 92.31% | 92.31% |

The primary reference-timing result does not show an HOS advantage: HOS minus baseline is -6.09 percentage points (95% CI -19.86 to +11.54). In the detector branch the difference is +8.33 points (0.00 to +21.45). With only 22 subjects and no external validation, these results do not establish consistent HOS superiority. They do not reproduce the original paper's five clinical classes or its accuracy.

Pooled QRS matching: TP=40388, FP=231, FN=522; sensitivity=98.724%, PPV=99.431%, matched timing MAE=3.180 ms. Local errors matter: segment 228_0000_0900 has only 65.16% sensitivity. See the report's worst-segment table and qrs_metrics.csv. Higher classifier accuracy in a detector branch is not evidence of more accurate beat measurement.

## Code and environment

- Feature/model run commit: `3c034c87c612941d622964803bc592d772dedaf6` (clean working tree).
- Report renderer commit: `66285b7db5adb8c976ebfe0895ae639085847170`.
- Interpreter used: `3.12.10`.
- Exact scientific package versions and input hashes are in evaluation.json, extraction.json and provenance.json.
- Run the four commands in the repository README in a fresh output directory to reproduce.

## Archive scope

Raw waveforms and annotation files are excluded from Git, as are the NPZ example arrays. Source files can be obtained from [MIT-BIH Arrhythmia Database 1.0.0 on PhysioNet](https://physionet.org/content/mitdb/1.0.0/), subject to its attribution and license terms. The signal plots here are derived illustrations from those public records.

This directory is a readable results snapshot, not an executable run folder. Features, predictions, splits, settings, input hashes and evaluation JSON are copied unchanged. cohort_summary.csv and accepted_segments.csv omit the machine-specific path column from the original cohort.csv and manifest.csv; report links point to those portable summaries. Frozen manifest hashes refer to the original local run. Recreate a manifest with mitdb-prepare before rerunning; do not evaluate this archive directly.

Line ending conversion is disabled in this archive to preserve byte hashes across platforms.

ARTIFACTS_SHA256.json fingerprints the archived copies, separately from the original run input fingerprints. Original local report SHA256: `708d6f7050366be023914218d6a705ebda256e7e0075a2194959e022a2ccb02f`.
