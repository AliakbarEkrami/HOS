# MIT-BIH PVC burden experiment: mitdb_pvc_v1

This is an exploratory adaptation of the paper's seven bispectral features to
public MIT-BIH Arrhythmia 1.0.0. It does not reproduce the paper's five disease
groups or its unavailable 330 recordings. LOW_PVC does not mean healthy.

## Frozen population and target

- Download `.hea`, `.dat`, and `.atr` for all 48 records from the version-pinned
  [PhysioNet dataset](https://physionet.org/content/mitdb/1.0.0/). The downloader
  verifies every file against the official SHA256SUMS.txt; preparation checks
  those sums again when present. Raw data stays outside Git.
- Exclude paced records 102, 104, 107, 217. Records 201 and 202 share subject 201.
- Select MLII by channel name. Require the native 360 Hz header rate and two
  complete, nonoverlapping intervals [0,900) and [900,1800) seconds.
- Count only actual beat annotations `N L R A a J S V F e j E / f Q` in the
  denominator. Nonbeat markers (including rhythm and signal markers) never count
  as beats. `V` is the positive beat type; fusion `F` is not counted as `V`.
- LOW_PVC: V/beat_count <= 0.01; HIGH_PVC: V/beat_count >= 0.05.
  The gap is excluded. Exact rational comparisons preserve the boundaries.
- Require >=99% beats from N/L/R/V and >=95% duration in (N/(B/(T/(SBR.
  Rhythm state carries into each interval from earlier annotations. Unknown
  rhythm duration counts as disallowed. Bounds are half-open at native samples.
- Reject incomplete/nonfinite waveforms and an exactly constant run longer than
  2 seconds. Nonbeat annotation symbols are also retained for inspection; a
  marker alone does not automatically prove a quality defect.
- Log every exclusion before training. Fewer than five distinct subjects in
  either class produces an insufficient-data report, without relaxing rules.

The initial annotation-only audit predicts 26 LOW segments/15 subjects and
12 HIGH segments/7 subjects, before waveform quality control. Always use the
generated audit.json for the actual run.

The `mitdb_pvc_v1` dataclass rejects changes to frozen fields. Modified studies
must use another identifier, such as `mitdb_pvc_sensitivity`. Algorithm settings
are separately recorded in paper_config.json; changing them changes the experiment
and must be described in comparisons.

## Two timing sources, the same labels

Reference annotations give beat times in the main feature experiment. Their type
codes construct targets and eligibility; no count, burden, type code, record ID,
subject ID, or quality statistic enters a classifier.

The detector branch resamples the full selected ECG to 320 Hz, applies the existing
35 Hz low-pass, 0.3 Hz high-pass, 60 Hz notch and median baseline correction, then
uses the existing Pan-Tompkins-style detector. Filter order, notch Q, median width
and refractory time are saved. This is an implementation assumption, not a claim
that the unavailable paper detector was reproduced exactly. Detector output is
never corrected using reference annotations. One-to-one chronological matching
within 150 ms measures TP/FP/FN and absolute timing error in seconds, independent
of native/resampled sample indices.

RR is measured in seconds and assigned to its ending beat. HR=60/RR is linearly
interpolated on a global 4 Hz grid with contextual beats from the full record;
the segment's grid remains within [start,end), without extrapolation. No ectopy
removal, smoothing or NN-only correction is applied. RR baseline statistics use
intervals whose ending beat lies inside the segment.

## Features and explicit numerical assumptions

The direct bispectrum averages complex FFT products X(k1)X(k2)conj(X(k1+k2))
before taking magnitude. Each segment supplies one feature vector, not a set of
independent windows. Defaults: 1024 samples, overlap512, periodic Hann, block mean
subtraction, raw FFT scaling; require at least five blocks (usually six). Retain
k2<=k1 and k1+k2<=NFFT/2, including DC.

HOS predictors are P1, P2, H1, H2, H3, f1m, f2m. P1/P2 normalize Shannon entropy
of magnitude/magnitude-squared weights by log(number of retained bins). H1 is
the sum of natural log magnitudes, floored at 1e-12. H2 is the diagonal sum;
H3 uses the one-based diagonal bin index. Weighted centroid coordinates use
one-based FFT-bin indices, not Hz. Figures use Hz on their frequency axes.
These assumptions are exposed rather than asserted to be uniquely determined
by the paper. See paper_audit.md and the independent DFT tests.

Eight baseline predictors: mean RR, sample SD RR, RMSSD_RR, pRR50 (fraction,
not percent), natural-log VLF/LF/HF power, normalized PSD entropy. Welch uses
the same Hann/1024/512 settings. Power integrates the HR spectrum in bands
0.0033–0.04, 0.04–0.15, 0.15–0.4 Hz; entropy normalizes bins in 0.0033–0.4 Hz.
Because ectopy is preserved, these describe all-beat intervals, not normal-only
HRV. Combined models use all fifteen predictors.

## Evaluation

Compare majority, baseline RBF-SVM, HOS RBF-SVM, combined RBF-SVM with the same
outer three folds and inner three folds grouped by subject. Search for feasible
stratified group splits from seed 20261026, incrementing by at most 99; both
classes must occur in training/testing. Selection uses only labels/groups.
Persist every inner/outer subject and segment membership.

StandardScaler fits only training data inside each inner fold and each outer
refit. SVC uses balanced class weights and a fixed grid C={1,10,100,190},
gamma={scale,0.01,0.1}. Choose mean inner balanced accuracy, retaining the first
grid point on a tie. Use the unchanged decision threshold zero. No test-based
selection of features, thresholds, eligibility, channel or hyperparameters.

Out-of-fold metrics include balanced accuracy, HIGH sensitivity, LOW specificity,
HIGH F1 and macro F1, accuracy, coverage, and a 2x3 confusion matrix including
ABSTAIN. Failed extraction remains a failed segment; in end-to-end metrics its
prediction counts as incorrect. A common-success comparison is separate.

Generate 2000 shared subject-cluster bootstrap draws for 95% percentile CIs and
paired HOS-minus-baseline differences. Draws missing either class are unusable
and their number is disclosed. These CIs condition on the existing out-of-fold
fits; they do not estimate the uncertainty of refitting the entire nested search.
Segments are not treated as independent patients.

## Artifacts and checks

Run output includes cohort.csv (all candidates), manifest.csv (eligible), source
hashes, versioned protocol/configuration, preparation/extraction/evaluation code
provenance, both feature tables, QRS metrics, splits, per-segment predictions,
model parameters, bootstrap intervals, figures and report_fa.md. Evaluating a run
whose frozen inputs changed is rejected. Prepare/extract require a fresh output
directory once downstream artifacts exist.

Tests cover threshold boundaries, annotation denominator and rhythm duration,
shared patient IDs, MLII selection, flatline quality, sample-rate conversions,
QRS matching, subject-disjoint splits, frozen predictor allowlists, numerical
bispectrum correctness and an offline synthetic WFDB-to-report experiment.
Synthetic results validate software, not clinical performance.

Jev's free advisory endpoint was requested during planning but failed with a
network error. The selected design rests on annotation counts, source methods
and code inspection. No paid model or alternative endpoint was used.
