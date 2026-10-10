# Paper audit: Chua et al. (2009)

## What the paper used

The target paper reports ECG recordings of 10-15 minutes from more than 330 reclining patients at Kasturba Medical College Hospital, Manipal, India. The original analog Holter recordings were digitized at 320 samples/s. The five classes and counts in Table 1 are:

| Class | Meaning in the paper | Records |
|---|---|---:|
| NSR | Normal sinus rhythm | 183 |
| PVC | Premature/prev­entricular contraction | 37 |
| CHB | Complete (third-degree) heart block | 42 |
| SSS | Sick sinus syndrome III | 43 |
| CHF | Congestive heart failure | 25 |
| **Total** |  | **330** |

The article gives no dataset DOI, repository URL, accession number, record identifiers, patient-level metadata, or data-availability statement. The cohort is therefore not downloadable from the paper itself. The exact source files would have to come from the authors or the hospital, subject to their permissions and the original clinical-data approvals.

## Signal processing stated in the paper

1. Low-pass filter at 35 Hz.
2. High-pass filter at 0.3 Hz.
3. Band-stop filter at 50 or 60 Hz for power-line interference.
4. Median filter to estimate baseline wander; subtract that baseline.
5. Detect R peaks with the Tompkins algorithm.
6. Convert successive R-peak intervals to heart rate with `HR = 60 / RR`.
7. Resample the heart-rate series to 4 samples/s.
8. Use 1024-sample (256 s) blocks with 512-sample overlap.

For public comparators with a native sampling rate other than 320 Hz, the project resamples the ECG to 320 Hz before applying these filters when `PaperConfig.ecg_target_fs` is left at its default. The native rate remains recorded in the manifest.

The median window, filter orders, interpolation method, ectopic-beat handling, and detailed Tompkins thresholds are not specified.

## Seven features

The paper computes the direct bispectrum from FFT products,

`B(f1,f2) = E[X(f1) X(f2) X*(f1+f2)]`,

over the non-redundant triangle shown in Figure 1. In normalized-by-Nyquist frequency-bin indices this is:

`0 <= f2 <= f1` and `f1 + f2 <= 1`.

The feature vector is:

* `P1`: normalized entropy of `|B|`;
* `P2`: normalized entropy of `|B|^2`;
* `H1`: sum of `log(|B|)` over the triangle;
* `H2`: sum of `log(|B(k,k)|)` over the diagonal;
* `H3`: `sum(k * log(|B(k,k)|))`;
* `f1m`, `f2m`: weighted bispectral centers.

The paper does not state the logarithm base. The implementation defaults to the natural logarithm and exposes the base as a configuration option. The paper writes raw entropy equations but says the results are between 0 and 1; this implementation divides Shannon entropy by `log(number of bins)` to make that normalization explicit. The WCOB equations are printed with complex `B`; the implementation uses `|B|`, which is the conventional weighted-center interpretation and is required for real-valued centers.

## Classifier and reported targets

The authors used a one-against-all RBF SVM with `C=190`. The implementation uses an explicit `OneVsRestClassifier` wrapper. Standardization is fitted only on training features; whether the original authors standardized is not reported. They selected approximately two-thirds of each class for training and one-third for testing, repeated five times with random combinations and no overlap within a trial. The RBF gamma and random seed are not reported. The current evaluator splits records, so multiple records from the same subject require a future grouped evaluation implementation.

For quantitative analysis, the paper says it applied ANOVA and then a t-test with more than 95% confidence, but does not state the t-test variant, multiple-comparison correction, or the exact implementation. This project reports one-way ANOVA and pairwise Welch t-tests as a transparent default.

The published aggregate confusion matrix contains 500 predictions, even though 330 records are described, which implies 100 test predictions per repetition. The exact class-wise allocation and split indices are not given. This project therefore supports a configurable `test_size=100` paper-like mode, but cannot claim the exact matrix without the original split information.

## Data and code search result

The article is available through [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2709931/) and the journal's [PDF archive](https://benthamopenarchives.com/contents/pdf/TOMINFOJ/TOMINFOJ-3-1.pdf), but neither page exposes the underlying ECG files or paper-specific source code. The local `HOSA(1).PDF` is the HOSA MATLAB toolbox user guide, not a reproduction package. Public general-purpose alternatives include the [HOSA MATLAB toolbox](https://www.mathworks.com/matlabcentral/fileexchange/3013-hosa-higher-order-spectral-analysis-toolbox), [HOSA_Octave](https://github.com/synergetics/HOSA_Octave), [higher-spectrum](https://github.com/synergetics/spectrum), and [PyBispectra](https://github.com/braindatalab/PyBispectra); none is the article's end-to-end implementation or dataset.

## Public validation data

The closest duration match is the [QT Database](https://physionet.org/content/qtdb/1.0.0/): more than 100 two-lead, fifteen-minute recordings with waveform annotations. Its records do not have the paper's diagnostic labels and the sampling frequency is record-specific, so it is suitable for a pipeline test after resampling, not an exact replication. [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/) provides two-lead, 30-minute records at 360 Hz with beat annotations and many PVC examples. [St Petersburg INCART](https://physionet.org/content/incartdb/1.0.0/) provides 12-lead, 30-minute Holter-derived records at 257 Hz and includes a small number of sinus-node dysfunction and AV-block patients. [MIT-BIH Normal Sinus Rhythm](https://physionet.org/content/nsrdb/1.0.0/) supplies the normal reference group, while [BIDMC Congestive Heart Failure](https://physionet.org/content/chfdb/1.0.0/) supplies a CHF group; their header files show 128 Hz and 250 Hz respectively. [European ST-T](https://physionet.org/content/edb/1.0.0/) adds two-hour, 250-Hz ambulatory records with expert annotations of rhythm and conduction events.

These datasets can be segmented and resampled into the paper's 320-Hz/10–15-minute processing format, but their patients, labels, recording hardware, and class definitions differ. Results on them are external validation or a methodological reproduction, not a re-creation of Tables 3 and 4.
