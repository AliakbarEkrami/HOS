# Contributing

## Workflow

1. Clone the repository and install `requirements.txt` in a virtual environment.
2. Update your checkout with `git pull --ff-only`.
3. Create a branch with `git switch -c your-change-name`.
4. Make changes and run `python -m pytest hos_hrv_reproduction/tests -q` from the repository root.
5. Commit, push your branch, and open a pull request explaining the change and validation.

If you do not have write access, fork the repository and submit a pull request from your fork. For direct access, the owner must invite your GitHub account in repository settings.

## Scientific requirements

- Keep reported paper settings separate from implementation assumptions; document any new assumption in `docs/paper_audit.md`.
- Do not report public comparator results as exact reproduction of the original study.
- Retain each subject's identity when creating multiple segments. Keep every subject in a single partition when evaluating segmented datasets. The current evaluator splits record-level features and does not enforce subject grouping; use one record per subject until grouped evaluation is implemented.
- Do not map beat labels to patient diagnoses automatically. Document every cohort and label-selection rule.
- Keep standardization and any parameter selection within training data.
- Add meaningful tests when changing equations, resampling, signal detection, or classification metrics.
- Keep raw ECG data, downloaded datasets, local environments, credentials, and generated results out of Git.
- Respect source-data access terms and preserve paper attribution.

CI runs the tests on Python 3.10 and 3.12. Use small synthetic fixtures for tests so they do not require clinical data or network downloads.
