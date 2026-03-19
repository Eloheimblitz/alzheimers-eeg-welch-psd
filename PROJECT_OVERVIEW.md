# Project Overview

## Summary
This repository contains the **Phase 1 progress-work implementation** of an M.Tech project on Alzheimer's disease detection from EEG using Welch-based frequency-domain features.

The current work focuses on building a reliable EEG preprocessing and feature-extraction pipeline using the OpenNeuro `ds004504` dataset. It prepares clean subject-wise PSD features that can later be used in Phase 2 for CNN-based classification experiments.

## Objective
The larger project goal is to compare convolutional neural network architectures for Alzheimer's disease detection. This repository currently covers only the pipeline required before any model training:
- dataset inspection
- metadata generation
- EEG preprocessing
- artifact rejection
- fixed-length epoching
- Welch PSD feature extraction
- output export and progress visualization

## Dataset Context
- Dataset: OpenNeuro `ds004504`
- Recording type: resting-state eyes-closed EEG
- Classes:
  - `AD`
  - `CN`
  - `FTD`
- Label encoding:
  - `AD = 0`
  - `CN = 1`
  - `FTD = 2`

## What Phase 1 Produces
- `subject_metadata.csv` linking subjects, labels, and EEG files
- preprocessed subject-wise EEG epochs
- subject-wise Welch PSD feature files in `.npz` format
- `psd_feature_summary.csv`
- progress-presentation figures

## Processing Pipeline
1. Scan the dataset recursively for supported EEG files.
2. Read `participants.tsv` and map subjects to class labels.
3. Load EEG with MNE.
4. Apply `0.5-45 Hz` band-pass filtering.
5. Segment EEG into fixed 4-second non-overlapping epochs.
6. Reject artifacts using fixed thresholds.
7. Compute Welch PSD features from `1-45 Hz`.
8. Save one feature file per subject.
9. Generate summary CSVs and presentation figures.

## Verified Local Results
- `176` EEG files detected
- `88` matched subjects
- `88` successfully processed subjects
- `0` failed subjects
- Example epoch shape: `(138, 19, 2000)`
- Example PSD shape: `(138, 19, 89)`

Observed class counts:
- `AD: 36`
- `CN: 29`
- `FTD: 23`

## Repository Role
This repo is meant to be:
- the clean Phase 1 baseline
- reproducible for progress presentations
- modular enough for Phase 2 expansion
- independent from the raw dataset in git history

## What Is Not Included Yet
- train/validation/test split
- CNN model definitions
- training scripts
- evaluation metrics
- comparative experiments

## Key Files
- `README.md`
- `src/dataset_loader.py`
- `src/preprocessing.py`
- `src/feature_extraction.py`
- `data/metadata/subject_metadata.csv`
- `data/metadata/psd_feature_summary.csv`

## Next Step
The next project phase should build on these saved PSD features and extend the repository with CNN-based classification and comparative evaluation.
