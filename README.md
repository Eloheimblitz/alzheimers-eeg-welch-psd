# Alzheimer's EEG Welch PSD Pipeline

Phase 1 implementation of an M.Tech project on Alzheimer's disease detection from resting-state EEG using Welch-based frequency-domain features.

This repository currently contains only the **progress-work pipeline**:
- dataset inspection
- subject metadata creation
- EEG loading and preprocessing
- artifact rejection
- 4-second epoch segmentation
- Welch PSD feature extraction
- subject-wise PSD feature export
- summary CSV generation
- progress-presentation visualizations

It does **not** yet include CNN training, evaluation, or comparative experiments.

## Project Title
**Comparative Evaluation of Convolutional Neural Networks for Alzheimer's Disease Detection Using Welch-Based Frequency-Domain EEG Features**

## Current Scope
Implemented in this repository:
- recursive EEG file detection from OpenNeuro `ds004504`
- participant-to-class mapping from `participants.tsv`
- label mapping:
  - `AD -> 0`
  - `CN -> 1`
  - `FTD -> 2`
- MNE-based EEG preprocessing
- fixed-threshold artifact rejection
- Welch PSD computation with relative normalization
- per-subject `.npz` feature storage
- progress figures for presentation

Not implemented yet:
- subject-wise train/validation/test split
- stratified group-wise validation
- EEGNet
- ShallowConvNet
- DeepConvNet
- SCCNet
- FBCNet
- training pipeline
- evaluation metrics
- comparative model analysis

## Dataset
- Source dataset: OpenNeuro `ds004504`
- Recording type: resting-state eyes-closed EEG
- Expected EEG formats supported by the scanner:
  - `.set`
  - `.edf`
  - `.vhdr`
  - `.fif`
- Dataset group codes used in `participants.tsv`:
  - `A -> AD`
  - `C -> CN`
  - `F -> FTD`

The raw dataset is **not tracked in git** and should be placed locally under:

`data/raw/ds004504/`

Expected key file:
- `data/raw/ds004504/participants.tsv`

## Repository Structure
```text
alz_project/
|-- data/
|   |-- raw/
|   |   `-- ds004504/
|   |-- metadata/
|   `-- processed/
|       `-- psd_features/
|-- notebooks/
|   |-- 01_dataset_overview.ipynb
|   |-- 02_preprocessing_single_sample.ipynb
|   |-- 03_feature_extraction_single_sample.ipynb
|   |-- 04_feature_extraction_all_subjects.ipynb
|   `-- 05_progress_visualizations.ipynb
|-- results/
|   `-- figures/
|-- src/
|   |-- dataset_loader.py
|   |-- preprocessing.py
|   |-- feature_extraction.py
|   `-- utils.py
|-- PROJECT_OVERVIEW.md
|-- README.md
`-- requirements.txt
```

## Setup
Windows + VS Code setup:

1. Create project folder:
   `alz_project`
2. Open the folder in VS Code
3. Create virtual environment:
   `python -m venv .venv`
4. Activate virtual environment in Windows PowerShell:
   `.\.venv\Scripts\Activate.ps1`
5. Install dependencies:
   `pip install -r requirements.txt`
6. Register Jupyter kernel:
   `python -m ipykernel install --user --name alz_env --display-name "Python (alz_env)"`
7. In VS Code, select the kernel:
   `Python (alz_env)`

## Notebook Order
Run the notebooks in this order:

1. `01_dataset_overview.ipynb`
2. `02_preprocessing_single_sample.ipynb`
3. `03_feature_extraction_single_sample.ipynb`
4. `04_feature_extraction_all_subjects.ipynb`
5. `05_progress_visualizations.ipynb`

## Notebook Summary
`01_dataset_overview.ipynb`
- tests the environment
- scans the dataset
- creates `subject_metadata.csv`
- summarizes class distribution

`02_preprocessing_single_sample.ipynb`
- loads one subject
- filters EEG from `0.5-45 Hz`
- creates 4-second epochs
- applies artifact rejection
- prints sampling frequency, channel count, and epoch shape

`03_feature_extraction_single_sample.ipynb`
- computes Welch PSD for one subject
- shows PSD shape and frequency bins
- visualizes normalized PSD and log-PSD

`04_feature_extraction_all_subjects.ipynb`
- preprocesses all valid subjects
- extracts PSD features for all subjects
- saves one `.npz` file per subject
- creates `psd_feature_summary.csv`
- reports failures without stopping the run

`05_progress_visualizations.ipynb`
- generates presentation-ready figures
- saves outputs under `results/figures/`

## Core Processing Choices
- filter band: `0.5-45 Hz`
- epoch duration: `4 seconds`
- overlap: `0`
- artifact amplitude threshold: `+/-150 uV`
- fixed high-frequency artifact check in `20-45 Hz`
- Welch PSD parameters:
  - `fmin=1`
  - `fmax=45`
  - `n_fft=1000`
  - `n_per_seg=1000`
  - `n_overlap=500`
  - `window='hamming'`

## Recording Selection Rule
If multiple EEG files are found for one subject, the selection rule is:

1. prefer the primary BIDS subject recording
2. ignore `derivatives/` copies when a primary recording exists
3. if multiple valid primary files remain, choose the alphabetically first path

This keeps the metadata and feature extraction pipeline reproducible.

## Generated Outputs
Main generated files:
- `data/metadata/subject_metadata.csv`
- `data/metadata/psd_feature_summary.csv`
- `data/processed/psd_features/sub-XXX_psd_features.npz`
- figures in `results/figures/`

Each PSD feature file stores:
- `subject_id`
- `class_name`
- `label`
- `psd_features`
- `freqs`

## Verified Phase 1 Results
Verified locally on the dataset used in this workspace:
- `176` EEG files detected recursively
- `88` matched subjects
- class distribution:
  - `AD: 36`
  - `CN: 29`
  - `FTD: 23`
- `88/88` subjects processed successfully
- `0` failed subjects in the full PSD extraction run

Example shapes:
- epoch array: `(138, 19, 2000)`
- PSD feature array: `(138, 19, 89)`

## Validation Checklist
- EEG files detected successfully
- metadata CSV created
- subject count checked against expected `88`
- PSD features exported per valid subject
- PSD summary CSV created
- failures, if any, reported cleanly

## Dependencies
Listed in `requirements.txt`:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `mne`
- `jupyter`
- `ipykernel`

## Future Work
Phase 2 will extend this repository with:
- subject-wise splitting
- group-aware validation
- CNN architectures
- training and evaluation
- comparative performance analysis

The current Phase 1 pipeline is designed so Phase 2 can build on the saved PSD features without rewriting the preprocessing code.
