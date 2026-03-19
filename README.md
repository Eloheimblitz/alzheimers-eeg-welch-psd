# Comparative Evaluation of Convolutional Neural Networks for Alzheimer's Disease Detection Using Welch-Based Frequency-Domain EEG Features

## Project Purpose
This repository currently implements only the **PROGRESS PHASE** pipeline for an M.Tech project based on the OpenNeuro `ds004504` EEG dataset.

Implemented in this phase:
- Dataset inspection and recursive EEG file detection
- Subject metadata creation from `participants.tsv`
- EEG loading and preprocessing with MNE
- Artifact rejection using fixed thresholds
- Fixed 4-second epoch segmentation
- Welch PSD feature extraction
- PSD feature generation for all valid subjects
- Progress-work visualizations and summary files

Not implemented yet:
- Subject-wise splitting
- Stratified group-wise validation
- EEGNet, ShallowConvNet, DeepConvNet, SCCNet, FBCNet
- Training, evaluation, and comparative classification experiments

## Dataset Information
- Dataset: OpenNeuro `ds004504`
- Expected recording type: resting-state eyes-closed EEG
- Supported EEG formats in the scanner: `.set`, `.edf`, `.vhdr`, `.fif`
- Class mapping from `participants.tsv`:
  - `AD` -> `0`
  - `CN` -> `1`
  - `FTD` -> `2`
- In this dataset, the `Group` column uses:
  - `A` -> `AD`
  - `C` -> `CN`
  - `F` -> `FTD`

## Folder Structure
```text
alz_project/
|
|-- data/
|   |-- raw/
|   |   `-- ds004504/
|   |-- metadata/
|   `-- processed/
|       `-- psd_features/
|
|-- notebooks/
|   |-- 01_dataset_overview.ipynb
|   |-- 02_preprocessing_single_sample.ipynb
|   |-- 03_feature_extraction_single_sample.ipynb
|   |-- 04_feature_extraction_all_subjects.ipynb
|   `-- 05_progress_visualizations.ipynb
|
|-- src/
|   |-- dataset_loader.py
|   |-- preprocessing.py
|   |-- feature_extraction.py
|   `-- utils.py
|
|-- results/
|   `-- figures/
|
|-- requirements.txt
`-- README.md
```

## Windows + VS Code Setup
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

## Dataset Placement
Place the OpenNeuro dataset inside:

`data/raw/ds004504/`

Expected key files:
- `data/raw/ds004504/participants.tsv`
- subject folders such as `sub-001/`, `sub-002/`, ...

## Recording Selection Rule
The dataset scanner searches recursively for supported EEG files and may detect multiple files for the same subject.

Selection rule used consistently in code:
1. Prefer EEG files from the primary subject folders.
2. Ignore `derivatives/` copies when a primary file exists.
3. If more than one eligible file remains, choose the alphabetically first path.

This keeps metadata creation reproducible and avoids mixing raw and derivative copies.

## Notebook Execution Order
Run the notebooks in this order:
1. `01_dataset_overview.ipynb`
2. `02_preprocessing_single_sample.ipynb`
3. `03_feature_extraction_single_sample.ipynb`
4. `04_feature_extraction_all_subjects.ipynb`
5. `05_progress_visualizations.ipynb`

## What Each Notebook Does
### 1. Dataset Overview
- Tests the environment
- Scans the dataset
- Detects EEG files recursively
- Creates `data/metadata/subject_metadata.csv`
- Shows metadata preview and class distribution

### 2. Preprocessing Single Sample
- Loads one sample subject
- Shows raw EEG information
- Applies 0.5-45 Hz filtering
- Segments 4-second epochs
- Applies artifact rejection
- Prints epoch array shape

### 3. Feature Extraction Single Sample
- Uses one preprocessed subject
- Computes Welch PSD features
- Shows PSD shape and frequency bins
- Visualizes relative normalization and optional log-PSD

### 4. Feature Extraction All Subjects
- Loops through all matched subjects
- Preprocesses EEG and rejects bad epochs
- Extracts Welch PSD features
- Saves one `.npz` file per subject
- Creates `data/metadata/psd_feature_summary.csv`
- Records failures without stopping the full run

### 5. Progress Visualizations
- Saves presentation-ready figures to `results/figures/`
- Summarizes the implemented Phase 1 workflow

## Expected Outputs
After running the notebooks, the main outputs should be:
- `data/metadata/subject_metadata.csv`
- `data/metadata/psd_feature_summary.csv`
- `data/processed/psd_features/sub-XXX_psd_features.npz`
- progress figures inside `results/figures/`

Each PSD feature file contains:
- `subject_id`
- `class_name`
- `label`
- `psd_features`
- `freqs`

## Validation Checklist
Use this checklist for the progress submission:
- EEG files detected successfully
- Metadata CSV created successfully
- Actual subject count checked against the expected total of 88
- Example preprocessing output shown, including epoch shape
- PSD feature files saved per valid subject
- PSD feature summary CSV created
- Failed subjects, if any, reported clearly without stopping the pipeline

## Typical Example Shapes
Depending on artifact rejection and recording duration, example shapes should look like:
- Epoch array: `(n_epochs, n_channels, n_samples)`
- Example: `(149, 19, 2000)`
- PSD array: `(n_epochs, n_channels, n_freq_bins)`
- Example: `(149, 19, 89)`

## Phase 1 Module Responsibilities
- `src/dataset_loader.py`: dataset scanning, subject ID extraction, `participants.tsv` parsing, metadata creation
- `src/preprocessing.py`: EEG loading, filtering, epoching, artifact rejection, NumPy conversion
- `src/feature_extraction.py`: Welch PSD extraction, relative normalization, optional log transform, PSD save/load helpers
- `src/utils.py`: project paths, folder creation, utility helpers

## Future Work
The final project phase will extend this repository with:
- subject-wise train/validation/test splitting
- group-aware validation strategy
- CNN architectures for EEG classification
- model training and evaluation
- comparative performance analysis across architectures

Phase 2 should build on the saved Phase 1 PSD features and modular preprocessing pipeline without rewriting the current codebase.
