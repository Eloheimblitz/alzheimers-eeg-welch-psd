# Phase 1 Final Report

## Project Title
Comparative Evaluation of Convolutional Neural Networks for Alzheimer's Disease Detection Using Welch-Based Frequency-Domain EEG Features

## Scope of Phase 1
Phase 1 focused on building and verifying the EEG data preparation pipeline only. This phase includes dataset inspection, metadata creation, EEG preprocessing, artifact rejection, epoch segmentation, Welch PSD feature extraction, per-subject feature saving, summary generation, and progress visualizations. No CNN modeling or classification experiments were implemented in this phase.

## Work Completed
- Recursive scan of the OpenNeuro `ds004504` dataset
- Subject label mapping using `participants.tsv`
- Metadata creation for all matched subjects
- EEG loading using MNE
- Band-pass filtering from `0.5-45 Hz`
- Fixed `4-second` non-overlapping epoch segmentation
- Artifact rejection using fixed amplitude and high-frequency thresholds
- Welch PSD feature extraction with relative normalization
- Saving one PSD feature file per valid subject
- Creation of summary CSV files
- Generation of progress-presentation figures

## Verified Dataset and Processing Outputs

### Dataset Metadata
File: [data/metadata/subject_metadata.csv](/d:/2026%20MTECH%20Project/alz_project/data/metadata/subject_metadata.csv)

- Total matched subjects: `88`
- Class distribution:
  - `AD = 36`
  - `CN = 29`
  - `FTD = 23`

Example rows:

```text
subject_id  label class_name                                                      eeg_file
sub-001         0         AD data\raw\ds004504\sub-001\eeg\sub-001_task-eyesclosed_eeg.set
sub-002         0         AD data\raw\ds004504\sub-002\eeg\sub-002_task-eyesclosed_eeg.set
sub-003         0         AD data\raw\ds004504\sub-003\eeg\sub-003_task-eyesclosed_eeg.set
sub-004         0         AD data\raw\ds004504\sub-004\eeg\sub-004_task-eyesclosed_eeg.set
sub-005         0         AD data\raw\ds004504\sub-005\eeg\sub-005_task-eyesclosed_eeg.set
```

### PSD Feature Summary
File: [data/metadata/psd_feature_summary.csv](/d:/2026%20MTECH%20Project/alz_project/data/metadata/psd_feature_summary.csv)

- Total processed subjects: `88`
- Successfully processed: `88`
- Failed subjects: `0`
- Average retained epochs per subject: `163.18`
- Minimum retained epochs: `1`
- Maximum retained epochs: `315`
- Channels per subject: `19`
- Frequency bins per subject: `89`

Example rows:

```text
subject_id class_name  label  n_epochs  n_channels  n_freq_bins                                          output_file
sub-001          AD      0       138          19           89 data\processed\psd_features\sub-001_psd_features.npz
sub-002          AD      0       180          19           89 data\processed\psd_features\sub-002_psd_features.npz
sub-003          AD      0        76          19           89 data\processed\psd_features\sub-003_psd_features.npz
sub-004          AD      0       143          19           89 data\processed\psd_features\sub-004_psd_features.npz
sub-005          AD      0       151          19           89 data\processed\psd_features\sub-005_psd_features.npz
```

### Saved Feature Files
Folder: [data/processed/psd_features](/d:/2026%20MTECH%20Project/alz_project/data/processed/psd_features)

- Total `.npz` PSD files saved: `88`

## Verified Example Outputs from Notebooks

### Single-Sample Preprocessing
Notebook: [notebooks/02_preprocessing_single_sample.ipynb](/d:/2026%20MTECH%20Project/alz_project/notebooks/02_preprocessing_single_sample.ipynb)

```text
Sampling Frequency: 500 Hz
Number of Channels: 19
subject_id: sub-001
class_name: AD
number of epochs: 138
epoch array shape: (138, 19, 2000)
artifact rejection stats: {'total_epochs': 149, 'retained_epochs': 138, 'rejected_epochs': 11, 'rejected_amplitude': 11, 'rejected_high_frequency': 0}
```

### Single-Sample Feature Extraction
Notebook: [notebooks/03_feature_extraction_single_sample.ipynb](/d:/2026%20MTECH%20Project/alz_project/notebooks/03_feature_extraction_single_sample.ipynb)

```text
PSD shape: (138, 19, 89)
Frequency bins: (89,)
First 10 frequencies: [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5]
```

### All-Subject Processing Summary
Notebook: [notebooks/04_feature_extraction_all_subjects.ipynb](/d:/2026%20MTECH%20Project/alz_project/notebooks/04_feature_extraction_all_subjects.ipynb)

```text
total subjects found: 88
successfully processed: 88
failed subjects: 0
Total Subjects: 88
Total Rejected Epochs: 3244
Average Epochs per Subject: 163.18
```

## Figures Generated
Folder: [results/figures](/d:/2026%20MTECH%20Project/alz_project/results/figures)

Main presentation figures:
- `01_pipeline_diagram.png`
- `02_class_distribution.png`
- `03_raw_vs_filtered_eeg.png`
- `04_example_epoch.png`
- `05_example_psd.png`
- `06_frequency_bin_summary.png`
- `07_processed_subjects_per_class.png`
- `08_epoch_counts_per_subject.png`
- `09_progress_work_summary.png`

Supporting notebook-exported figures:
- `01_dataset_overview_class_distribution_notebook.png`
- `02_preprocessing_artifact_rejection_summary_notebook.png`
- `02_preprocessing_raw_eeg_browser_notebook.png`
- `02_preprocessing_filtered_eeg_browser_notebook.png`
- `02_preprocessing_epoch_view_notebook_1.png`
- `02_preprocessing_epoch_view_notebook_2.png`
- `03_feature_extraction_selected_channels_psd_notebook.png`
- `03_feature_extraction_relative_vs_log_psd_notebook.png`
- `04_all_subjects_processing_summary_notebook.png`

## Phase 1 Outcome
Phase 1 was completed successfully. The EEG preprocessing and Welch PSD feature extraction pipeline was built, executed, and verified on `88` matched subjects from the dataset. The project now has reusable subject-wise PSD feature files and summary outputs ready for Phase 2 modeling.

## Future Work
The following items are intentionally excluded from Phase 1 and remain for the final project phase:
- subject-wise train/validation/test split
- EEGNet
- ShallowConvNet
- DeepConvNet
- SCCNet
- FBCNet
- model training
- evaluation metrics
- comparative analysis
- final classification experiments
