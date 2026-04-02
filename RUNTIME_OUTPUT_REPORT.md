# Full Runtime Output Report (Notebooks 01-06)

## 01_dataset_overview.ipynb
- Total cells: 6

### Cell 2 (code)
- Execution count: 1
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code starts with: `print('Dataset directory:', RAW_DATASET_DIR)`
- Output blocks: 1
  - Block 1: `stream`
    - Stream preview:
      - Dataset directory: D:\2026 MTECH Project\alz_project\data\raw\ds004504
      - Dataset exists: True
      - participants.tsv exists: True
      - Total EEG files detected recursively: 176
      - Sample EEG paths:
      -  - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-001\eeg\sub-001_task-eyesclosed_eeg.set
      -  - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-002\eeg\sub-002_task-eyesclosed_eeg.set
      -  - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-003\eeg\sub-003_task-eyesclosed_eeg.set
      -  - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-004\eeg\sub-004_task-eyesclosed_eeg.set
      -  - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-005\eeg\sub-005_task-eyesclosed_eeg.set

### Cell 4 (code)
- Execution count: 3
- Code starts with: `metadata_df = create_subject_metadata(RAW_DATASET_DIR)`
- Output blocks: 2
  - Block 1: `stream`
    - Stream preview:
      - [alz_project] Total EEG files detected: 176
      - [alz_project] Total matched subjects: 88
      - [alz_project] Class distribution: {'AD': 36, 'CN': 29, 'FTD': 23}
      - [alz_project] Sample EEG files: ['data\\raw\\ds004504\\sub-001\\eeg\\sub-001_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-002\\eeg\\sub-002_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-003\\eeg\\sub-003_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-004\\eeg\\sub-004_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-005\\eeg\\sub-005_task-eyesclosed_eeg.set']
  - Block 2: `execute_result`
    - Data types: text/html, text/plain
    - text/plain preview:
      - subject_id  label class_name  \
      - 0    sub-001      0         AD   
      - 1    sub-002      0         AD   
      - 2    sub-003      0         AD   
      - 3    sub-004      0         AD   
      - 4    sub-005      0         AD   
      - 
      -                                             eeg_file  
      - 0  data\raw\ds004504\sub-001\eeg\sub-001_task-eye...  
      - 1  data\raw\ds004504\sub-002\eeg\sub-002_task-eye...  
      - 2  data\raw\ds004504\sub-003\eeg\sub-003_task-eye...  
      - 3  data\raw\ds004504\sub-004\eeg\sub-004_task-eye...  
      - 4  data\raw\ds004504\sub-005\eeg\sub-005_task-eye...
    - Contains text/html output

### Cell 5 (code)
- Execution count: 4
- Code starts with: `print('Total matched subjects:', len(metadata_df))`
- Output blocks: 2
  - Block 1: `stream`
    - Stream preview:
      - Total matched subjects: 88
      - Expected subject count check against 88: MATCH
  - Block 2: `execute_result`
    - Data types: text/plain
    - text/plain preview:
      - class_name
      - AD     36
      - CN     29
      - FTD    23
      - Name: count, dtype: int64

### Cell 6 (code)
- Execution count: 5
- Code starts with: `fig, ax = plt.subplots(figsize=(10, 6))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1000x600 with 1 Axes>
    - Contains image/png output

## 02_preprocessing_single_sample.ipynb
- Total cells: 9

### Cell 2 (code)
- Execution count: 1
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code starts with: `metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)`
- Output blocks: 1
  - Block 1: `execute_result`
    - Data types: text/plain
    - text/plain preview:
      - subject_id                                              sub-001
      - label                                                         0
      - class_name                                                   AD
      - eeg_file      data\raw\ds004504\sub-001\eeg\sub-001_task-eye...
      - Name: 0, dtype: object

### Cell 4 (code)
- Execution count: 3
- Code starts with: `config = PreprocessingConfig()`
- Output blocks: 1
  - Block 1: `execute_result`
    - Data types: text/plain
    - text/plain preview:
      - {'subject_id': 'sub-001',
      -  'class_name': 'AD',
      -  'n_channels': 19,
      -  'sampling_frequency': 500.0,
      -  'n_epochs': 138,
      -  'epoch_shape': (138, 19, 2000)}

### Cell 5 (code)
- Execution count: 4
- Code starts with: `raw = preprocessed['raw']`
- Output blocks: 1
  - Block 1: `stream`
    - Stream preview:
      - Sampling Frequency: 500 Hz
      - Number of Channels: 19
      - subject_id: sub-001
      - class_name: AD
      - number of epochs: 138
      - epoch array shape: (138, 19, 2000)
      - artifact rejection stats: {'total_epochs': 149, 'retained_epochs': 138, 'rejected_epochs': 11, 'rejected_amplitude': 11, 'rejected_high_frequency': 0}

### Cell 6 (code)
- Execution count: 5
- Code starts with: `fig, ax = plt.subplots(figsize=(12, 6))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1200x600 with 1 Axes>
    - Contains image/png output

### Cell 7 (code)
- Execution count: 6
- Code starts with: `raw.plot(duration=8, n_channels=min(19, len(raw.ch_names)), scalings='auto');`
- Output blocks: 2
  - Block 1: `stream`
    - Stream preview:
      - Using matplotlib as 2D backend.
  - Block 2: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <MNEBrowseFigure size 800x800 with 4 Axes>
    - Contains image/png output

### Cell 8 (code)
- Execution count: 7
- Code starts with: `filtered.plot(duration=8, n_channels=min(19, len(filtered.ch_names)), scalings='auto');`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <MNEBrowseFigure size 800x800 with 4 Axes>
    - Contains image/png output

### Cell 9 (code)
- Execution count: 8
- Code starts with: `fig = epochs[:5].plot(scalings='auto')`
- Output blocks: 2
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <MNEBrowseFigure size 800x800 with 4 Axes>
    - Contains image/png output
  - Block 2: `execute_result`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <MNEBrowseFigure size 800x800 with 4 Axes>
    - Contains image/png output

## 03_feature_extraction_single_sample.ipynb
- Total cells: 6

### Cell 2 (code)
- Execution count: 1
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code starts with: `metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)`
- Output blocks: 1
  - Block 1: `stream`
    - Stream preview:
      - PSD shape: (138, 19, 89)
      - Frequency bins: (89,)
      - First 10 frequencies: [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5]

### Cell 4 (code)
- Execution count: 3
- Code starts with: `epoch_idx = 0`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1250x600 with 1 Axes>
    - Contains image/png output

### Cell 6 (code)
- Execution count: 4
- Code starts with: `fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharex=True)`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1450x550 with 2 Axes>
    - Contains image/png output

## 04_feature_extraction_all_subjects.ipynb
- Total cells: 5

### Cell 2 (code)
- Execution count: 1
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code starts with: `metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)`
- Output blocks: 1
  - Block 1: `execute_result`
    - Data types: text/html, text/plain
    - text/plain preview:
      - setting  epoch_duration_sec  overlap_sec  subjects_found  \
      - 0  non_overlap_0pct                 4.0          0.0              88   
      - 1     overlap_50pct                 4.0          2.0              88   
      - 
      -    successfully_processed  failed_subjects  total_rejected_epochs  \
      - 0                      88                0                   3244   
      - 1                      88                0                   6490   
      - 
      -    avg_retained_epochs_per_subject  
      - 0                       163.181818  
      - 1                       325.863636
    - Contains text/html output

### Cell 4 (code)
- Execution count: 3
- Code starts with: `print('----- Non-overlap (0%) -----')`
- Output blocks: 2
  - Block 1: `stream`
    - Stream preview:
      - ----- Non-overlap (0%) -----
      - total subjects found: 88
      - successfully processed: 88
      - failed subjects: 0
      - Total Rejected Epochs: 3244
      - Average Epochs per Subject: 163.18
      - summary CSV: D:\2026 MTECH Project\alz_project\data\metadata\psd_feature_summary.csv
      - 
      - ----- Overlap 50% (2s) -----
      - total subjects found: 88
      - successfully processed: 88
      - failed subjects: 0
      - Total Rejected Epochs: 6490
      - Average Epochs per Subject: 325.86
      - summary CSV: D:\2026 MTECH Project\alz_project\data\metadata\psd_feature_summary_overlap50.csv
      - 
      - ----- Comparison -----
  - Block 2: `display_data`
    - Data types: text/html, text/plain
    - text/plain preview:
      - setting  epoch_duration_sec  overlap_sec  subjects_found  \
      - 0  non_overlap_0pct                 4.0          0.0              88   
      - 1     overlap_50pct                 4.0          2.0              88   
      - 
      -    successfully_processed  failed_subjects  total_rejected_epochs  \
      - 0                      88                0                   3244   
      - 1                      88                0                   6490   
      - 
      -    avg_retained_epochs_per_subject  
      - 0                       163.181818  
      - 1                       325.863636
    - Contains text/html output

### Cell 5 (code)
- Execution count: 4
- Code starts with: `fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 2000x580 with 3 Axes>
    - Contains image/png output

## 05_progress_visualizations.ipynb
- Total cells: 13

### Cell 2 (code)
- Execution count: 1
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code starts with: `FIGURES_DIR.mkdir(parents=True, exist_ok=True)`
- Output: *(no output)*

### Cell 4 (code)
- Execution count: 3
- Code starts with: `fig, ax = plt.subplots(figsize=(16, 6))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1600x600 with 1 Axes>
    - Contains image/png output

### Cell 5 (code)
- Execution count: 4
- Code starts with: `fig, ax = plt.subplots(figsize=(11, 6))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1100x600 with 1 Axes>
    - Contains image/png output

### Cell 6 (code)
- Execution count: 5
- Code starts with: `raw = preprocessed['raw']`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1500x800 with 2 Axes>
    - Contains image/png output

### Cell 7 (code)
- Execution count: 6
- Code starts with: `fig, ax = plt.subplots(figsize=(15, 5))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1500x500 with 1 Axes>
    - Contains image/png output

### Cell 8 (code)
- Execution count: 7
- Code starts with: `fig, ax = plt.subplots(figsize=(13, 6))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1300x600 with 1 Axes>
    - Contains image/png output

### Cell 9 (code)
- Execution count: 8
- Code starts with: `fig, ax = plt.subplots(figsize=(13, 4.5))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1300x450 with 1 Axes>
    - Contains image/png output

### Cell 11 (code)
- Execution count: 9
- Code starts with: `if not summary_df.empty:`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1100x600 with 1 Axes>
    - Contains image/png output

### Cell 12 (code)
- Execution count: 10
- Code starts with: `if not summary_df.empty:`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1600x600 with 1 Axes>
    - Contains image/png output

### Cell 13 (code)
- Execution count: 11
- Code starts with: `fig, ax = plt.subplots(figsize=(13, 7))`
- Output blocks: 1
  - Block 1: `display_data`
    - Data types: image/png, text/plain
    - text/plain preview:
      - <Figure size 1300x700 with 1 Axes>
    - Contains image/png output

## 06_single_cnn_eegnet_baseline.ipynb
- Total cells: 9

### Cell 2 (code)
- Execution count: None
- Code starts with: `import sys, torch`
- Output blocks: 1
  - Block 1: `stream`
    - Stream preview:
      - c:\Users\Jediael\AppData\Local\Programs\Python\Python312\python.exe

### Cell 3 (code)
- Execution count: None
- Code starts with: `%matplotlib inline`
- Output: *(no output)*

### Cell 4 (code)
- Execution count: None
- Code starts with: `try:`
- Output blocks: 1
  - Block 1: `error`
    - Error: RuntimeError: PyTorch is not installed. Run: pip install torch
    - Traceback preview:
      - [31m---------------------------------------------------------------------------[39m
      - [31mModuleNotFoundError[39m                       Traceback (most recent call last)
      - [36mCell[39m[36m [39m[32mIn[4][39m[32m, line 2[39m
[32m      1[39m [38;5;28;01mtry[39;00m:
[32m----> [39m[32m2[39m     [38;5;28;01mimport[39;00m[38;5;250m [39m[34;01mtorch[39;00m  [38;5;66;03m# noqa: F401[39;00m
[32m      3[39m     [38;5;28mprint[39m([33m'[39m[33mPyTorch detected. Ready to train EEGNet.[39m[33m'[39m)

      - [31mModuleNotFoundError[39m: No module named 'torch'
      - 
During handling of the above exception, another exception occurred:

      - [31mRuntimeError[39m                              Traceback (most recent call last)
      - [36mCell[39m[36m [39m[32mIn[4][39m[32m, line 5[39m
[32m      3[39m     [38;5;28mprint[39m([33m'[39m[33mPyTorch detected. Ready to train EEGNet.[39m[33m'[39m)
[32m      4[39m [38;5;28;01mexcept[39;00m [38;5;167;01mException[39;00m:
[32m----> [39m[32m5[39m     [38;5;28;01mraise[39;00m [38;5;167;01mRuntimeError[39;00m([33m'[39m[33mPyTorch is not installed. Run: pip install torch[39m[33m'[39m)

      - [31mRuntimeError[39m: PyTorch is not installed. Run: pip install torch

### Cell 5 (code)
- Execution count: None
- Code starts with: `summary_csv_non_overlap = PROJECT_ROOT / 'data' / 'metadata' / 'psd_feature_summary.csv'`
- Output: *(no output)*

### Cell 6 (code)
- Execution count: None
- Code starts with: `cfg = EEGNetConfig(`
- Output: *(no output)*

### Cell 7 (code)
- Execution count: None
- Code starts with: `# Presentation-ready comparison plot`
- Output: *(no output)*

### Cell 8 (code)
- Execution count: None
- Code starts with: `# Optional publication-style chart: dumbbell comparison (Test Accuracy)`
- Output: *(no output)*
