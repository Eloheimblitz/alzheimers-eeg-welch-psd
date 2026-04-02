# Full Runtime Output Report (Complete, Untrimmed)

## 01_dataset_overview.ipynb
- Total cells: 6

### Cell 2 (code)
- Execution count: 1
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
import pandas as pd

from dataset_loader import create_subject_metadata, scan_eeg_files
from utils import RAW_DATASET_DIR
```
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code:
```python
print('Dataset directory:', RAW_DATASET_DIR)
print('Dataset exists:', RAW_DATASET_DIR.exists())
print('participants.tsv exists:', (RAW_DATASET_DIR / 'participants.tsv').exists())
eeg_files = scan_eeg_files(RAW_DATASET_DIR)
print('Total EEG files detected recursively:', len(eeg_files))
print('Sample EEG paths:')
for path in eeg_files[:5]:
    print(' -', path)
```

#### Output Block 1 (stream)
```text
Dataset directory: D:\2026 MTECH Project\alz_project\data\raw\ds004504
Dataset exists: True
participants.tsv exists: True
Total EEG files detected recursively: 176
Sample EEG paths:
 - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-001\eeg\sub-001_task-eyesclosed_eeg.set
 - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-002\eeg\sub-002_task-eyesclosed_eeg.set
 - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-003\eeg\sub-003_task-eyesclosed_eeg.set
 - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-004\eeg\sub-004_task-eyesclosed_eeg.set
 - D:\2026 MTECH Project\alz_project\data\raw\ds004504\sub-005\eeg\sub-005_task-eyesclosed_eeg.set
```

### Cell 4 (code)
- Execution count: 3
- Code:
```python
metadata_df = create_subject_metadata(RAW_DATASET_DIR)
metadata_df.head()
```

#### Output Block 1 (stream)
```text
[alz_project] Total EEG files detected: 176
[alz_project] Total matched subjects: 88
[alz_project] Class distribution: {'AD': 36, 'CN': 29, 'FTD': 23}
[alz_project] Sample EEG files: ['data\\raw\\ds004504\\sub-001\\eeg\\sub-001_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-002\\eeg\\sub-002_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-003\\eeg\\sub-003_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-004\\eeg\\sub-004_task-eyesclosed_eeg.set', 'data\\raw\\ds004504\\sub-005\\eeg\\sub-005_task-eyesclosed_eeg.set']
```

#### Output Block 2 (execute_result)
- Data MIME types: text/html, text/plain
```text
  subject_id  label class_name  \
0    sub-001      0         AD   
1    sub-002      0         AD   
2    sub-003      0         AD   
3    sub-004      0         AD   
4    sub-005      0         AD   

                                            eeg_file  
0  data\raw\ds004504\sub-001\eeg\sub-001_task-eye...  
1  data\raw\ds004504\sub-002\eeg\sub-002_task-eye...  
2  data\raw\ds004504\sub-003\eeg\sub-003_task-eye...  
3  data\raw\ds004504\sub-004\eeg\sub-004_task-eye...  
4  data\raw\ds004504\sub-005\eeg\sub-005_task-eye...  
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject_id</th>
      <th>label</th>
      <th>class_name</th>
      <th>eeg_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub-001</td>
      <td>0</td>
      <td>AD</td>
      <td>data\raw\ds004504\sub-001\eeg\sub-001_task-eye...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sub-002</td>
      <td>0</td>
      <td>AD</td>
      <td>data\raw\ds004504\sub-002\eeg\sub-002_task-eye...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sub-003</td>
      <td>0</td>
      <td>AD</td>
      <td>data\raw\ds004504\sub-003\eeg\sub-003_task-eye...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sub-004</td>
      <td>0</td>
      <td>AD</td>
      <td>data\raw\ds004504\sub-004\eeg\sub-004_task-eye...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sub-005</td>
      <td>0</td>
      <td>AD</td>
      <td>data\raw\ds004504\sub-005\eeg\sub-005_task-eye...</td>
    </tr>
  </tbody>
</table>
</div>
```

### Cell 5 (code)
- Execution count: 4
- Code:
```python
print('Total matched subjects:', len(metadata_df))
print('Expected subject count check against 88:', 'MATCH' if len(metadata_df) == 88 else 'DIFFERS')
class_counts = metadata_df['class_name'].value_counts().sort_index()
class_counts
```

#### Output Block 1 (stream)
```text
Total matched subjects: 88
Expected subject count check against 88: MATCH
```

#### Output Block 2 (execute_result)
- Data MIME types: text/plain
```text
class_name
AD     36
CN     29
FTD    23
Name: count, dtype: int64
```

### Cell 6 (code)
- Execution count: 5
- Code:
```python
fig, ax = plt.subplots(figsize=(10, 6))
class_counts.plot(kind='bar', ax=ax)
ax.set_title('Class Distribution from subject_metadata.csv')
ax.set_xlabel('Class')
ax.set_ylabel('Number of Subjects')
ax.grid(True, axis='y', alpha=0.3)
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1000x600 with 1 Axes>
```
- image/png present (base64 length: 23624)

## 02_preprocessing_single_sample.ipynb
- Total cells: 9

### Cell 2 (code)
- Execution count: 1
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
import pandas as pd

from dataset_loader import create_subject_metadata
from preprocessing import PreprocessingConfig, describe_preprocessed_subject, preprocess_subject
from utils import RAW_DATASET_DIR
```
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code:
```python
metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)
sample_row = metadata_df.iloc[0]
sample_row
```

#### Output Block 1 (execute_result)
- Data MIME types: text/plain
```text
subject_id                                              sub-001
label                                                         0
class_name                                                   AD
eeg_file      data\raw\ds004504\sub-001\eeg\sub-001_task-eye...
Name: 0, dtype: object
```

### Cell 4 (code)
- Execution count: 3
- Code:
```python
config = PreprocessingConfig()
preprocessed = preprocess_subject(sample_row['eeg_file'], config=config)
summary = describe_preprocessed_subject(sample_row['subject_id'], sample_row['class_name'], preprocessed)
summary
```

#### Output Block 1 (execute_result)
- Data MIME types: text/plain
```text
{'subject_id': 'sub-001',
 'class_name': 'AD',
 'n_channels': 19,
 'sampling_frequency': 500.0,
 'n_epochs': 138,
 'epoch_shape': (138, 19, 2000)}
```

### Cell 5 (code)
- Execution count: 4
- Code:
```python
raw = preprocessed['raw']
filtered = preprocessed['filtered']
epochs = preprocessed['epochs']
epoch_array = preprocessed['epoch_array']
reject_stats = preprocessed['reject_stats']

print(f"Sampling Frequency: {int(raw.info['sfreq'])} Hz")
print(f"Number of Channels: {len(raw.ch_names)}")
print('subject_id:', sample_row['subject_id'])
print('class_name:', sample_row['class_name'])
print('number of epochs:', len(epochs))
print('epoch array shape:', epoch_array.shape)
print('artifact rejection stats:', reject_stats)
```

#### Output Block 1 (stream)
```text
Sampling Frequency: 500 Hz
Number of Channels: 19
subject_id: sub-001
class_name: AD
number of epochs: 138
epoch array shape: (138, 19, 2000)
artifact rejection stats: {'total_epochs': 149, 'retained_epochs': 138, 'rejected_epochs': 11, 'rejected_amplitude': 11, 'rejected_high_frequency': 0}
```

### Cell 6 (code)
- Execution count: 5
- Code:
```python
fig, ax = plt.subplots(figsize=(12, 6))
labels = ['total_epochs', 'retained_epochs', 'rejected_epochs']
values = [reject_stats['total_epochs'], reject_stats['retained_epochs'], reject_stats['rejected_epochs']]
ax.bar(labels, values)
ax.set_title('Artifact Rejection Summary')
ax.set_xlabel('Epoch Category')
ax.set_ylabel('Number of Epochs')
ax.grid(True, axis='y', alpha=0.3)
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1200x600 with 1 Axes>
```
- image/png present (base64 length: 26964)

### Cell 7 (code)
- Execution count: 6
- Code:
```python
raw.plot(duration=8, n_channels=min(19, len(raw.ch_names)), scalings='auto');
```

#### Output Block 1 (stream)
```text
Using matplotlib as 2D backend.
```

#### Output Block 2 (display_data)
- Data MIME types: image/png, text/plain
```text
<MNEBrowseFigure size 800x800 with 4 Axes>
```
- image/png present (base64 length: 238044)

### Cell 8 (code)
- Execution count: 7
- Code:
```python
filtered.plot(duration=8, n_channels=min(19, len(filtered.ch_names)), scalings='auto');
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<MNEBrowseFigure size 800x800 with 4 Axes>
```
- image/png present (base64 length: 272324)

### Cell 9 (code)
- Execution count: 8
- Code:
```python
fig = epochs[:5].plot(scalings='auto')
fig
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<MNEBrowseFigure size 800x800 with 4 Axes>
```
- image/png present (base64 length: 365824)

#### Output Block 2 (execute_result)
- Data MIME types: image/png, text/plain
```text
<MNEBrowseFigure size 800x800 with 4 Axes>
```
- image/png present (base64 length: 365824)

## 03_feature_extraction_single_sample.ipynb
- Total cells: 6

### Cell 2 (code)
- Execution count: 1
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt

from dataset_loader import create_subject_metadata
from preprocessing import preprocess_subject
from feature_extraction import compute_welch_psd
from utils import RAW_DATASET_DIR

PALETTE = {
    'navy': '#1f3c88',
    'teal': '#2a9d8f',
    'coral': '#e76f51',
    'gold': '#e9c46a',
    'light': '#f7f7f5',
    'grid': '#d8d8d8',
    'slate': '#3d405b',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': PALETTE['slate'],
    'axes.labelcolor': PALETTE['slate'],
    'axes.titleweight': 'bold',
    'axes.titlesize': 15,
    'font.size': 11,
    'grid.color': PALETTE['grid'],
    'grid.linestyle': '--',
    'grid.alpha': 0.35,
})

def finalize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.35)
    ax.set_axisbelow(True)
```
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code:
```python
metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)
sample_row = metadata_df.iloc[0]
preprocessed = preprocess_subject(sample_row['eeg_file'])
epoch_array = preprocessed['epoch_array']
sfreq = preprocessed['raw'].info['sfreq']

psd_features, freqs = compute_welch_psd(epoch_array, sfreq=sfreq)
log_psd_features, _ = compute_welch_psd(epoch_array, sfreq=sfreq, log_transform=True)

print('PSD shape:', psd_features.shape)
print('Frequency bins:', freqs.shape)
print('First 10 frequencies:', freqs[:10])
```

#### Output Block 1 (stream)
```text
PSD shape: (138, 19, 89)
Frequency bins: (89,)
First 10 frequencies: [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5]
```

### Cell 4 (code)
- Execution count: 3
- Code:
```python
epoch_idx = 0
channel_indices = [0, min(1, psd_features.shape[1] - 1), min(2, psd_features.shape[1] - 1)]
line_colors = [PALETTE['navy'], PALETTE['teal'], PALETTE['coral']]

fig, ax = plt.subplots(figsize=(12.5, 6))
fig.patch.set_facecolor(PALETTE['light'])
for color, channel_idx in zip(line_colors, channel_indices):
    ax.plot(freqs, psd_features[epoch_idx, channel_idx], lw=2.2, color=color, label=f'Channel {channel_idx + 1}')
for start, end in [(1, 4), (4, 8), (8, 13), (13, 30)]:
    ax.axvspan(start, end, color=PALETTE['gold'], alpha=0.08)
ax.set_title('Relative Welch PSD for Selected Channels')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Relative Power')
finalize_axis(ax)
ax.legend(frameon=False)
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1250x600 with 1 Axes>
```
- image/png present (base64 length: 66784)

### Cell 6 (code)
- Execution count: 4
- Code:
```python
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), sharex=True)
fig.patch.set_facecolor(PALETTE['light'])
axes[0].plot(freqs, psd_features[epoch_idx, 0], lw=2.2, color=PALETTE['navy'])
axes[0].fill_between(freqs, psd_features[epoch_idx, 0], color=PALETTE['gold'], alpha=0.18)
axes[0].set_title('Relative-Normalized PSD')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Relative Power')
finalize_axis(axes[0])

axes[1].plot(freqs, log_psd_features[epoch_idx, 0], lw=2.2, color=PALETTE['coral'])
axes[1].fill_between(freqs, log_psd_features[epoch_idx, 0], color=PALETTE['teal'], alpha=0.12)
axes[1].set_title('Log-Transformed Relative PSD')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Log10 Relative Power')
finalize_axis(axes[1])

fig.suptitle('PSD Representation Comparison', fontsize=16, fontweight='bold', color=PALETTE['navy'])
fig.tight_layout()
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1450x550 with 2 Axes>
```
- image/png present (base64 length: 105356)

## 04_feature_extraction_all_subjects.ipynb
- Total cells: 5

### Cell 2 (code)
- Execution count: 1
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_loader import create_subject_metadata
from feature_extraction import build_summary_row, save_psd_features, compute_welch_psd
from preprocessing import PreprocessingConfig, preprocess_subject
from utils import METADATA_DIR, PSD_FEATURE_DIR, RAW_DATASET_DIR, ensure_directories

PALETTE = {
    'navy': '#1f3c88',
    'teal': '#2a9d8f',
    'coral': '#e76f51',
    'gold': '#e9c46a',
    'blue': '#4f6d7a',
    'light': '#f7f7f5',
    'grid': '#d8d8d8',
    'slate': '#3d405b',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': PALETTE['slate'],
    'axes.labelcolor': PALETTE['slate'],
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'font.size': 10.5,
    'grid.color': PALETTE['grid'],
    'grid.linestyle': '--',
    'grid.alpha': 0.35,
})

def finalize_axis(ax, axis='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis=axis, alpha=0.35)
    ax.set_axisbelow(True)
```
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code:
```python
metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)

def run_feature_extraction_for_config(metadata_df, config, output_dir, summary_filename):
    summary_rows = []
    failures = []
    run_rows = []

    for row in metadata_df.itertuples(index=False):
        try:
            preprocessed = preprocess_subject(row.eeg_file, config=config)
            epoch_array = preprocessed['epoch_array']
            if epoch_array.size == 0:
                raise ValueError('No epochs retained after artifact rejection.')

            psd_features, freqs = compute_welch_psd(epoch_array, sfreq=preprocessed['raw'].info['sfreq'])
            output_path = save_psd_features(
                row.subject_id,
                row.class_name,
                int(row.label),
                psd_features,
                freqs,
                output_dir=output_dir,
            )
            summary_rows.append(build_summary_row(row.subject_id, row.class_name, int(row.label), psd_features, output_path))
            run_rows.append({
                'subject_id': row.subject_id,
                'class_name': row.class_name,
                'label': int(row.label),
                'n_epochs': int(epoch_array.shape[0]),
                'rejected_epochs': preprocessed['reject_stats']['rejected_epochs'],
                'retained_epochs': preprocessed['reject_stats']['retained_epochs'],
            })
        except Exception as exc:
            failures.append({'subject_id': row.subject_id, 'class_name': row.class_name, 'error': str(exc)})

    summary_df = pd.DataFrame(summary_rows)
    run_df = pd.DataFrame(run_rows)
    failures_df = pd.DataFrame(failures)

    summary_path = METADATA_DIR / summary_filename
    summary_df.to_csv(summary_path, index=False)

    return summary_df, run_df, failures_df, summary_path

overlap_output_dir = PSD_FEATURE_DIR.parent / 'psd_features_overlap50'
ensure_directories(paths=[METADATA_DIR, PSD_FEATURE_DIR, overlap_output_dir])

config_no = PreprocessingConfig(epoch_duration=4.0, epoch_overlap=0.0)
config_ov = PreprocessingConfig(epoch_duration=4.0, epoch_overlap=2.0)

summary_df_no, run_df_no, failures_df_no, summary_path_no = run_feature_extraction_for_config(
    metadata_df,
    config_no,
    PSD_FEATURE_DIR,
    'psd_feature_summary.csv',
)

summary_df_ov, run_df_ov, failures_df_ov, summary_path_ov = run_feature_extraction_for_config(
    metadata_df,
    config_ov,
    overlap_output_dir,
    'psd_feature_summary_overlap50.csv',
)

comparison_df = pd.DataFrame([
    {
        'setting': 'non_overlap_0pct',
        'epoch_duration_sec': config_no.epoch_duration,
        'overlap_sec': config_no.epoch_overlap,
        'subjects_found': len(metadata_df),
        'successfully_processed': len(summary_df_no),
        'failed_subjects': len(failures_df_no),
        'total_rejected_epochs': int(run_df_no['rejected_epochs'].sum()) if not run_df_no.empty else 0,
        'avg_retained_epochs_per_subject': float(run_df_no['retained_epochs'].mean()) if not run_df_no.empty else 0.0,
    },
    {
        'setting': 'overlap_50pct',
        'epoch_duration_sec': config_ov.epoch_duration,
        'overlap_sec': config_ov.epoch_overlap,
        'subjects_found': len(metadata_df),
        'successfully_processed': len(summary_df_ov),
        'failed_subjects': len(failures_df_ov),
        'total_rejected_epochs': int(run_df_ov['rejected_epochs'].sum()) if not run_df_ov.empty else 0,
        'avg_retained_epochs_per_subject': float(run_df_ov['retained_epochs'].mean()) if not run_df_ov.empty else 0.0,
    },
])

comparison_df
```

#### Output Block 1 (execute_result)
- Data MIME types: text/html, text/plain
```text
            setting  epoch_duration_sec  overlap_sec  subjects_found  \
0  non_overlap_0pct                 4.0          0.0              88   
1     overlap_50pct                 4.0          2.0              88   

   successfully_processed  failed_subjects  total_rejected_epochs  \
0                      88                0                   3244   
1                      88                0                   6490   

   avg_retained_epochs_per_subject  
0                       163.181818  
1                       325.863636  
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setting</th>
      <th>epoch_duration_sec</th>
      <th>overlap_sec</th>
      <th>subjects_found</th>
      <th>successfully_processed</th>
      <th>failed_subjects</th>
      <th>total_rejected_epochs</th>
      <th>avg_retained_epochs_per_subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>non_overlap_0pct</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>88</td>
      <td>88</td>
      <td>0</td>
      <td>3244</td>
      <td>163.181818</td>
    </tr>
    <tr>
      <th>1</th>
      <td>overlap_50pct</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>88</td>
      <td>88</td>
      <td>0</td>
      <td>6490</td>
      <td>325.863636</td>
    </tr>
  </tbody>
</table>
</div>
```

### Cell 4 (code)
- Execution count: 3
- Code:
```python
print('----- Non-overlap (0%) -----')
print('total subjects found:', len(metadata_df))
print('successfully processed:', len(summary_df_no))
print('failed subjects:', len(failures_df_no))
print(f"Total Rejected Epochs: {int(run_df_no['rejected_epochs'].sum()) if not run_df_no.empty else 0}")
print(f"Average Epochs per Subject: {run_df_no['retained_epochs'].mean():.2f}" if not run_df_no.empty else 'Average Epochs per Subject: 0.00')
print('summary CSV:', summary_path_no)

print('')
print('----- Overlap 50% (2s) -----')
print('total subjects found:', len(metadata_df))
print('successfully processed:', len(summary_df_ov))
print('failed subjects:', len(failures_df_ov))
print(f"Total Rejected Epochs: {int(run_df_ov['rejected_epochs'].sum()) if not run_df_ov.empty else 0}")
print(f"Average Epochs per Subject: {run_df_ov['retained_epochs'].mean():.2f}" if not run_df_ov.empty else 'Average Epochs per Subject: 0.00')
print('summary CSV:', summary_path_ov)

print('')
print('----- Comparison -----')
display(comparison_df)

if not failures_df_no.empty:
    print('Non-overlap failures:')
    display(failures_df_no)

if not failures_df_ov.empty:
    print('Overlap 50% failures:')
    display(failures_df_ov)
```

#### Output Block 1 (stream)
```text
----- Non-overlap (0%) -----
total subjects found: 88
successfully processed: 88
failed subjects: 0
Total Rejected Epochs: 3244
Average Epochs per Subject: 163.18
summary CSV: D:\2026 MTECH Project\alz_project\data\metadata\psd_feature_summary.csv

----- Overlap 50% (2s) -----
total subjects found: 88
successfully processed: 88
failed subjects: 0
Total Rejected Epochs: 6490
Average Epochs per Subject: 325.86
summary CSV: D:\2026 MTECH Project\alz_project\data\metadata\psd_feature_summary_overlap50.csv

----- Comparison -----
```

#### Output Block 2 (display_data)
- Data MIME types: text/html, text/plain
```text
            setting  epoch_duration_sec  overlap_sec  subjects_found  \
0  non_overlap_0pct                 4.0          0.0              88   
1     overlap_50pct                 4.0          2.0              88   

   successfully_processed  failed_subjects  total_rejected_epochs  \
0                      88                0                   3244   
1                      88                0                   6490   

   avg_retained_epochs_per_subject  
0                       163.181818  
1                       325.863636  
```
```html
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>setting</th>
      <th>epoch_duration_sec</th>
      <th>overlap_sec</th>
      <th>subjects_found</th>
      <th>successfully_processed</th>
      <th>failed_subjects</th>
      <th>total_rejected_epochs</th>
      <th>avg_retained_epochs_per_subject</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>non_overlap_0pct</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>88</td>
      <td>88</td>
      <td>0</td>
      <td>3244</td>
      <td>163.181818</td>
    </tr>
    <tr>
      <th>1</th>
      <td>overlap_50pct</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>88</td>
      <td>88</td>
      <td>0</td>
      <td>6490</td>
      <td>325.863636</td>
    </tr>
  </tbody>
</table>
</div>
```

### Cell 5 (code)
- Execution count: 4
- Code:
```python
fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
fig.patch.set_facecolor(PALETTE['light'])

# 1) Processed subjects per class (side-by-side)
classes = sorted(set(run_df_no['class_name'].unique()).union(set(run_df_ov['class_name'].unique())))
counts_no = run_df_no.groupby('class_name')['subject_id'].count().reindex(classes, fill_value=0)
counts_ov = run_df_ov.groupby('class_name')['subject_id'].count().reindex(classes, fill_value=0)

x = np.arange(len(classes))
width = 0.36
axes[0].bar(x - width / 2, counts_no.values, width=width, color=PALETTE['navy'], label='0% overlap')
axes[0].bar(x + width / 2, counts_ov.values, width=width, color=PALETTE['teal'], label='50% overlap')
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].set_title('Processed Subjects per Class')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Processed Subjects')
finalize_axis(axes[0], axis='y')
axes[0].legend(frameon=False)

# 2) Average retained/rejected epochs per subject
avg_no_ret = run_df_no['retained_epochs'].mean() if not run_df_no.empty else 0
avg_ov_ret = run_df_ov['retained_epochs'].mean() if not run_df_ov.empty else 0
avg_no_rej = run_df_no['rejected_epochs'].mean() if not run_df_no.empty else 0
avg_ov_rej = run_df_ov['rejected_epochs'].mean() if not run_df_ov.empty else 0

labels = ['Retained', 'Rejected']
x2 = np.arange(len(labels))
axes[1].bar(x2 - width / 2, [avg_no_ret, avg_no_rej], width=width, color=PALETTE['blue'], label='0% overlap')
axes[1].bar(x2 + width / 2, [avg_ov_ret, avg_ov_rej], width=width, color=PALETTE['coral'], label='50% overlap')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(labels)
axes[1].set_title('Average Epochs per Subject')
axes[1].set_ylabel('Average Epoch Count')
finalize_axis(axes[1], axis='y')
axes[1].legend(frameon=False)

# 3) Retained epochs distribution comparison
axes[2].hist(run_df_no['retained_epochs'], bins=20, alpha=0.65, color=PALETTE['navy'], label='0% overlap')
axes[2].hist(run_df_ov['retained_epochs'], bins=20, alpha=0.55, color=PALETTE['teal'], label='50% overlap')
axes[2].set_title('Retained Epoch Distribution')
axes[2].set_xlabel('Retained Epochs')
axes[2].set_ylabel('Number of Subjects')
finalize_axis(axes[2], axis='y')
axes[2].legend(frameon=False)

fig.suptitle('All-Subject Feature Extraction Comparison: Non-overlap vs 50% Overlap', fontsize=16, fontweight='bold', color=PALETTE['navy'])
fig.tight_layout()
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 2000x580 with 3 Axes>
```
- image/png present (base64 length: 94872)

## 05_progress_visualizations.ipynb
- Total cells: 13

### Cell 2 (code)
- Execution count: 1
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import numpy as np

from dataset_loader import create_subject_metadata, scan_eeg_files
from feature_extraction import compute_welch_psd
from preprocessing import preprocess_subject
from utils import FIGURES_DIR, METADATA_DIR, RAW_DATASET_DIR

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = {
    'navy': '#1f3c88',
    'blue': '#4f6d7a',
    'teal': '#2a9d8f',
    'gold': '#e9c46a',
    'coral': '#e76f51',
    'slate': '#3d405b',
    'light': '#f7f7f5',
    'grid': '#d8d8d8',
}
plt.rcParams.update({
    'figure.facecolor': PALETTE['light'],
    'axes.facecolor': 'white',
    'axes.titlesize': 18,
    'axes.titleweight': 'bold',
    'axes.edgecolor': '#c8c8c8',
    'axes.labelcolor': PALETTE['slate'],
    'axes.labelsize': 12,
    'xtick.color': PALETTE['slate'],
    'ytick.color': PALETTE['slate'],
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

def finalize_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=PALETTE['grid'], alpha=0.45)
    return ax
```
- Output: *(no output)*

### Cell 3 (code)
- Execution count: 2
- Code:
```python
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
total_eeg_files = len(scan_eeg_files(RAW_DATASET_DIR))
metadata_df = create_subject_metadata(RAW_DATASET_DIR, verbose=False)
summary_path = METADATA_DIR / 'psd_feature_summary.csv'
summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
class_counts = metadata_df['class_name'].value_counts().sort_index()
sample_row = metadata_df.iloc[0]
preprocessed = preprocess_subject(sample_row['eeg_file'])
epoch_array = preprocessed['epoch_array']
psd_features, freqs = compute_welch_psd(epoch_array, sfreq=preprocessed['raw'].info['sfreq'])
subject_order = summary_df['subject_id'].tolist() if not summary_df.empty else []
epoch_counts = summary_df['n_epochs'].tolist() if not summary_df.empty else []
mean_epochs = float(summary_df['n_epochs'].mean()) if not summary_df.empty else 0.0
```
- Output: *(no output)*

### Cell 4 (code)
- Execution count: 3
- Code:
```python
fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor(PALETTE['light'])
ax.axis('off')
steps = ['Dataset Scan', 'Metadata', 'Preprocessing', 'Artifact Rejection', 'Epoching', 'Welch PSD', 'Saved Features']
x_positions = np.linspace(0.08, 0.92, len(steps))
y_positions = [0.62, 0.38, 0.62, 0.38, 0.62, 0.38, 0.62]
step_colors = [PALETTE['navy'], PALETTE['blue'], PALETTE['teal'], PALETTE['coral'], PALETTE['gold'], PALETTE['teal'], PALETTE['navy']]
for x, y, step, color in zip(x_positions, y_positions, steps, step_colors):
    ax.add_patch(FancyBboxPatch((x - 0.06, y - 0.06), 0.12, 0.12, boxstyle='round,pad=0.02,rounding_size=0.02', fc='white', ec=color, lw=2))
    ax.text(x, y, step, ha='center', va='center', fontsize=11, color=PALETTE['slate'])
for start, end in zip(x_positions[:-1], x_positions[1:]):
    y0 = y_positions[list(x_positions).index(start)]
    y1 = y_positions[list(x_positions).index(end)]
    ax.annotate('', xy=(end - 0.065, y1), xytext=(start + 0.065, y0), arrowprops=dict(arrowstyle='->', lw=2.2, color=PALETTE['slate']))
ax.text(0.03, 0.92, 'Overall Phase 1 Pipeline Diagram', fontsize=20, fontweight='bold', color=PALETTE['navy'])
ax.text(0.03, 0.84, 'From raw EEG discovery to per-subject Welch PSD feature files', fontsize=11, color=PALETTE['slate'])
fig.savefig(FIGURES_DIR / '01_pipeline_diagram.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1600x600 with 1 Axes>
```
- image/png present (base64 length: 57088)

### Cell 5 (code)
- Execution count: 4
- Code:
```python
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(PALETTE['light'])
bars = ax.bar(class_counts.index, class_counts.values, width=0.58, color=[PALETTE['navy'], PALETTE['teal'], PALETTE['coral']])
ax.set_title('Class Distribution')
ax.set_xlabel('Class')
ax.set_ylabel('Number of Subjects')
finalize_axis(ax)
for bar, value in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, str(value), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(class_counts.values) + 6)
fig.savefig(FIGURES_DIR / '02_class_distribution.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1100x600 with 1 Axes>
```
- image/png present (base64 length: 27496)

### Cell 6 (code)
- Execution count: 5
- Code:
```python
raw = preprocessed['raw']
filtered = preprocessed['filtered']
times = raw.times[:int(raw.info['sfreq'] * 8)]
raw_data, _ = raw[:3, :len(times)]
filtered_data, _ = filtered[:3, :len(times)]

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.patch.set_facecolor(PALETTE['light'])
offsets = np.arange(raw_data.shape[0]) * np.max(np.abs(raw_data)) * 2.4
for idx in range(raw_data.shape[0]):
    axes[0].plot(times, raw_data[idx] + offsets[idx], lw=1.2)
    axes[1].plot(times, filtered_data[idx] + offsets[idx], lw=1.2)
axes[0].set_title('Example Raw EEG')
axes[0].set_ylabel('Stacked Channels')
finalize_axis(axes[0])
axes[1].set_title('Example Filtered EEG (0.5-45 Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Stacked Channels')
finalize_axis(axes[1])
axes[1].xaxis.set_major_locator(ticker.MaxNLocator(8))
fig.savefig(FIGURES_DIR / '03_raw_vs_filtered_eeg.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1500x800 with 2 Axes>
```
- image/png present (base64 length: 200512)

### Cell 7 (code)
- Execution count: 6
- Code:
```python
fig, ax = plt.subplots(figsize=(15, 5))
fig.patch.set_facecolor(PALETTE['light'])
epoch_time = np.arange(epoch_array.shape[-1]) / preprocessed['raw'].info['sfreq']
ax.plot(epoch_time, epoch_array[0, 0], lw=1.6, color=PALETTE['navy'])
ax.fill_between(epoch_time, epoch_array[0, 0], alpha=0.18, color=PALETTE['gold'])
ax.set_title('Example Epoch Visualization')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (V)')
finalize_axis(ax)
ax.xaxis.set_major_locator(ticker.MaxNLocator(9))
fig.savefig(FIGURES_DIR / '04_example_epoch.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1500x500 with 1 Axes>
```
- image/png present (base64 length: 120612)

### Cell 8 (code)
- Execution count: 7
- Code:
```python
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(PALETTE['light'])
band_edges = [(1, 4, 'Delta'), (4, 8, 'Theta'), (8, 13, 'Alpha'), (13, 30, 'Beta')]
for start, end, label in band_edges:
    ax.axvspan(start, end, alpha=0.08, color=PALETTE['gold'])
line_colors = [PALETTE['navy'], PALETTE['teal'], PALETTE['coral']]
for channel_idx in range(min(3, psd_features.shape[1])):
    ax.plot(freqs, psd_features[0, channel_idx], lw=2.2, color=line_colors[channel_idx], label=f'Channel {channel_idx + 1}')
ax.set_title('Example PSD Visualization')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Relative Power')
finalize_axis(ax)
ax.legend()
fig.savefig(FIGURES_DIR / '05_example_psd.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1300x600 with 1 Axes>
```
- image/png present (base64 length: 53068)

### Cell 9 (code)
- Execution count: 8
- Code:
```python
fig, ax = plt.subplots(figsize=(13, 4.5))
fig.patch.set_facecolor(PALETTE['light'])
bin_index = np.arange(len(freqs))
ax.plot(freqs, bin_index, lw=1.6, color=PALETTE['blue'])
ax.scatter(freqs, bin_index, s=18, color=PALETTE['navy'])
ax.set_title('Frequency-Bin Summary')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Bin Index')
finalize_axis(ax)
for boundary in [4, 8, 13, 30]:
    ax.axvline(boundary, ls='--', lw=1, alpha=0.6, color=PALETTE['coral'])
fig.savefig(FIGURES_DIR / '06_frequency_bin_summary.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1300x450 with 1 Axes>
```
- image/png present (base64 length: 50572)

### Cell 11 (code)
- Execution count: 9
- Code:
```python
if not summary_df.empty:
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(PALETTE['light'])
    processed_counts = summary_df.groupby('class_name')['subject_id'].count().sort_index()
    bars = ax.bar(processed_counts.index, processed_counts.values, width=0.58, color=[PALETTE['navy'], PALETTE['teal'], PALETTE['coral']])
    ax.set_title('Processed Subjects per Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Processed Subjects')
    finalize_axis(ax)
    for bar, value in zip(bars, processed_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, str(value), ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(processed_counts.values) + 6)
    fig.savefig(FIGURES_DIR / '07_processed_subjects_per_class.png', dpi=240, bbox_inches='tight')
    plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1100x600 with 1 Axes>
```
- image/png present (base64 length: 32092)

### Cell 12 (code)
- Execution count: 10
- Code:
```python
if not summary_df.empty:
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(PALETTE['light'])
    bars = ax.bar(subject_order, epoch_counts, color=PALETTE['blue'])
    ax.set_title('Epoch Counts per Subject')
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Retained Epochs')
    ax.tick_params(axis='x', rotation=90)
    finalize_axis(ax)
    ax.axhline(mean_epochs, ls='--', lw=1.7, alpha=0.9, color=PALETTE['coral'], label=f'Average = {mean_epochs:.2f}')
    ax.legend(loc='upper right')
    fig.savefig(FIGURES_DIR / '08_epoch_counts_per_subject.png', dpi=240, bbox_inches='tight')
    plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1600x600 with 1 Axes>
```
- image/png present (base64 length: 54128)

### Cell 13 (code)
- Execution count: 11
- Code:
```python
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(PALETTE['light'])
ax.axis('off')
class_distribution_text = ', '.join([f"{cls}: {count}" for cls, count in metadata_df['class_name'].value_counts().sort_index().items()])
summary_lines = [
    ('Total EEG files detected', str(total_eeg_files)),
    ('Total subjects processed', str(len(summary_df))),
    ('Class distribution', class_distribution_text),
    ('Example epoch shape', str(epoch_array.shape)),
    ('Example PSD shape', str(psd_features.shape)),
]
ax.text(0.05, 0.92, 'Progress Work Summary', fontsize=20, fontweight='bold', color=PALETTE['navy'])
for idx, (label, value) in enumerate(summary_lines):
    y = 0.78 - idx * 0.13
    ax.add_patch(FancyBboxPatch((0.05, y - 0.04), 0.88, 0.085, boxstyle='round,pad=0.015,rounding_size=0.02', fc='white', ec='#d0d0d0', lw=1.2))
    ax.text(0.08, y, label, fontsize=12, fontweight='bold', color=PALETTE['slate'])
    ax.text(0.42, y, value, fontsize=12, color=PALETTE['navy'])
ax.text(0.06, 0.09, 'Phase 1 complete: Feature extraction ready for CNN modeling', fontsize=14, fontweight='bold', color=PALETTE['teal'])
fig.savefig(FIGURES_DIR / '09_progress_work_summary.png', dpi=240, bbox_inches='tight')
plt.show()
```

#### Output Block 1 (display_data)
- Data MIME types: image/png, text/plain
```text
<Figure size 1300x700 with 1 Axes>
```
- image/png present (base64 length: 64740)

## 06_single_cnn_eegnet_baseline.ipynb
- Total cells: 9

### Cell 2 (code)
- Execution count: None
- Code:
```python
import sys, torch
print(sys.executable)
print(torch.__version__)
```

#### Output Block 1 (stream)
```text
c:\Users\Jediael\AppData\Local\Programs\Python\Python312\python.exe
```

### Cell 3 (code)
- Execution count: None
- Code:
```python
%matplotlib inline
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eegnet_baseline import EEGNetConfig, train_eegnet_baseline
```
- Output: *(no output)*

### Cell 4 (code)
- Execution count: None
- Code:
```python
try:
    import torch  # noqa: F401
    print('PyTorch detected. Ready to train EEGNet.')
except Exception:
    raise RuntimeError('PyTorch is not installed. Run: pip install torch')
```

#### Output Block 1 (error)
- Error: RuntimeError: PyTorch is not installed. Run: pip install torch
```text
[31m---------------------------------------------------------------------------[39m
[31mModuleNotFoundError[39m                       Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[4][39m[32m, line 2[39m
[32m      1[39m [38;5;28;01mtry[39;00m:
[32m----> [39m[32m2[39m     [38;5;28;01mimport[39;00m[38;5;250m [39m[34;01mtorch[39;00m  [38;5;66;03m# noqa: F401[39;00m
[32m      3[39m     [38;5;28mprint[39m([33m'[39m[33mPyTorch detected. Ready to train EEGNet.[39m[33m'[39m)

[31mModuleNotFoundError[39m: No module named 'torch'

During handling of the above exception, another exception occurred:

[31mRuntimeError[39m                              Traceback (most recent call last)
[36mCell[39m[36m [39m[32mIn[4][39m[32m, line 5[39m
[32m      3[39m     [38;5;28mprint[39m([33m'[39m[33mPyTorch detected. Ready to train EEGNet.[39m[33m'[39m)
[32m      4[39m [38;5;28;01mexcept[39;00m [38;5;167;01mException[39;00m:
[32m----> [39m[32m5[39m     [38;5;28;01mraise[39;00m [38;5;167;01mRuntimeError[39;00m([33m'[39m[33mPyTorch is not installed. Run: pip install torch[39m[33m'[39m)

[31mRuntimeError[39m: PyTorch is not installed. Run: pip install torch
```

### Cell 5 (code)
- Execution count: None
- Code:
```python
summary_csv_non_overlap = PROJECT_ROOT / 'data' / 'metadata' / 'psd_feature_summary.csv'
summary_csv_overlap50 = PROJECT_ROOT / 'data' / 'metadata' / 'psd_feature_summary_overlap50.csv'

summary_non_overlap_df = pd.read_csv(summary_csv_non_overlap)
summary_overlap50_df = pd.read_csv(summary_csv_overlap50)

print('Non-overlap summary rows:', len(summary_non_overlap_df))
print('Overlap-50 summary rows:', len(summary_overlap50_df))

summary_non_overlap_df.head()
```
- Output: *(no output)*

### Cell 6 (code)
- Execution count: None
- Code:
```python
cfg = EEGNetConfig(
    epochs=20,
    batch_size=8,
    learning_rate=1e-3,
    test_size=0.2,
    random_state=42,
)

metrics_non_overlap = train_eegnet_baseline(summary_csv_non_overlap, cfg=cfg)
metrics_overlap50 = train_eegnet_baseline(summary_csv_overlap50, cfg=cfg)

comparison_df = pd.DataFrame([
    {'setting': 'non_overlap_0pct', **metrics_non_overlap},
    {'setting': 'overlap_50pct', **metrics_overlap50},
])

comparison_df
```
- Output: *(no output)*

### Cell 7 (code)
- Execution count: None
- Code:
```python
# Presentation-ready comparison plot
fig, ax = plt.subplots(figsize=(11.5, 6.5))
fig.patch.set_facecolor('#f7f7f5')
ax.set_facecolor('white')

settings = comparison_df['setting'].tolist()
train_vals = comparison_df['train_accuracy'].astype(float).values
test_vals = comparison_df['test_accuracy'].astype(float).values

x = np.arange(len(settings))
width = 0.34

train_bars = ax.bar(x - width/2, train_vals, width=width, color='#1f3c88', label='Train Accuracy')
test_bars = ax.bar(x + width/2, test_vals, width=width, color='#2a9d8f', label='Test Accuracy')

ax.set_title('EEGNet Baseline: Non-Overlap vs 50% Overlap', fontsize=15, fontweight='bold', color='#1f3c88')
ax.set_xlabel('Feature Extraction Setting')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(['Non-overlap (0%)', 'Overlap (50%)'])
ax.set_ylim(0, 1.05)
ax.grid(True, axis='y', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='upper left')

for bars in [train_bars, test_bars]:
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Improvement annotation based on test accuracy
delta_test = test_vals[1] - test_vals[0]
improve_txt = f"Test accuracy change (50% overlap - 0% overlap): {delta_test:+.3f}"
ax.text(0.5, -0.18, improve_txt, transform=ax.transAxes, ha='center', va='center', fontsize=10.5,
        color='#3d405b', bbox=dict(boxstyle='round,pad=0.35', facecolor='#eef3f7', edgecolor='#d8d8d8'))

plt.tight_layout()
fig.savefig(PROJECT_ROOT / 'results' / 'figures' / '10_eegnet_accuracy_bar_comparison.png', dpi=220, bbox_inches='tight')
plt.show()
```
- Output: *(no output)*

### Cell 8 (code)
- Execution count: None
- Code:
```python
# Optional publication-style chart: dumbbell comparison (Test Accuracy)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
fig.patch.set_facecolor('#f7f7f5')
ax.set_facecolor('white')

labels = ['Non-overlap (0%)', 'Overlap (50%)']
y = np.arange(len(labels))
train_vals = comparison_df['train_accuracy'].astype(float).values
test_vals = comparison_df['test_accuracy'].astype(float).values

for i in range(len(labels)):
    ax.plot([train_vals[i], test_vals[i]], [y[i], y[i]], color='#8da0ae', lw=3, alpha=0.85)

ax.scatter(train_vals, y, s=120, color='#1f3c88', label='Train', zorder=3)
ax.scatter(test_vals, y, s=120, color='#2a9d8f', label='Test', zorder=3)

for i in range(len(labels)):
    ax.text(train_vals[i] - 0.015, y[i] + 0.12, f"{train_vals[i]:.3f}", color='#1f3c88', fontsize=10, ha='right', va='bottom')
    ax.text(test_vals[i] + 0.015, y[i] + 0.12, f"{test_vals[i]:.3f}", color='#2a9d8f', fontsize=10, ha='left', va='bottom')

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(0, 1.02)
ax.set_xlabel('Accuracy')
ax.set_title('EEGNet Accuracy Shift by Feature Setting (Dumbbell View)', fontsize=14.5, fontweight='bold', color='#1f3c88')
ax.grid(True, axis='x', linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, loc='lower right')

plt.tight_layout()
fig.savefig(PROJECT_ROOT / 'results' / 'figures' / '11_eegnet_accuracy_dumbbell_comparison.png', dpi=220, bbox_inches='tight')
plt.show()
```
- Output: *(no output)*
