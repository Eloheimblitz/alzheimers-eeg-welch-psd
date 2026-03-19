from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch

from utils import PSD_FEATURE_DIR, relative_to_project


def compute_welch_psd(
    epoch_array: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_fft: int = 1000,
    n_per_seg: int = 1000,
    n_overlap: int = 500,
    window: str = "hamming",
    relative_normalization: bool = True,
    log_transform: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if epoch_array.ndim != 3:
        raise ValueError("Expected epoch_array with shape (n_epochs, n_channels, n_samples).")

    freqs, psd = welch(
        epoch_array,
        fs=sfreq,
        window=window,
        nperseg=n_per_seg,
        noverlap=n_overlap,
        nfft=n_fft,
        axis=-1,
    )

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[band_mask]
    psd = psd[..., band_mask]

    if relative_normalization:
        psd_sum = np.sum(psd, axis=-1, keepdims=True)
        psd = psd / np.maximum(psd_sum, np.finfo(float).eps)

    if log_transform:
        psd = np.log10(np.maximum(psd, np.finfo(float).eps))

    return psd.astype(np.float32), freqs.astype(np.float32)


def save_psd_features(
    subject_id: str,
    class_name: str,
    label: int,
    psd_features: np.ndarray,
    freqs: np.ndarray,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or PSD_FEATURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject_id}_psd_features.npz"

    np.savez_compressed(
        output_path,
        subject_id=subject_id,
        class_name=class_name,
        label=label,
        psd_features=psd_features,
        freqs=freqs,
    )
    return output_path


def load_psd_features(npz_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_summary_row(
    subject_id: str,
    class_name: str,
    label: int,
    psd_features: np.ndarray,
    output_file: str | Path,
) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "class_name": class_name,
        "label": label,
        "n_epochs": int(psd_features.shape[0]),
        "n_channels": int(psd_features.shape[1]),
        "n_freq_bins": int(psd_features.shape[2]),
        "output_file": relative_to_project(Path(output_file)),
    }
