from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import welch

from utils import PROJECT_ROOT


READERS = {
    ".set": mne.io.read_raw_eeglab,
    ".edf": mne.io.read_raw_edf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".fif": mne.io.read_raw_fif,
}


@dataclass
class PreprocessingConfig:
    l_freq: float = 0.5
    h_freq: float = 45.0
    epoch_duration: float = 4.0
    amplitude_threshold_uv: float = 150.0
    hf_band: tuple[float, float] = (20.0, 45.0)
    hf_ratio_threshold: float = 0.35


def resolve_eeg_path(eeg_file: str | Path) -> Path:
    path = Path(eeg_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_raw_eeg(eeg_file: str | Path, preload: bool = True, verbose: str = "ERROR") -> mne.io.BaseRaw:
    path = resolve_eeg_path(eeg_file)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"Unsupported EEG file format: {path.suffix}")

    raw = reader(path, preload=preload, verbose=verbose)
    raw.pick("eeg")
    if raw.info.get("sfreq", 0) <= 0:
        raise ValueError(f"Invalid sampling frequency for {path}")
    return raw


def filter_raw(raw: mne.io.BaseRaw, config: PreprocessingConfig | None = None) -> mne.io.BaseRaw:
    config = config or PreprocessingConfig()
    filtered = raw.copy()
    filtered.filter(l_freq=config.l_freq, h_freq=config.h_freq, verbose="ERROR")
    return filtered


def make_fixed_length_epochs(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig | None = None,
) -> mne.Epochs:
    config = config or PreprocessingConfig()
    return mne.make_fixed_length_epochs(
        raw,
        duration=config.epoch_duration,
        overlap=0.0,
        preload=True,
        verbose="ERROR",
    )


def _high_frequency_ratio(epoch_data: np.ndarray, sfreq: float, hf_band: tuple[float, float]) -> float:
    freqs, psd = welch(epoch_data, fs=sfreq, axis=-1, nperseg=min(epoch_data.shape[-1], 512))
    total_power = np.sum(psd, axis=-1)
    band_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    hf_power = np.sum(psd[..., band_mask], axis=-1)
    ratio = hf_power / np.maximum(total_power, np.finfo(float).eps)
    return float(np.max(ratio))


def reject_artifacts(
    epochs: mne.Epochs,
    config: PreprocessingConfig | None = None,
) -> tuple[mne.Epochs, dict[str, int]]:
    config = config or PreprocessingConfig()
    data = epochs.get_data(copy=True)
    sfreq = float(epochs.info["sfreq"])
    amplitude_threshold_v = config.amplitude_threshold_uv * 1e-6

    keep_mask = np.ones(len(data), dtype=bool)
    rejected_amplitude = 0
    rejected_hf = 0

    for idx, epoch_data in enumerate(data):
        max_abs_amplitude = float(np.max(np.abs(epoch_data)))
        if max_abs_amplitude > amplitude_threshold_v:
            keep_mask[idx] = False
            rejected_amplitude += 1
            continue

        hf_ratio = _high_frequency_ratio(epoch_data, sfreq=sfreq, hf_band=config.hf_band)
        if hf_ratio > config.hf_ratio_threshold:
            keep_mask[idx] = False
            rejected_hf += 1

    retained_epochs = epochs[keep_mask]
    stats = {
        "total_epochs": int(len(data)),
        "retained_epochs": int(np.sum(keep_mask)),
        "rejected_epochs": int(np.sum(~keep_mask)),
        "rejected_amplitude": rejected_amplitude,
        "rejected_high_frequency": rejected_hf,
    }
    return retained_epochs, stats


def epochs_to_numpy(epochs: mne.Epochs) -> np.ndarray:
    return epochs.get_data(copy=True)


def preprocess_subject(
    eeg_file: str | Path,
    config: PreprocessingConfig | None = None,
) -> dict[str, object]:
    config = config or PreprocessingConfig()
    raw = load_raw_eeg(eeg_file)
    filtered = filter_raw(raw, config=config)
    epochs = make_fixed_length_epochs(filtered, config=config)
    clean_epochs, reject_stats = reject_artifacts(epochs, config=config)
    epoch_array = epochs_to_numpy(clean_epochs)

    return {
        "raw": raw,
        "filtered": filtered,
        "epochs": clean_epochs,
        "epoch_array": epoch_array,
        "reject_stats": reject_stats,
    }


def describe_preprocessed_subject(subject_id: str, class_name: str, preprocessed: dict[str, object]) -> dict[str, object]:
    raw = preprocessed["raw"]
    epoch_array = preprocessed["epoch_array"]
    return {
        "subject_id": subject_id,
        "class_name": class_name,
        "n_channels": len(raw.ch_names),
        "sampling_frequency": float(raw.info["sfreq"]),
        "n_epochs": int(epoch_array.shape[0]),
        "epoch_shape": tuple(epoch_array.shape),
    }
