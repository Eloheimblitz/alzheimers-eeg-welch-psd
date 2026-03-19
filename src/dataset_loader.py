from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import METADATA_DIR, RAW_DATASET_DIR, log, relative_to_project


SUPPORTED_EEG_EXTENSIONS = (".set", ".edf", ".vhdr", ".fif")
LABEL_MAP = {"AD": 0, "CN": 1, "FTD": 2}
GROUP_ALIAS_MAP = {
    "A": "AD",
    "AD": "AD",
    "ALZHEIMER": "AD",
    "ALZHEIMERS": "AD",
    "C": "CN",
    "CN": "CN",
    "CONTROL": "CN",
    "HC": "CN",
    "HEALTHY": "CN",
    "F": "FTD",
    "FTD": "FTD",
    "FRONTOTEMPORAL": "FTD",
}
SUBJECT_PATTERN = re.compile(r"(sub-\d+)", re.IGNORECASE)


def extract_subject_id(path_like: str | Path) -> str | None:
    match = SUBJECT_PATTERN.search(str(path_like))
    return match.group(1).lower() if match else None


def is_primary_recording(path: Path) -> bool:
    return "derivatives" not in {part.lower() for part in path.parts}


def eeg_priority_key(path: Path) -> tuple[int, str]:
    # Prefer original BIDS EEG files over derivative copies, then sort alphabetically.
    return (0 if is_primary_recording(path) else 1, str(path).lower())


def scan_eeg_files(root_dir: Path = RAW_DATASET_DIR) -> list[Path]:
    root_dir = Path(root_dir)
    eeg_files = [
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EEG_EXTENSIONS
    ]
    return sorted(eeg_files, key=eeg_priority_key)


def choose_subject_recordings(eeg_files: Iterable[Path]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for eeg_file in sorted(eeg_files, key=eeg_priority_key):
        subject_id = extract_subject_id(eeg_file)
        if subject_id is None:
            continue
        selected.setdefault(subject_id, eeg_file)
    return selected


def read_participants(participants_path: Path | None = None) -> pd.DataFrame:
    participants_path = participants_path or (RAW_DATASET_DIR / "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep="\t")
    participants_df["subject_id"] = participants_df["participant_id"].str.lower()
    participants_df["class_name"] = participants_df["Group"].astype(str).str.upper().map(GROUP_ALIAS_MAP)
    participants_df["label"] = participants_df["class_name"].map(LABEL_MAP)
    return participants_df


def create_subject_metadata(
    root_dir: Path = RAW_DATASET_DIR,
    metadata_path: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    eeg_files = scan_eeg_files(root_dir)
    selected_files = choose_subject_recordings(eeg_files)
    participants_df = read_participants(Path(root_dir) / "participants.tsv")

    metadata_df = (
        participants_df[["subject_id", "label", "class_name"]]
        .dropna(subset=["label", "class_name"])
        .drop_duplicates(subset=["subject_id"])
        .copy()
    )
    metadata_df["eeg_file"] = metadata_df["subject_id"].map(selected_files)
    metadata_df = metadata_df.dropna(subset=["eeg_file"]).sort_values("subject_id").reset_index(drop=True)
    metadata_df["eeg_file"] = metadata_df["eeg_file"].map(lambda p: relative_to_project(Path(p)))

    output_path = metadata_path or (METADATA_DIR / "subject_metadata.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)

    if verbose:
        log(f"Total EEG files detected: {len(eeg_files)}")
        log(f"Total matched subjects: {len(metadata_df)}")
        distribution = Counter(metadata_df["class_name"])
        log(f"Class distribution: {dict(distribution)}")
        sample_paths = metadata_df["eeg_file"].head(5).tolist()
        log(f"Sample EEG files: {sample_paths}")

    return metadata_df


def load_subject_metadata(metadata_path: Path | None = None) -> pd.DataFrame:
    metadata_path = metadata_path or (METADATA_DIR / "subject_metadata.csv")
    return pd.read_csv(metadata_path)
