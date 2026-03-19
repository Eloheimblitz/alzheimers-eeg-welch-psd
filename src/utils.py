from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_DIR = DATA_DIR / "raw" / "ds004504"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
PSD_FEATURE_DIR = PROCESSED_DIR / "psd_features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path = PROJECT_ROOT
    raw_dataset_dir: Path = RAW_DATASET_DIR
    metadata_dir: Path = METADATA_DIR
    processed_dir: Path = PROCESSED_DIR
    psd_feature_dir: Path = PSD_FEATURE_DIR
    results_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    target_paths = list(paths) if paths is not None else [
        DATA_DIR,
        RAW_DATASET_DIR,
        METADATA_DIR,
        PROCESSED_DIR,
        PSD_FEATURE_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
    ]
    for path in target_paths:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[alz_project] {message}")


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())
