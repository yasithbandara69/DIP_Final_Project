"""CellScope EDU -- dataset manifest utilities (Phase 3).

Validates a dataset manifest CSV against files on disk. This module never
invents records: it only reports what the manifest and filesystem contain.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

import pipeline as pl

MANIFEST_COLUMNS = [
    "filename", "source", "source_url", "student_created", "capture_device",
    "magnification", "resolution", "format", "purpose", "split",
    "ground_truth_count", "annotation_file", "notes",
]

_TRUE_VALUES = {"true", "yes", "y", "1"}


def _load_frame(source: str | Path | bytes | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    text = (
        bytes(source).decode("utf-8-sig") if isinstance(source, (bytes, bytearray))
        else Path(source).read_text(encoding="utf-8-sig")
    )
    return pd.read_csv(io.StringIO(text))


def validate_dataset_manifest(
    manifest: str | Path | bytes | pd.DataFrame,
    dataset_root: str | Path,
    image_subdir: str = "",
    annotation_subdir: str = "",
) -> dict[str, Any]:
    """Validate a manifest against files under ``dataset_root``.

    Reports total images, student-created count, calibration/training vs
    unseen-test split counts, and missing image/annotation files. Unknown
    or blank values are reported as unknown, never guessed.
    """
    frame = _load_frame(manifest)
    missing_columns = [c for c in MANIFEST_COLUMNS if c not in frame.columns]
    root = Path(dataset_root)
    image_dir = root / image_subdir if image_subdir else root
    ann_dir = root / annotation_subdir if annotation_subdir else root

    missing_files: list[str] = []
    missing_annotations: list[str] = []
    for _, row in frame.iterrows():
        filename = str(row.get("filename", "")).strip()
        if not filename or filename.lower() == "nan":
            continue
        if not (image_dir / filename).exists():
            missing_files.append(filename)
        annotation = str(row.get("annotation_file", "")).strip()
        if annotation and annotation.lower() != "nan":
            if not (ann_dir / annotation).exists():
                missing_annotations.append(annotation)

    def _count(series: pd.Series, predicate) -> int:
        return int(sum(1 for v in series if predicate(str(v).strip().lower())))

    splits = frame.get("split", pd.Series(dtype=str))
    student = frame.get("student_created", pd.Series(dtype=str))
    report = {
        "total_images": int(len(frame)),
        "student_created_count": _count(student, lambda v: v in _TRUE_VALUES),
        "calibration_training_count": _count(
            splits, lambda v: v in {"calibration", "training", "train", "cal"}
        ),
        "unseen_test_count": _count(
            splits, lambda v: v in {"test", "unseen_test", "unseen test", "testing"}
        ),
        "split_unknown_count": int(len(frame)) - _count(
            splits,
            lambda v: v in {"calibration", "training", "train", "cal",
                            "test", "unseen_test", "unseen test", "testing"},
        ),
        "missing_files": missing_files,
        "missing_annotations": missing_annotations,
        "missing_columns": missing_columns,
    }
    return report


def scan_image_directory(directory: str | Path) -> list[dict[str, Any]]:
    """List supported images in a directory with measured properties only.

    Source/provenance fields are left blank for a human to fill in honestly
    in the manifest; this function measures width/height/format itself.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in pl.SUPPORTED_EXTENSIONS:
            continue
        try:
            image = pl.load_image_from_path(path)
            meta = pl.extract_metadata(
                image, filename=path.name, file_size_bytes=path.stat().st_size
            )
            records.append({
                "filename": path.name,
                "resolution": meta["dimensions"],
                "format": meta["format"],
                "channels": meta["channels"],
                "file_size_kb": meta["file_size_kb"],
                "status": "ok",
            })
        except Exception as exc:  # corrupted files are reported, not skipped
            records.append({
                "filename": path.name, "resolution": "", "format": path.suffix[1:],
                "channels": "", "file_size_kb": round(path.stat().st_size / 1024, 2),
                "status": f"error: {exc}",
            })
    return records


def empty_manifest_frame() -> pd.DataFrame:
    """An empty manifest DataFrame with the template columns."""
    return pd.DataFrame(columns=MANIFEST_COLUMNS)
