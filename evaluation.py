"""CellScope EDU -- quantitative evaluation utilities (Phase 3).

Compares pipeline predictions against *user-supplied* ground truth. Nothing
in this module invents ground truth or results: if no ground truth is given,
callers must show an instruction instead of scores.

Supported ground truth
----------------------
1. CSV with a filename column and an RBC-count column
   (``ground_truth_count``, ``rbc_count`` or ``count`` accepted).
2. Pascal VOC XML annotations; objects whose label includes ``RBC``
   (case-insensitive, BCCD-style) are the positive class.
3. Optional true binary segmentation masks for IoU / Dice.

Bounding-box matching rule (documented, deterministic)
------------------------------------------------------
Greedy one-to-one matching by descending IoU: all predicted/GT box pairs
with IoU >= threshold are sorted by IoU (highest first); each predicted box
and each GT box may be matched at most once. Unmatched predictions are
false positives, unmatched GT boxes are false negatives.

Count accuracy is reported as ``100 - MAPE`` clipped to [0, 100], where
MAPE is computed only over images whose ground-truth count is > 0
(images with GT = 0 are listed but excluded from MAPE, and counted in
``mape_excluded_zero_gt``).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------
def _read_text(source: str | Path | bytes) -> str:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8-sig")
    return Path(source).read_text(encoding="utf-8-sig")


def load_ground_truth_csv(source: str | Path | bytes) -> dict[str, int]:
    """Load ``{filename: ground-truth RBC count}`` from a CSV.

    Raises ``ValueError`` with a friendly message for missing columns,
    non-numeric or negative counts, and unreadable files.
    """
    try:
        text = _read_text(source)
        frame = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise ValueError(f"Could not read ground-truth CSV: {exc}") from exc
    columns = {c.lower().strip(): c for c in frame.columns}
    name_col = next(
        (columns[k] for k in ("filename", "file", "image", "name") if k in columns),
        None,
    )
    count_col = next(
        (columns[k] for k in ("ground_truth_count", "rbc_count", "gt_count", "count")
         if k in columns),
        None,
    )
    if name_col is None or count_col is None:
        raise ValueError(
            "Ground-truth CSV must contain a 'filename' column and a count "
            "column ('ground_truth_count', 'rbc_count', or 'count')."
        )
    result: dict[str, int] = {}
    for _, row in frame.iterrows():
        filename = str(row[name_col]).strip()
        if not filename or filename.lower() == "nan":
            continue
        try:
            count = int(float(row[count_col]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid count for '{filename}': {row[count_col]!r}."
            ) from exc
        if count < 0:
            raise ValueError(f"Count for '{filename}' is negative ({count}).")
        result[Path(filename).name] = count
    if not result:
        raise ValueError("Ground-truth CSV contained no usable rows.")
    return result


def parse_voc_xml(source: str | Path | bytes) -> dict[str, Any]:
    """Parse one Pascal VOC XML file.

    Returns ``{'filename', 'width', 'height', 'objects'}`` where each object
    is ``{'label': str, 'bbox': (xmin, ymin, xmax, ymax)}``. Raises
    ``ValueError`` for corrupted XML or invalid boxes.
    """
    try:
        root = ET.fromstring(_read_text(source))
    except ET.ParseError as exc:
        raise ValueError(f"Corrupted annotation XML: {exc}") from exc
    filename = root.findtext("filename") or ""
    size = root.find("size")
    width = int(float(size.findtext("width", 0))) if size is not None else 0
    height = int(float(size.findtext("height", 0))) if size is not None else 0
    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            bbox: Box = tuple(
                float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")
            )  # type: ignore[assignment]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid bounding box in XML for '{filename}'.") from exc
        # Zero-area boxes occur in real datasets (point annotations); they
        # cannot participate in IoU matching, so they are skipped.
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        objects.append({"label": label, "bbox": bbox})
    return {
        "filename": Path(filename).name if filename else "",
        "width": width,
        "height": height,
        "objects": objects,
    }


def load_voc_annotations(
    sources: Sequence[str | Path | bytes],
    filenames: Optional[Sequence[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Parse many VOC files into ``{filename: parsed_annotation}``.

    ``filenames`` (parallel to ``sources``) overrides the name stored inside
    each XML, which is useful for uploaded files.
    """
    result: dict[str, dict[str, Any]] = {}
    for i, src in enumerate(sources):
        parsed = parse_voc_xml(src)
        raw_name = ""
        if filenames and i < len(filenames):
            raw_name = filenames[i]
        raw_name = raw_name or parsed["filename"] or f"annotation_{i}"
        # Index by stem so 'BloodImage_00012.xml' matches 'BloodImage_00012.jpg'.
        result[Path(raw_name).stem] = parsed
    return result


def rbc_boxes(annotation: dict[str, Any]) -> list[Box]:
    """Return boxes whose label includes 'RBC' (BCCD-style), else empty."""
    return [
        tuple(float(v) for v in o["bbox"])
        for o in annotation.get("objects", [])
        if "rbc" in str(o.get("label", "")).lower()
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def scale_bbox(
    bbox: Box,
    original_size: tuple[int, int],
    new_size: tuple[int, int],
) -> Box:
    """Scale ``(xmin, ymin, xmax, ymax)`` from original to new (w, h) size."""
    ow, oh = original_size
    nw, nh = new_size
    if ow <= 0 or oh <= 0 or nw <= 0 or nh <= 0:
        raise ValueError("Image sizes must be positive for annotation scaling.")
    sx, sy = nw / ow, nh / oh
    return (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)


def bbox_iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes; 0.0 for degenerate inputs."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    predicted: Sequence[Box],
    ground_truth: Sequence[Box],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Greedy one-to-one matching by descending IoU (see module docstring)."""
    pairs: list[tuple[float, int, int]] = []
    for pi, p in enumerate(predicted):
        for gi, g in enumerate(ground_truth):
            value = bbox_iou(tuple(float(v) for v in p), tuple(float(v) for v in g))
            if value >= iou_threshold:
                pairs.append((value, pi, gi))
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi, value))
    return {
        "tp": len(matches),
        "fp": len(predicted) - len(matches),
        "fn": len(ground_truth) - len(matches),
        "matches": matches,
        "mean_matched_iou": float(np.mean([m[2] for m in matches])) if matches else 0.0,
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision/recall/F1 with zero-division guards."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return float(precision), float(recall), float(f1)


# ---------------------------------------------------------------------------
# Mask metrics (only for real ground-truth masks)
# ---------------------------------------------------------------------------
def mask_iou_dice(predicted_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """IoU and Dice for two binary masks of equal shape (values >0 = fg)."""
    if predicted_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Mask shapes differ: {predicted_mask.shape} vs {gt_mask.shape}."
        )
    pred = predicted_mask > 0
    gt = gt_mask > 0
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = inter / union if union > 0 else 1.0  # both empty = perfect agreement
    dice_den = int(pred.sum()) + int(gt.sum())
    dice = (2 * inter / dice_den) if dice_den > 0 else 1.0
    return float(iou), float(dice)


# ---------------------------------------------------------------------------
# Count evaluation
# ---------------------------------------------------------------------------
def evaluate_counts(per_image: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute per-image rows + summary from count pairs.

    Each input item needs ``filename``, ``predicted_count``,
    ``ground_truth_count`` and optionally ``runtime_seconds``. Returns a
    DataFrame with signed/abs/percentage errors and a summary dict with
    MAE, MAPE, count accuracy (100 - MAPE clipped to [0, 100]) and mean
    runtime. GT=0 images are excluded from MAPE and reported in
    ``mape_excluded_zero_gt``.
    """
    rows: list[dict[str, Any]] = []
    for item in per_image:
        gt = int(item["ground_truth_count"])
        pred = int(item["predicted_count"])
        error = pred - gt
        rows.append({
            "filename": item["filename"],
            "predicted_count": pred,
            "ground_truth_count": gt,
            "signed_error": error,
            "absolute_error": abs(error),
            "percentage_error": (100.0 * abs(error) / gt) if gt > 0 else None,
            "runtime_seconds": item.get("runtime_seconds"),
            **{k: item[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1",
                                    "iou", "dice") if k in item},
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {
            "image_count": 0, "mae": None, "mape": None,
            "count_accuracy": None, "mean_runtime_seconds": None,
            "mape_excluded_zero_gt": 0,
        }
    pct = frame["percentage_error"].dropna()
    mape = float(pct.mean()) if len(pct) else None
    runtimes = frame["runtime_seconds"].dropna()
    summary = {
        "image_count": int(len(frame)),
        "mae": float(frame["absolute_error"].mean()),
        "mape": mape,
        "count_accuracy": (
            float(min(100.0, max(0.0, 100.0 - mape))) if mape is not None else None
        ),
        "mean_runtime_seconds": float(runtimes.mean()) if len(runtimes) else None,
        "total_predicted": int(frame["predicted_count"].sum()),
        "total_ground_truth": int(frame["ground_truth_count"].sum()),
        "mape_excluded_zero_gt": int(frame["percentage_error"].isna().sum()),
    }
    # Aggregate box metrics when present.
    if "tp" in frame.columns:
        tp, fp, fn = (int(frame[c].sum()) for c in ("tp", "fp", "fn"))
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        summary.update({"tp": tp, "fp": fp, "fn": fn,
                        "precision": p, "recall": r, "f1": f1})
    if "iou" in frame.columns:
        summary["mean_iou"] = float(frame["iou"].dropna().mean())
    if "dice" in frame.columns:
        summary["mean_dice"] = float(frame["dice"].dropna().mean())
    return frame, summary


def summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """One-row DataFrame of a summary dict, for CSV export."""
    return pd.DataFrame([summary])
