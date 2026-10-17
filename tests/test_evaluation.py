"""Phase-3 tests for evaluation.py and dataset_utils.py (synthetic only)."""

import numpy as np
import pandas as pd
import pytest

import dataset_utils as ds
import evaluation as ev

VOC = """<annotation>
<filename>BloodImage_00001.jpg</filename>
<size><width>640</width><height>480</height><depth>3</depth></size>
<object><name>RBC</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
<object><name>RBC</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>140</xmax><ymax>145</ymax></bndbox></object>
<object><name>WBC</name><bndbox><xmin>200</xmin><ymin>200</ymin><xmax>260</xmax><ymax>260</ymax></bndbox></object>
<object><name>Platelets</name><bndbox><xmin>300</xmin><ymin>300</ymin><xmax>310</xmax><ymax>310</ymax></bndbox></object>
</annotation>"""


# --- CSV ground truth -------------------------------------------------------
def test_load_ground_truth_csv():
    csv = b"filename,ground_truth_count\nimg1.jpg,42\nimg2.jpg,0\n"
    assert ev.load_ground_truth_csv(csv) == {"img1.jpg": 42, "img2.jpg": 0}


def test_load_ground_truth_csv_accepts_rbc_count_column():
    csv = b"filename,rbc_count\na.png,7\n"
    assert ev.load_ground_truth_csv(csv) == {"a.png": 7}


def test_load_ground_truth_csv_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ev.load_ground_truth_csv(b"filename,other\na.jpg,1\n")
    with pytest.raises(ValueError):
        ev.load_ground_truth_csv(b"filename,ground_truth_count\na.jpg,three\n")
    with pytest.raises(ValueError):
        ev.load_ground_truth_csv(b"filename,ground_truth_count\na.jpg,-2\n")
    with pytest.raises(ValueError):
        ev.load_ground_truth_csv(b"not a csv at all \x00\x01")


# --- VOC XML ------------------------------------------------------------------
def test_parse_voc_xml_and_rbc_filter():
    ann = ev.parse_voc_xml(VOC.encode())
    assert ann["filename"] == "BloodImage_00001.jpg"
    assert (ann["width"], ann["height"]) == (640, 480)
    boxes = ev.rbc_boxes(ann)
    assert len(boxes) == 2
    assert boxes[0] == (10.0, 10.0, 50.0, 50.0)


def test_parse_voc_xml_corrupted():
    with pytest.raises(ValueError):
        ev.parse_voc_xml(b"<annotation><object>not closed")
    with pytest.raises(ValueError):
        ev.parse_voc_xml(b"")


def test_load_voc_annotations_indexed_by_stem():
    anns = ev.load_voc_annotations([VOC.encode()], filenames=["BloodImage_00001.xml"])
    assert "BloodImage_00001" in anns
    assert len(ev.rbc_boxes(anns["BloodImage_00001"])) == 2


# --- scaling / IoU / matching --------------------------------------------------
def test_scale_bbox():
    assert ev.scale_bbox((10, 20, 30, 40), (100, 100), (200, 50)) == (20, 10, 60, 20)
    with pytest.raises(ValueError):
        ev.scale_bbox((0, 0, 1, 1), (0, 100), (200, 200))


def test_bbox_iou():
    assert ev.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert ev.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert ev.bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_match_boxes_greedy_one_to_one():
    pred = [(0, 0, 10, 10), (1, 1, 11, 11)]  # both overlap GT box 0
    gt = [(0, 0, 10, 10), (50, 50, 60, 60)]
    out = ev.match_boxes(pred, gt, iou_threshold=0.5)
    assert out["tp"] == 1 and out["fp"] == 1 and out["fn"] == 1
    # exact match wins the GT box (highest IoU first)
    assert out["matches"][0][0] == 0


def test_precision_recall_f1_and_zero_guards():
    p, r, f1 = ev.precision_recall_f1(8, 2, 2)
    assert p == pytest.approx(0.8) and r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)
    assert ev.precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
    assert ev.precision_recall_f1(0, 5, 0) == (0.0, 0.0, 0.0)


def test_mask_iou_dice():
    a = np.zeros((10, 10), dtype=np.uint8); a[0:5, 0:5] = 255
    b = np.zeros((10, 10), dtype=np.uint8); b[0:5, 0:10] = 255
    iou, dice = ev.mask_iou_dice(a, b)
    assert iou == pytest.approx(0.5)
    assert dice == pytest.approx(2 * 25 / 75)
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert ev.mask_iou_dice(empty, empty) == (1.0, 1.0)  # no division by zero
    with pytest.raises(ValueError):
        ev.mask_iou_dice(a, np.zeros((5, 5), dtype=np.uint8))


# --- count evaluation ----------------------------------------------------------
def _per_image():
    return [
        {"filename": "a.jpg", "predicted_count": 10, "ground_truth_count": 12,
         "runtime_seconds": 0.5},
        {"filename": "b.jpg", "predicted_count": 8, "ground_truth_count": 8,
         "runtime_seconds": 1.5},
        {"filename": "c.jpg", "predicted_count": 3, "ground_truth_count": 0,
         "runtime_seconds": 1.0},
    ]


def test_evaluate_counts_metrics():
    frame, summary = ev.evaluate_counts(_per_image())
    row_a = frame.set_index("filename").loc["a.jpg"]
    assert row_a["signed_error"] == -2 and row_a["absolute_error"] == 2
    assert row_a["percentage_error"] == pytest.approx(100 * 2 / 12)
    assert summary["mae"] == pytest.approx((2 + 0 + 3) / 3)
    # MAPE excludes the GT=0 image: mean of (16.67%, 0%)
    assert summary["mape"] == pytest.approx((100 * 2 / 12 + 0) / 2)
    assert summary["count_accuracy"] == pytest.approx(100 - summary["mape"])
    assert summary["mape_excluded_zero_gt"] == 1
    assert summary["mean_runtime_seconds"] == pytest.approx(1.0)


def test_evaluate_counts_empty():
    frame, summary = ev.evaluate_counts([])
    assert frame.empty and summary["image_count"] == 0 and summary["mae"] is None


def test_evaluate_counts_aggregates_box_metrics():
    items = _per_image()
    items[0].update(tp=5, fp=1, fn=2, precision=0.83, recall=0.71, f1=0.77)
    items[1].update(tp=3, fp=0, fn=1, precision=1.0, recall=0.75, f1=0.86)
    items[2].update(tp=0, fp=3, fn=0, precision=0.0, recall=0.0, f1=0.0)
    _, summary = ev.evaluate_counts(items)
    assert (summary["tp"], summary["fp"], summary["fn"]) == (8, 4, 3)
    p, r, f1 = ev.precision_recall_f1(8, 4, 3)
    assert summary["f1"] == pytest.approx(f1)


# --- dataset manifest utilities -------------------------------------------------
def _manifest(tmp_path, rows):
    df = pd.DataFrame(rows, columns=ds.MANIFEST_COLUMNS)
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    return path


def test_validate_manifest_reports(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "a.xml").write_bytes(b"x")
    manifest = _manifest(tmp_path, [
        ["a.jpg", "BCCD", "https://example", "false", "", "", "640x480", "jpg",
         "evaluation", "unseen_test", 12, "a.xml", ""],
        ["gone.jpg", "student", "", "true", "microscope", "", "640x480", "jpg",
         "calibration", "calibration", "", "gone.xml", ""],
    ])
    report = ds.validate_dataset_manifest(
        manifest, tmp_path
    )
    assert report["total_images"] == 2
    assert report["student_created_count"] == 1
    assert report["calibration_training_count"] == 1
    assert report["unseen_test_count"] == 1
    assert report["missing_files"] == ["gone.jpg"]
    assert report["missing_annotations"] == ["gone.xml"]
    assert report["missing_columns"] == []


def test_validate_manifest_missing_columns(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame([{"filename": "a.jpg"}]).to_csv(path, index=False)
    report = ds.validate_dataset_manifest(path, tmp_path)
    assert "source" in report["missing_columns"]


def test_scan_image_directory(tmp_path):
    import cv2
    img = np.full((20, 30, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "ok.jpg"), img)
    (tmp_path / "bad.jpg").write_bytes(b"junk")
    (tmp_path / "ignore.txt").write_text("x")
    records = ds.scan_image_directory(tmp_path)
    by_name = {r["filename"]: r for r in records}
    assert by_name["ok.jpg"]["status"] == "ok"
    assert by_name["ok.jpg"]["resolution"] == "30 x 20"
    assert by_name["bad.jpg"]["status"].startswith("error")
    assert "ignore.txt" not in by_name
    with pytest.raises(FileNotFoundError):
        ds.scan_image_directory(tmp_path / "nope")


def test_real_bccd_annotations_if_present():
    """Integration check using the downloaded BCCD files, when present."""
    from pathlib import Path
    ann_dir = Path("dataset/annotations")
    img_dir = Path("dataset/raw/test")
    if not ann_dir.is_dir() or not any(ann_dir.glob("*.xml")):
        pytest.skip("BCCD annotation files not present in this checkout")
    files = sorted(ann_dir.glob("*.xml"))
    anns = ev.load_voc_annotations(
        [f.read_bytes() for f in files], filenames=[f.name for f in files]
    )
    total_rbc = sum(len(ev.rbc_boxes(a)) for a in anns.values())
    assert len(anns) == len(files)
    assert total_rbc > 0
    # Every annotation has a matching image file.
    for f in files:
        assert (img_dir / f"{f.stem}.jpg").exists()
