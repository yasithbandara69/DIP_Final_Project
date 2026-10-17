"""Phase-1 unit tests for the CellScope EDU pipeline.

All images are small synthetic arrays generated in-memory. These tests check
pipeline correctness only -- they are NOT project evaluation results.
"""

import cv2
import numpy as np
import pytest

import pipeline as pl


def smear_image(circles, shape=(240, 320), bg=210, fg=80):
    """Light background with dark filled circles [(cx, cy, r), ...]."""
    img = np.full((shape[0], shape[1], 3), bg, dtype=np.uint8)
    for cx, cy, r in circles:
        cv2.circle(img, (cx, cy), r, (fg, fg, fg), -1)
    return img


# --- loading / validation -------------------------------------------------
def test_load_image_from_bytes_valid_png():
    img = smear_image([(100, 100, 25)])
    ok, buf = cv2.imencode(".png", img)
    assert ok
    loaded = pl.load_image_from_bytes(buf.tobytes(), filename="field.png")
    assert loaded.shape == img.shape


def test_load_image_from_bytes_corrupted():
    with pytest.raises(ValueError):
        pl.load_image_from_bytes(b"this is not an image", filename="bad.png")


def test_load_image_from_bytes_unsupported_extension():
    img = smear_image([(50, 50, 10)])
    ok, buf = cv2.imencode(".png", img)
    with pytest.raises(ValueError):
        pl.load_image_from_bytes(buf.tobytes(), filename="field.gif")


def test_load_image_from_path_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pl.load_image_from_path(tmp_path / "nope.png")


def test_validate_image_rejects_none_and_empty():
    with pytest.raises(ValueError):
        pl.validate_image(None)
    with pytest.raises(ValueError):
        pl.validate_image(np.zeros((0, 0, 3), dtype=np.uint8))


def test_validate_image_promotes_grayscale():
    out = pl.validate_image(np.zeros((10, 10), dtype=np.uint8))
    assert out.shape == (10, 10, 3)


# --- metadata / geometry / colour -----------------------------------------
def test_extract_metadata_fields():
    img = smear_image([(100, 100, 20)])
    meta = pl.extract_metadata(img, filename="a.jpg", file_size_bytes=2048)
    assert meta["width"] == 320 and meta["height"] == 240
    assert meta["channels"] == 3 and meta["format"] == "jpg"
    assert meta["file_size_kb"] == 2.0
    assert len(meta["mean_hsv"]) == 3


def test_roi_clamping_and_rotation():
    img = smear_image([(100, 100, 20)])
    cropped, roi = pl.apply_roi_and_rotation(img, roi=(300, 200, 500, 500))
    assert roi == (300, 200, 20, 40)
    assert cropped.shape[:2] == (40, 20)
    rotated, _ = pl.apply_roi_and_rotation(img, rotation_angle=90)
    assert rotated.shape[:2] == (320, 240)
    with pytest.raises(ValueError):
        pl.apply_roi_and_rotation(img, roi=(0, 0, 0, 10))


def test_convert_colour_spaces_shapes():
    gray, hsv = pl.convert_colour_spaces(smear_image([(50, 50, 10)]))
    assert gray.ndim == 2 and hsv.shape[2] == 3


# --- filters / enhancement --------------------------------------------------
def test_compare_noise_filters_metrics():
    rng = np.random.default_rng(0)
    gray = (rng.random((60, 60)) * 255).astype(np.uint8)
    result = pl.compare_noise_filters(gray, kernel=5)
    assert result["gaussian"].shape == gray.shape
    assert result["median"].shape == gray.shape
    for key in ("gaussian_metrics", "median_metrics"):
        assert result[key]["residual_noise"] >= 0
        assert result[key]["laplacian_sharpness"] >= 0


def test_enhance_contrast_methods_and_gamma():
    gray = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
    for method in ("none", "hist_eq", "clahe"):
        out = pl.enhance_contrast(gray, method=method, gamma=1.4)
        assert out.shape == gray.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        pl.enhance_contrast(gray, method="bogus")
    with pytest.raises(ValueError):
        pl.enhance_contrast(gray, gamma=0)


def test_illumination_correction_runs():
    gray = np.full((80, 80), 200, dtype=np.uint8)
    gray[:, :40] = 120  # crude illumination step
    out = pl.correct_illumination(gray)
    assert out.shape == gray.shape and out.dtype == np.uint8


def test_calculate_histograms():
    gray = np.zeros((10, 10), dtype=np.uint8)
    hists = pl.calculate_histograms(gray, gray)
    assert hists["before"].shape == (256,)
    assert hists["before"][0] == 100


# --- segmentation / morphology / edges --------------------------------------
def test_segment_otsu_dark_objects_on_light_background():
    result = pl.process_smear_image(smear_image([(100, 100, 25), (220, 120, 25)]))
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})
    assert result.summary["detected_rbc_like_count"] == 2


def test_segment_hsv_saturated_objects():
    img = np.full((200, 260, 3), 230, dtype=np.uint8)
    cv2.circle(img, (80, 100), 25, (60, 60, 220), -1)   # saturated red-ish
    cv2.circle(img, (180, 100), 25, (60, 60, 220), -1)
    gray, hsv = pl.convert_colour_spaces(img)
    mask = pl.segment_rbc_like_cells(
        gray, hsv=hsv, method="hsv", hsv_lower=(0, 80, 50), hsv_upper=(179, 255, 255)
    )
    mask = pl.clean_binary_mask(mask, min_component_area=30)
    assert np.count_nonzero(mask) > 1000


def test_segment_rejects_unknown_method():
    gray = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        pl.segment_rbc_like_cells(gray, method="bogus")


def test_clean_binary_mask_empty_and_speck_removal():
    empty = np.zeros((50, 50), dtype=np.uint8)
    assert np.count_nonzero(pl.clean_binary_mask(empty)) == 0
    speck = np.zeros((50, 50), dtype=np.uint8)
    speck[10, 10] = 255  # single noise pixel
    cv2.circle(speck, (30, 30), 10, 255, -1)
    cleaned = pl.clean_binary_mask(speck, min_component_area=30)
    assert cleaned[10, 10] == 0 and cleaned[30, 30] == 255


def test_sobel_edges_nonnegative():
    gray, _ = pl.convert_colour_spaces(smear_image([(100, 100, 25)]))
    mag = pl.compute_sobel_edges(gray)
    assert mag.shape == gray.shape and float(mag.min()) >= 0.0
    assert float(mag.max()) > 0.0


# --- full pipeline: counting, watershed, border, flags -----------------------
def test_empty_image_no_cells_no_crash():
    img = np.full((200, 260, 3), 210, dtype=np.uint8)  # blank field
    result = pl.process_smear_image(img)
    assert result.summary["detected_rbc_like_count"] == 0
    assert result.records == []
    assert any("empty mask" in w for w in result.warnings)


def test_single_synthetic_circle():
    result = pl.process_smear_image(smear_image([(160, 120, 30)]))
    assert result.summary["detected_rbc_like_count"] == 1
    rec = result.records[0]
    assert rec["circularity"] > 0.8
    assert rec["equivalent_diameter_px"] == pytest.approx(60, abs=6)
    # Fewer than 5 objects -> batch-relative stats must not be claimed.
    assert result.summary["reference_status"] == pl.INSUFFICIENT_REFERENCE


def test_multiple_separated_circles():
    circles = [(60, 60, 22), (160, 60, 22), (260, 60, 22),
               (60, 170, 22), (160, 170, 22), (260, 170, 22)]
    result = pl.process_smear_image(smear_image(circles))
    assert result.summary["detected_rbc_like_count"] == 6
    assert result.summary["reference_status"] == "ok"


def test_touching_circles_separated_by_watershed():
    # Two circles whose edges overlap (centres 36 px apart, r = 28).
    img = smear_image([(130, 120, 28), (166, 120, 28)])
    on = pl.process_smear_image(
        img, pl.PipelineConfig(use_watershed=True, min_object_area=100)
    )
    off = pl.process_smear_image(
        img, pl.PipelineConfig(use_watershed=False, min_object_area=100)
    )
    assert off.summary["total_objects_found"] == 1  # merged without watershed
    assert on.summary["detected_rbc_like_count"] == 2  # split with watershed


def test_watershed_handles_empty_mask_directly():
    out = pl.separate_touching_cells_watershed(
        np.zeros((40, 40), dtype=np.uint8), np.zeros((40, 40, 3), dtype=np.uint8)
    )
    assert out["num_objects"] == 0 and out["contours"] == []


def test_border_object_excluded():
    # One interior circle + one circle clipped by the left border.
    result = pl.process_smear_image(smear_image([(160, 120, 28), (0, 120, 28)]))
    assert result.summary["border_excluded_count"] == 1
    assert result.summary["detected_rbc_like_count"] == 1
    border_rec = [r for r in result.records if r["touches_border"]]
    assert len(border_rec) == 1
    assert pl.FLAG_BORDER_EXCLUDED in border_rec[0]["review_flags"]


def test_feature_extraction_zero_perimeter_contour():
    gray, hsv = pl.convert_colour_spaces(smear_image([(100, 100, 20)]))
    sobel = pl.compute_sobel_edges(gray)
    degenerate = np.array([[[10, 10]], [[10, 10]]], dtype=np.int32)  # zero area
    records, accepted = pl.extract_cell_features(
        [degenerate], gray, hsv, sobel, min_area=0
    )
    # contourArea == 0 < min_area guard path / no crash, no division by zero.
    assert records == [] and accepted == []


def test_flag_outliers_insufficient_reference():
    records = [{
        "object_id": 1, "area_px": 500.0, "circularity": 0.9,
        "mean_gray": 100.0, "mean_hue": 90.0, "sobel_boundary_strength": 50.0,
        "touches_border": False, "review_flags": "", "flagged": False,
    }]
    flagged, summary = pl.flag_morphology_outliers(records)
    assert summary["reference_status"] == pl.INSUFFICIENT_REFERENCE
    assert flagged[0]["review_flags"] == ""


def test_flag_outliers_detects_irregular_and_cluster():
    base = {
        "circularity": 0.9, "mean_gray": 100.0, "mean_hue": 90.0,
        "sobel_boundary_strength": 50.0, "touches_border": False,
        "review_flags": "", "flagged": False,
    }
    records = []
    for i in range(8):
        records.append(dict(base, object_id=i + 1, area_px=500.0))
    records.append(dict(base, object_id=9, area_px=4000.0))          # cluster-size
    records.append(dict(base, object_id=10, area_px=500.0, circularity=0.15,
                        sobel_boundary_strength=1.0))                # irregular/weak
    flagged, summary = pl.flag_morphology_outliers(records)
    by_id = {r["object_id"]: r for r in flagged}
    assert pl.FLAG_UNRESOLVED_CLUSTER in by_id[9]["review_flags"]
    assert pl.FLAG_IRREGULAR_MORPHOLOGY in by_id[10]["review_flags"]
    assert summary["flagged_count"] >= 2


def test_full_pipeline_roi_and_rotation_config():
    img = smear_image([(160, 120, 28), (260, 120, 28)])
    cfg = pl.PipelineConfig(roi=(100, 60, 220, 120), rotation_angle=180,
                            segmentation_method="adaptive")
    result = pl.process_smear_image(img, cfg)
    assert result.processed_image.shape[:2] == (120, 220)
    assert result.metadata["roi_width"] == 220
    assert result.summary["detected_rbc_like_count"] >= 1


def test_non_diagnostic_language_in_summary():
    result = pl.process_smear_image(smear_image([(160, 120, 30)]))
    text = str(result.summary).lower()
    for banned in ("diagnos", "disease", "patient", "normal", "abnormal"):
        assert banned not in text
