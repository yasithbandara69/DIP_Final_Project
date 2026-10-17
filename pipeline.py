"""CellScope EDU -- core image-processing pipeline (Phase 1).

Educational prototype for image-based counting and morphology screening of
RBC-like objects in stained blood-smear microscopy images.

This module is pure image processing: it contains **no** Streamlit code, no
clinical/diagnostic logic, and no pill/tablet/tray logic.  All review flags
are image-relative statistical descriptions (e.g. "Irregular morphology
review"), never diagnoses.

Pipeline order (Part B + Part C of SE3071):
    load/validate -> ROI + rotation -> grayscale/HSV -> noise-filter
    comparison/selection -> illumination correction -> contrast enhancement
    (histogram equalisation / CLAHE / gamma) -> segmentation (Otsu / HSV /
    adaptive / manual) -> morphological cleanup -> Sobel boundary strength
    -> distance-transform watershed -> contour features -> robust
    (median/MAD) morphology review flags -> annotation.

Only OpenCV + NumPy are used.  Masks are always ``uint8`` with values 0/255.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Non-diagnostic review-flag labels (single source of truth).
FLAG_UNRESOLVED_CLUSTER = "Unresolved cluster"
FLAG_POSSIBLE_DEBRIS = "Possible debris/fragment"
FLAG_IRREGULAR_MORPHOLOGY = "Irregular morphology review"
FLAG_STAINING_OUTLIER = "Staining outlier review"
FLAG_LOW_BOUNDARY = "Low boundary confidence"
FLAG_BORDER_EXCLUDED = "Border object excluded"
INSUFFICIENT_REFERENCE = "Insufficient reference objects"


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """All adjustable image-processing parameters (never clinical values)."""

    # ROI / geometry: (x, y, width, height) in pixels, or None for full frame.
    roi: Optional[tuple[int, int, int, int]] = None
    rotation_angle: float = 0.0  # degrees, counter-clockwise positive
    # Filtering: "gaussian", "median" or "none"; kernel forced to odd >= 3.
    filter_method: str = "median"
    filter_kernel: int = 5
    # Enhancement.
    illumination_correction: bool = True
    illumination_kernel: int = 81  # must exceed typical object diameter (px)
    enhancement_method: str = "clahe"  # "none" | "hist_eq" | "clahe"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    gamma: float = 1.0  # 1.0 = no gamma correction
    # Segmentation: "otsu" | "hsv" | "adaptive" | "manual"
    segmentation_method: str = "otsu"
    invert_foreground: bool = True  # cells darker/more stained than background
    manual_threshold: int = 127
    adaptive_block_size: int = 51
    adaptive_c: float = 5.0
    hsv_lower: tuple[int, int, int] = (0, 40, 0)
    hsv_upper: tuple[int, int, int] = (179, 255, 255)
    # Morphological cleanup.
    morph_kernel: int = 3
    morph_iterations: int = 1
    min_component_area: int = 30  # foreground specks smaller than this are removed
    # Object acceptance / watershed.
    min_object_area: int = 50
    use_watershed: bool = True
    watershed_peak_kernel: int = 15  # local-maxima neighbourhood for DT peaks
    watershed_fg_ratio: float = 0.35  # DT peak must exceed ratio * DT maximum
    border_margin: int = 2
    exclude_border_objects: bool = True
    # Robust flagging.
    min_reference_objects: int = 5
    robust_z_threshold: float = 3.5


@dataclass
class ProcessingResult:
    """Structured result returned by :func:`process_smear_image`."""

    processed_image: np.ndarray  # ROI/rotated BGR image actually analysed
    gray: np.ndarray
    hsv: np.ndarray
    filtered_gray: np.ndarray
    enhanced_gray: np.ndarray
    mask: np.ndarray  # cleaned binary mask, uint8 0/255
    sobel_magnitude: np.ndarray  # float32 gradient magnitude
    watershed_image: np.ndarray  # labelled-region visualisation
    watershed_labels: np.ndarray
    annotated_image: np.ndarray
    histograms: dict[str, np.ndarray]
    filter_comparison: dict[str, Any]
    records: list[dict[str, Any]]  # one dict per detected object (CSV-ready)
    contours: list[np.ndarray]
    summary: dict[str, Any]
    metadata: dict[str, Any]
    runtime_seconds: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Image acquisition and validation
# ---------------------------------------------------------------------------
def validate_image(image: Optional[np.ndarray]) -> np.ndarray:
    """Validate *image* and return it normalised to a 3-channel BGR array.

    Raises:
        ValueError: with a user-friendly message if the image is missing,
            empty, or has an unsupported shape/dtype layout.
    """
    if image is None:
        raise ValueError("No image data was provided or the file could not be decoded.")
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image).__name__}.")
    if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("The image is empty (zero width or height).")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}.")


def load_image_from_bytes(data: bytes, filename: Optional[str] = None) -> np.ndarray:
    """Decode an image from uploaded *bytes* using ``cv2.imdecode``.

    No temporary file is written. Raises ``ValueError`` for empty, corrupted,
    or unsupported input.
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Please use PNG, JPG, or JPEG."
            )
    if not data:
        raise ValueError("The uploaded file is empty.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("The file could not be decoded. It may be corrupted or not an image.")
    return validate_image(image)


def load_image_from_path(path: str | Path) -> np.ndarray:
    """Load and validate an image from a filesystem *path*."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{file_path.suffix}'. Please use PNG, JPG, or JPEG."
        )
    image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unsupported or corrupted image file: {file_path}")
    return validate_image(image)


def extract_metadata(
    image: np.ndarray,
    filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> dict[str, Any]:
    """Extract display metadata: size, channels, dtype, format, ROI, stats."""
    bgr = validate_image(image)
    gray, hsv = convert_colour_spaces(bgr)
    height, width = bgr.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    roi_info: dict[str, Any] = {}
    if roi is not None:
        x, y, w, h = roi
        roi_info = {"roi_x": x, "roi_y": y, "roi_width": w, "roi_height": h}
    return {
        "filename": filename,
        "format": Path(filename).suffix.lower().lstrip(".") if filename else None,
        "width": int(width),
        "height": int(height),
        "dimensions": f"{width} x {height}",
        "channels": int(channels),
        "dtype": str(bgr.dtype),
        "bit_depth": int(np.iinfo(bgr.dtype).bits) if np.issubdtype(bgr.dtype, np.integer) else None,
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024.0, 2) if file_size_bytes else None,
        "mean_gray": float(np.mean(gray)),
        "std_gray": float(np.std(gray)),
        "mean_hsv": [float(v) for v in cv2.mean(hsv)[:3]],
        **roi_info,
    }


# ---------------------------------------------------------------------------
# 2. Geometry: ROI + rotation
# ---------------------------------------------------------------------------
def apply_roi_and_rotation(
    image: np.ndarray,
    roi: Optional[tuple[int, int, int, int]] = None,
    rotation_angle: float = 0.0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop to *roi* (clamped to the image) then rotate by *rotation_angle*.

    Returns ``(processed_image, effective_roi)`` where ``effective_roi`` is
    the clamped ``(x, y, width, height)`` actually used. No physical/clinical
    measurement is performed -- sizes remain in processed pixels.
    """
    bgr = validate_image(image)
    height, width = bgr.shape[:2]
    if roi is None:
        effective = (0, 0, width, height)
        cropped = bgr
    else:
        x, y, w, h = (int(v) for v in roi)
        if w <= 0 or h <= 0:
            raise ValueError("ROI width and height must be positive.")
        x = min(max(x, 0), width - 1)
        y = min(max(y, 0), height - 1)
        w = min(w, width - x)
        h = min(h, height - y)
        effective = (x, y, w, h)
        cropped = bgr[y : y + h, x : x + w]

    angle = float(rotation_angle) % 360.0
    if angle == 0.0:
        return cropped, effective
    if angle == 90.0:
        return cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE), effective
    if angle == 180.0:
        return cv2.rotate(cropped, cv2.ROTATE_180), effective
    if angle == 270.0:
        return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE), effective
    ch, cw = cropped.shape[:2]
    centre = (cw / 2.0, ch / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(cropped, matrix, (cw, ch), flags=cv2.INTER_LINEAR)
    return rotated, effective


# ---------------------------------------------------------------------------
# 3. Colour spaces, filtering, enhancement, histograms
# ---------------------------------------------------------------------------
def convert_colour_spaces(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(grayscale, HSV)`` for a BGR image (converted exactly once)."""
    image = validate_image(bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return gray, hsv


def _odd_kernel(kernel: int) -> int:
    kernel = max(3, int(kernel))
    return kernel if kernel % 2 == 1 else kernel + 1


def compare_noise_filters(gray: np.ndarray, kernel: int = 5) -> dict[str, Any]:
    """Compare Gaussian vs median filtering on a grayscale image.

    Returns both filtered images plus simple comparison measurements:
    ``residual_noise`` (std of input-minus-filtered; higher means more was
    removed) and ``laplacian_sharpness`` (variance of the Laplacian; higher
    means edges were preserved more strongly). Median filtering suits
    isolated speck (salt-and-pepper) noise; Gaussian filtering suits
    approximately Gaussian sensor noise but softens edges more.
    """
    if gray.ndim != 2:
        raise ValueError("compare_noise_filters expects a single-channel image.")
    k = _odd_kernel(kernel)
    gaussian = cv2.GaussianBlur(gray, (k, k), 0)
    median = cv2.medianBlur(gray, k)

    def metrics(filtered: np.ndarray) -> dict[str, float]:
        residual = gray.astype(np.float32) - filtered.astype(np.float32)
        lap = cv2.Laplacian(filtered, cv2.CV_64F)
        return {
            "residual_noise": float(np.std(residual)),
            "laplacian_sharpness": float(np.var(lap)),
        }

    return {
        "gaussian": gaussian,
        "median": median,
        "kernel_size": k,
        "gaussian_metrics": metrics(gaussian),
        "median_metrics": metrics(median),
    }


def apply_selected_filter(gray: np.ndarray, method: str, kernel: int = 5) -> np.ndarray:
    """Apply the user-selected filter (``gaussian``/``median``/``none``)."""
    k = _odd_kernel(kernel)
    if method == "gaussian":
        return cv2.GaussianBlur(gray, (k, k), 0)
    if method == "median":
        return cv2.medianBlur(gray, k)
    if method in ("none", None, ""):
        return gray
    raise ValueError(f"Unknown filter method: {method!r}")


def correct_illumination(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Normalise uneven illumination by smooth-background division.

    A large Gaussian blur estimates the slowly varying background; dividing
    by it flattens vignetting/uneven staining illumination.
    """
    if gray.ndim != 2:
        raise ValueError("correct_illumination expects a single-channel image.")
    k = _odd_kernel(max(15, kernel_size))
    # Estimate the background with a large morphological closing first so
    # dark cell-sized objects are filled with background brightness before
    # smoothing; a plain blur would sit inside large objects and turn them
    # into rings after division.
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, closing_kernel)
    background = cv2.GaussianBlur(closed, (k, k), 0)
    background = np.where(background == 0, 1, background).astype(np.uint8)
    corrected = cv2.divide(gray, background, scale=255.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def enhance_contrast(
    gray: np.ndarray,
    method: str = "clahe",
    gamma: float = 1.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    """Enhance contrast via histogram equalisation or CLAHE, plus gamma.

    ``method`` is one of ``none``/``hist_eq``/``clahe``. Gamma correction
    (Lab 03) is applied afterwards when ``gamma != 1.0`` using a LUT.
    """
    if gray.ndim != 2:
        raise ValueError("enhance_contrast expects a single-channel image.")
    if method == "clahe":
        tile = max(2, int(clahe_tile_grid))
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit), tileGridSize=(tile, tile)
        )
        enhanced = clahe.apply(gray)
    elif method == "hist_eq":
        enhanced = cv2.equalizeHist(gray)
    elif method in ("none", None, ""):
        enhanced = gray.copy()
    else:
        raise ValueError(f"Unknown enhancement method: {method!r}")

    if gamma is None:
        gamma = 1.0
    if float(gamma) <= 0:
        raise ValueError("Gamma must be positive.")
    if abs(float(gamma) - 1.0) > 1e-9:
        inv = 1.0 / float(gamma)
        lut = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8
        )
        enhanced = cv2.LUT(enhanced, lut)
    return enhanced


def calculate_histograms(
    gray_before: np.ndarray, gray_after: Optional[np.ndarray] = None
) -> dict[str, np.ndarray]:
    """Return 256-bin grayscale histograms for before/after comparison."""
    def hist(img: np.ndarray) -> np.ndarray:
        return cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    result = {"before": hist(gray_before)}
    if gray_after is not None:
        result["after"] = hist(gray_after)
    return result


# ---------------------------------------------------------------------------
# 4. Segmentation + morphological cleanup
# ---------------------------------------------------------------------------
def segment_rbc_like_cells(
    enhanced_gray: np.ndarray,
    hsv: Optional[np.ndarray] = None,
    method: str = "otsu",
    invert_foreground: bool = True,
    manual_threshold: int = 127,
    adaptive_block_size: int = 51,
    adaptive_c: float = 5.0,
    hsv_lower: Sequence[int] = (0, 40, 0),
    hsv_upper: Sequence[int] = (179, 255, 255),
) -> np.ndarray:
    """Segment RBC-like objects into a uint8 0/255 mask.

    Methods: ``otsu`` (Otsu grayscale thresholding), ``hsv`` (stain /
    saturation range in HSV -- requires *hsv*), ``adaptive`` (Gaussian
    adaptive threshold) and ``manual`` (fixed grayscale threshold). All
    thresholds are image-processing parameters, not clinical values.
    """
    if enhanced_gray.ndim != 2:
        raise ValueError("segment_rbc_like_cells expects a single-channel image.")
    if method == "otsu":
        flag = cv2.THRESH_BINARY_INV if invert_foreground else cv2.THRESH_BINARY
        _, mask = cv2.threshold(
            enhanced_gray, 0, 255, flag + cv2.THRESH_OTSU
        )
    elif method == "manual":
        flag = cv2.THRESH_BINARY_INV if invert_foreground else cv2.THRESH_BINARY
        _, mask = cv2.threshold(
            enhanced_gray, int(np.clip(manual_threshold, 0, 255)), 255, flag
        )
    elif method == "adaptive":
        flag = cv2.THRESH_BINARY_INV if invert_foreground else cv2.THRESH_BINARY
        mask = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag,
            _odd_kernel(adaptive_block_size), float(adaptive_c),
        )
    elif method == "hsv":
        if hsv is None:
            raise ValueError("HSV segmentation requires an HSV image.")
        lower = np.array(hsv_lower, dtype=np.uint8)
        upper = np.array(hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    else:
        raise ValueError(f"Unknown segmentation method: {method!r}")
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    # Guard: an (almost) fully-foreground mask means polarity is likely wrong.
    if method in ("otsu", "manual", "adaptive"):
        if float(np.mean(mask == 255)) > 0.90:
            mask = cv2.bitwise_not(mask)
    return mask


def clean_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    min_component_area: int = 30,
) -> np.ndarray:
    """Morphological cleanup: opening then closing, small-component removal.

    Uses elliptical structuring elements (Lab 09). Removes very small
    foreground components as likely noise. Always returns uint8 0/255.
    """
    if mask is None or mask.size == 0:
        raise ValueError("clean_binary_mask received an empty mask.")
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    k = _odd_kernel(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    it = max(1, int(iterations))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=it)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=it)

    if min_component_area and min_component_area > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        keep = np.zeros_like(cleaned)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                keep[labels == label] = 255
        cleaned = keep
    return cleaned


# ---------------------------------------------------------------------------
# 5. Edge information (Sobel boundary strength)
# ---------------------------------------------------------------------------
def compute_sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Return the Sobel gradient magnitude (float32, >= 0).

    Used meaningfully as a per-object boundary-strength feature; it is not
    merely displayed.
    """
    if gray.ndim != 2:
        raise ValueError("compute_sobel_edges expects a single-channel image.")
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(sobel_x, sobel_y)


def compute_laplacian(gray: np.ndarray) -> np.ndarray:
    """Laplacian response, provided only as a sharpness comparison metric."""
    if gray.ndim != 2:
        raise ValueError("compute_laplacian expects a single-channel image.")
    return cv2.Laplacian(gray, cv2.CV_32F)


# ---------------------------------------------------------------------------
# 6. Separation of touching cells (distance transform + watershed)
# ---------------------------------------------------------------------------
def separate_touching_cells_watershed(
    mask: np.ndarray,
    bgr_image: Optional[np.ndarray] = None,
    use_watershed: bool = True,
    peak_kernel: int = 15,
    fg_ratio: float = 0.35,
) -> dict[str, Any]:
    """Separate touching objects with distance-transform watershed.

    Returns a dict with ``labels`` (int32, 0 = background/excluded,
    >0 = object id), ``contours`` (one largest contour per object),
    ``watershed_image`` (labelled-region visualisation with boundaries),
    and ``num_objects``. Empty masks and single-object images are handled
    without crashing. Uses only OpenCV (no scipy/scikit-image).
    """
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = binary.shape[:2]
    labels = np.zeros((height, width), dtype=np.int32)
    vis_source = (
        validate_image(bgr_image) if bgr_image is not None
        else cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    )
    empty = {
        "labels": labels,
        "contours": [],
        "watershed_image": vis_source.copy(),
        "num_objects": 0,
    }
    if not np.any(binary):
        return empty

    if not use_watershed:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for idx, cnt in enumerate(contours, start=1):
            cv2.drawContours(labels, [cnt], -1, idx, thickness=cv2.FILLED)
        return {
            "labels": labels,
            "contours": list(contours),
            "watershed_image": _watershed_visualisation(vis_source, labels),
            "num_objects": len(contours),
        }

    # Foreground markers from distance-transform local maxima.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)  # smooth plateau noise in peaks
    dist_max = float(dist.max())
    if dist_max <= 0:
        return empty
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_kernel(peak_kernel),) * 2
    )
    local_max = cv2.dilate(dist, kernel)
    peaks = np.logical_and(dist == local_max, dist > fg_ratio * dist_max)
    sure_fg = np.where(peaks, 255, 0).astype(np.uint8)
    # Peaks of one round object can fragment; merge near-duplicate peaks.
    sure_fg = cv2.morphologyEx(
        sure_fg, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    # Give each marker a small area: a single-pixel marker on the image
    # border is otherwise absorbed by cv2.watershed and its object is lost.
    sure_fg = cv2.dilate(
        sure_fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    marker_count, markers = cv2.connectedComponents(sure_fg)
    if marker_count <= 1:  # no reliable peak: fall back to external contours
        return separate_touching_cells_watershed(
            binary, bgr_image, use_watershed=False
        )

    sure_bg = cv2.dilate(
        binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3
    )
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers = markers + 1  # background becomes 1, objects start at 2
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(vis_source.copy(), markers)

    # Relabel to 1..N and extract one contour per surviving region.
    contours: list[np.ndarray] = []
    for new_id, old_label in enumerate(
        [v for v in np.unique(markers_ws) if v > 1], start=1
    ):
        region = np.where(markers_ws == old_label, 255, 0).astype(np.uint8)
        region = cv2.bitwise_and(region, binary)
        labels = np.where(region == 255, new_id, labels)
        found, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if found:
            contours.append(max(found, key=cv2.contourArea))
    # Re-sort contours by centroid position for deterministic IDs.
    order = sorted(
        range(len(contours)),
        key=lambda i: (
            cv2.boundingRect(contours[i])[1], cv2.boundingRect(contours[i])[0]
        ),
    )
    contours = [contours[i] for i in order]
    # Rebuild the label image from the final contours so IDs always match.
    final_labels = np.zeros_like(labels)
    final_contours: list[np.ndarray] = []
    for new_id, cnt in enumerate(contours, start=1):
        cv2.drawContours(final_labels, [cnt], -1, new_id, thickness=cv2.FILLED)
        final_contours.append(cnt)
    return {
        "labels": final_labels,
        "contours": final_contours,
        "watershed_image": _watershed_visualisation(
            vis_source, final_labels, markers_ws
        ),
        "num_objects": len(final_contours),
    }


def _watershed_visualisation(
    bgr: np.ndarray,
    labels: np.ndarray,
    markers_ws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Colour labelled regions and draw watershed boundaries (-1 pixels)."""
    vis = bgr.copy()
    rng = np.random.default_rng(42)  # deterministic colours
    for label in [v for v in np.unique(labels) if v > 0]:
        colour = tuple(int(c) for c in rng.integers(60, 255, size=3))
        region = labels == label
        overlay = vis.copy()
        overlay[region] = colour
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        ys, xs = np.where(region)
        if len(xs):
            cv2.putText(
                vis, str(int(label)), (int(xs.mean()), int(ys.mean())),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
            )
    if markers_ws is not None:
        vis[markers_ws == -1] = (0, 0, 255)
    return vis


# ---------------------------------------------------------------------------
# 7. Contour / feature extraction
# ---------------------------------------------------------------------------
def extract_cell_features(
    contours: Sequence[np.ndarray],
    gray: np.ndarray,
    hsv: np.ndarray,
    sobel_magnitude: np.ndarray,
    min_area: float = 50,
    border_margin: int = 2,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Extract per-object features for accepted RBC-like objects.

    Returns ``(records, accepted_contours)`` in matching order. Records are
    plain dicts (CSV/DataFrame ready) containing object_id, pixel area,
    perimeter, circularity ``4*pi*A/P^2``, bounding box, centroid, aspect
    ratio, equivalent diameter, mean gray/HSV, Sobel boundary strength and
    border-touching status. Zero/tiny perimeters never divide by zero.
    """
    height, width = gray.shape[:2]
    records: list[dict[str, Any]] = []
    accepted: list[np.ndarray] = []
    for cnt in contours:
        if cnt is None or len(cnt) == 0:
            continue
        area = float(cv2.contourArea(cnt))
        if area <= 0 or area < float(min_area):
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        circularity = (
            float(4.0 * np.pi * area / (perimeter * perimeter))
            if perimeter > 0 else 0.0
        )
        x, y, w, h = (int(v) for v in cv2.boundingRect(cnt))
        moments = cv2.moments(cnt)
        if moments["m00"] > 0:
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0
        obj_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_gray = float(cv2.mean(gray, mask=obj_mask)[0])
        mean_h, mean_s, mean_v = (float(v) for v in cv2.mean(hsv, mask=obj_mask)[:3])
        # Boundary strength: mean Sobel magnitude on the contour pixels.
        boundary_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(boundary_mask, [cnt], -1, 255, thickness=1)
        boundary_values = sobel_magnitude[boundary_mask == 255]
        boundary_strength = float(np.mean(boundary_values)) if boundary_values.size else 0.0
        touches = bool(
            x <= border_margin or y <= border_margin
            or x + w >= width - border_margin or y + h >= height - border_margin
        )
        accepted.append(cnt)
        records.append({
            "object_id": len(records) + 1,
            "area_px": round(area, 2),
            "perimeter_px": round(perimeter, 2),
            "circularity": round(min(max(circularity, 0.0), 1.0), 4),
            "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
            "centroid_x": round(float(cx), 2), "centroid_y": round(float(cy), 2),
            "aspect_ratio": round(float(w) / float(h), 4) if h > 0 else 0.0,
            "equivalent_diameter_px": round(float(np.sqrt(4.0 * area / np.pi)), 2),
            "mean_gray": round(mean_gray, 2),
            "mean_hue": round(mean_h, 2),
            "mean_saturation": round(mean_s, 2),
            "mean_value": round(mean_v, 2),
            "sobel_boundary_strength": round(boundary_strength, 2),
            "touches_border": touches,
            "review_flags": "",
            "flagged": False,
        })
    return records, accepted


# ---------------------------------------------------------------------------
# 8. Non-diagnostic morphology screening (median / MAD robust statistics)
# ---------------------------------------------------------------------------
def _robust_z(values: np.ndarray) -> np.ndarray:
    """Median/MAD robust z-scores with safe zero-MAD fallbacks.

    If MAD is zero (many identical values), fall back to the IQR scale; if
    that is also zero, any value differing from the median is scored as a
    clear outlier (+/-10) instead of dividing by zero.
    """
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0 and np.isfinite(mad):
        return 0.6745 * (values - median) / mad
    q75, q25 = np.percentile(values, [75, 25])
    iqr_scale = (q75 - q25) / 1.349
    if iqr_scale > 0:
        return (values - median) / iqr_scale
    return np.where(values == median, 0.0, np.sign(values - median) * 10.0)


def flag_morphology_outliers(
    records: list[dict[str, Any]],
    min_reference_objects: int = 5,
    robust_z_threshold: float = 3.5,
    exclude_border_objects: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add image-relative, non-diagnostic review flags to *records* (in place).

    With fewer than ``min_reference_objects`` valid objects, batch-relative
    statistics are not presented as reliable: the summary reports
    "Insufficient reference objects" and only border exclusion is applied.
    Hue is compared on the circular 0-179 OpenCV scale via the shortest arc.
    """
    summary: dict[str, Any] = {
        "reference_status": "ok",
        "reference_object_count": 0,
        "flagged_count": 0,
        "border_excluded_count": 0,
    }
    valid = [r for r in records if not (exclude_border_objects and r["touches_border"])]
    for r in records:
        flags: list[str] = []
        if exclude_border_objects and r["touches_border"]:
            flags.append(FLAG_BORDER_EXCLUDED)
            summary["border_excluded_count"] += 1
        r["_flags"] = flags

    summary["reference_object_count"] = len(valid)
    if len(valid) < max(2, int(min_reference_objects)):
        summary["reference_status"] = INSUFFICIENT_REFERENCE
    else:
        z = float(robust_z_threshold)
        area = np.array([r["area_px"] for r in valid], dtype=float)
        circ = np.array([r["circularity"] for r in valid], dtype=float)
        gray_v = np.array([r["mean_gray"] for r in valid], dtype=float)
        bound = np.array([r["sobel_boundary_strength"] for r in valid], dtype=float)
        hue = np.array([r["mean_hue"] for r in valid], dtype=float)
        # Circular hue deviation from the circular median direction.
        hue_median = float(np.median(hue))
        hue_dev = np.minimum(
            np.abs(hue - hue_median), 180.0 - np.abs(hue - hue_median)
        )
        med_area = float(np.median(area)) if area.size else 0.0
        z_area, z_circ = _robust_z(area), _robust_z(circ)
        z_gray, z_bound = _robust_z(gray_v), _robust_z(bound)
        z_hue = _robust_z(hue_dev)  # hue scored on circular deviations
        for i, r in enumerate(valid):
            flags = r["_flags"]
            if med_area > 0 and z_area[i] > z and area[i] > 1.5 * med_area:
                flags.append(FLAG_UNRESOLVED_CLUSTER)
            elif med_area > 0 and z_area[i] < -z and area[i] < 0.5 * med_area:
                flags.append(FLAG_POSSIBLE_DEBRIS)
            if z_circ[i] < -z:
                flags.append(FLAG_IRREGULAR_MORPHOLOGY)
            if abs(z_hue[i]) > z or abs(z_gray[i]) > z:
                flags.append(FLAG_STAINING_OUTLIER)
            if z_bound[i] < -z:
                flags.append(FLAG_LOW_BOUNDARY)

    for r in records:
        flags = r.pop("_flags", [])
        r["review_flags"] = "; ".join(flags)
        r["flagged"] = any(f != FLAG_BORDER_EXCLUDED for f in flags)
    summary["flagged_count"] = int(sum(1 for r in records if r["flagged"]))
    return records, summary


# ---------------------------------------------------------------------------
# 9. Annotation + full pipeline
# ---------------------------------------------------------------------------
def annotate_results(
    bgr: np.ndarray,
    records: list[dict[str, Any]],
    contours: Sequence[np.ndarray],
) -> np.ndarray:
    """Draw contours, boxes and IDs; colours are review-status, not clinical."""
    annotated = validate_image(bgr).copy()
    for record, cnt in zip(records, contours):
        if record.get("touches_border"):
            colour = (128, 128, 128)  # grey: border object excluded
        elif record.get("flagged"):
            colour = (0, 165, 255)  # orange: morphology review flag
        else:
            colour = (0, 200, 0)  # green: counted RBC-like object
        cv2.drawContours(annotated, [cnt], -1, colour, 2)
        x, y = record["bbox_x"], record["bbox_y"]
        cv2.rectangle(
            annotated, (x, y),
            (x + record["bbox_w"], y + record["bbox_h"]), colour, 1,
        )
        cv2.putText(
            annotated, f"#{record['object_id']}", (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
        )
    total = len(records)
    excluded = sum(1 for r in records if r.get("touches_border"))
    flagged = sum(1 for r in records if r.get("flagged"))
    banner = (
        f"RBC-like objects: {total - excluded} | Review flags: {flagged} "
        f"| Border excluded: {excluded}"
    )
    cv2.putText(
        annotated, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 2,
    )
    return annotated


def process_smear_image(
    image: np.ndarray,
    config: Optional[PipelineConfig] = None,
    filename: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
) -> ProcessingResult:
    """Run the full CellScope EDU pipeline on one BGR image.

    The selected filter and enhancement outputs are the actual inputs to
    segmentation (no disconnected demonstration steps).
    """
    cfg = config or PipelineConfig()
    start = time.perf_counter()
    warnings: list[str] = []

    bgr = validate_image(image)
    processed, effective_roi = apply_roi_and_rotation(
        bgr, roi=cfg.roi, rotation_angle=cfg.rotation_angle
    )
    metadata = extract_metadata(
        processed, filename=filename, file_size_bytes=file_size_bytes,
        roi=effective_roi,
    )
    metadata["original_width"] = int(bgr.shape[1])
    metadata["original_height"] = int(bgr.shape[0])

    gray, hsv = convert_colour_spaces(processed)
    filter_comparison = compare_noise_filters(gray, kernel=cfg.filter_kernel)
    filtered = apply_selected_filter(gray, cfg.filter_method, cfg.filter_kernel)

    enhanced_input = (
        correct_illumination(filtered, cfg.illumination_kernel)
        if cfg.illumination_correction else filtered
    )
    enhanced = enhance_contrast(
        enhanced_input, method=cfg.enhancement_method, gamma=cfg.gamma,
        clahe_clip_limit=cfg.clahe_clip_limit, clahe_tile_grid=cfg.clahe_tile_grid,
    )
    histograms = calculate_histograms(gray, enhanced)

    raw_mask = segment_rbc_like_cells(
        enhanced, hsv=hsv, method=cfg.segmentation_method,
        invert_foreground=cfg.invert_foreground,
        manual_threshold=cfg.manual_threshold,
        adaptive_block_size=cfg.adaptive_block_size, adaptive_c=cfg.adaptive_c,
        hsv_lower=cfg.hsv_lower, hsv_upper=cfg.hsv_upper,
    )
    mask = clean_binary_mask(
        raw_mask, kernel_size=cfg.morph_kernel,
        iterations=cfg.morph_iterations,
        min_component_area=cfg.min_component_area,
    )
    sobel = compute_sobel_edges(enhanced)

    separation = separate_touching_cells_watershed(
        mask, processed, use_watershed=cfg.use_watershed,
        peak_kernel=cfg.watershed_peak_kernel, fg_ratio=cfg.watershed_fg_ratio,
    )
    if not np.any(mask):
        warnings.append("Segmentation produced an empty mask; no RBC-like objects detected.")
    elif separation["num_objects"] == 0:
        warnings.append("Foreground was present but no object met the separation criteria.")

    records, accepted = extract_cell_features(
        separation["contours"], enhanced, hsv, sobel,
        min_area=cfg.min_object_area, border_margin=cfg.border_margin,
    )
    if len(accepted) != separation["num_objects"]:
        warnings.append(
            f"{separation['num_objects'] - len(accepted)} separated region(s) "
            "were below the minimum object area and were not counted."
        )
    records, flag_summary = flag_morphology_outliers(
        records, min_reference_objects=cfg.min_reference_objects,
        robust_z_threshold=cfg.robust_z_threshold,
        exclude_border_objects=cfg.exclude_border_objects,
    )
    if flag_summary["reference_status"] == INSUFFICIENT_REFERENCE and records:
        warnings.append(
            "Insufficient reference objects: field-relative morphology "
            "statistics are not reliable for this image."
        )

    annotated = annotate_results(processed, records, accepted)
    detected = len(records) - flag_summary["border_excluded_count"]
    summary = {
        "detected_rbc_like_count": int(detected),
        "total_objects_found": int(len(records)),
        "flagged_count": int(flag_summary["flagged_count"]),
        "border_excluded_count": int(flag_summary["border_excluded_count"]),
        "reference_status": flag_summary["reference_status"],
        "segmentation_method": cfg.segmentation_method,
        "filter_method": cfg.filter_method,
        "enhancement_method": cfg.enhancement_method,
        "watershed_enabled": bool(cfg.use_watershed),
        "note": (
            "Educational image-based screening only; counts are RBC-like "
            "objects per processed image field, not a clinical measurement."
        ),
    }
    return ProcessingResult(
        processed_image=processed, gray=gray, hsv=hsv,
        filtered_gray=filtered, enhanced_gray=enhanced, mask=mask,
        sobel_magnitude=sobel,
        watershed_image=separation["watershed_image"],
        watershed_labels=separation["labels"],
        annotated_image=annotated, histograms=histograms,
        filter_comparison=filter_comparison, records=records,
        contours=accepted, summary=summary, metadata=metadata,
        runtime_seconds=float(time.perf_counter() - start), warnings=warnings,
    )
