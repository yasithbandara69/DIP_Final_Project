"""CellScope EDU -- Streamlit interface (Phase 2).

Educational prototype for automated blood-smear RBC-like cell counting and
morphology screening. This interface is a thin layer over ``pipeline.py``;
all image processing lives there. Evaluation calculations arrive in Phase 3.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import dataset_utils as ds
import evaluation as ev
import pipeline as pl

DISCLAIMER = (
    "This educational prototype performs image-based cell counting and "
    "morphology screening only. It is not validated for clinical diagnosis "
    "or patient care."
)

st.set_page_config(page_title="CellScope EDU", page_icon="🔬", layout="wide")

st.title("🔬 CellScope EDU")
st.caption(
    "Automated Blood-Smear RBC-Like Cell Counting and Morphology Screening Tool "
    "— educational Digital Image Processing prototype"
)
st.warning(f"⚠️ {DISCLAIMER}")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for _key, _default in {
    "uploader_key": 0,
    "single_result": None,
    "single_filename": None,
    "batch_records": None,
    "batch_errors": [],
    "eval_frame": None,
    "eval_summary": None,
    "eval_notes": [],
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


def reset_app() -> None:
    """Clear stored results and reset uploaders via a fresh uploader key."""
    st.session_state["single_result"] = None
    st.session_state["single_filename"] = None
    st.session_state["batch_records"] = None
    st.session_state["batch_errors"] = []
    st.session_state["eval_frame"] = None
    st.session_state["eval_summary"] = None
    st.session_state["eval_notes"] = []
    st.session_state["uploader_key"] += 1
    st.rerun()


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Processing Controls")

    if st.button("🔄 Reset / Clear All", use_container_width=True):
        reset_app()

    with st.expander("✂️ ROI & Rotation", expanded=True):
        use_roi = st.checkbox("Enable ROI crop", value=False)
        roi_x = st.slider("ROI start X (%)", 0, 95, 0, 1)
        roi_y = st.slider("ROI start Y (%)", 0, 95, 0, 1)
        roi_w = st.slider("ROI width (%)", 5, 100, 100, 1)
        roi_h = st.slider("ROI height (%)", 5, 100, 100, 1)
        rotation = st.selectbox("Rotation (degrees)", [0, 90, 180, 270], index=0)

    with st.expander("🧹 Preprocessing", expanded=True):
        filter_method = st.selectbox(
            "Noise filter", ["median", "gaussian", "none"], index=0,
            help="Median suits isolated speck noise; Gaussian suits "
                 "approximately Gaussian image noise.",
        )
        filter_kernel = st.select_slider("Filter kernel size", [3, 5, 7], value=5)
        illumination = st.checkbox("Illumination correction", value=True)
        enhancement = st.selectbox(
            "Contrast enhancement", ["clahe", "hist_eq", "none"], index=0
        )
        gamma = st.slider("Gamma correction", 0.4, 2.5, 1.0, 0.05)

    with st.expander("🎯 Segmentation", expanded=True):
        seg_method = st.selectbox(
            "Segmentation method", ["otsu", "hsv", "adaptive", "manual"], index=0
        )
        invert_fg = st.checkbox("Cells darker than background (invert)", value=True)
        manual_thresh = st.slider("Manual threshold", 0, 255, 127, 1)
        hsv_s_min = st.slider("HSV min saturation", 0, 255, 40, 1)
        hsv_v_min = st.slider("HSV min value", 0, 255, 0, 1)
        morph_kernel = st.select_slider("Morphology kernel", [3, 5, 7], value=3)
        morph_iter = st.slider("Morphology iterations", 1, 3, 1, 1)
        min_component = st.slider("Min component area (px)", 1, 500, 30, 1)
        min_object = st.slider("Min object area (px)", 10, 2000, 50, 10)

    with st.expander("💧 Watershed Separation", expanded=False):
        use_watershed = st.checkbox("Enable watershed", value=True)
        peak_kernel = st.slider("Peak separation kernel", 5, 41, 15, 2)

    exclude_borders = st.checkbox("Exclude border-touching objects", value=True)


def build_config(image_shape=None) -> pl.PipelineConfig:
    """Build a PipelineConfig from the sidebar, resolving % ROI to pixels."""
    roi = None
    if use_roi and image_shape is not None:
        h, w = image_shape[:2]
        x = int(w * roi_x / 100)
        y = int(h * roi_y / 100)
        roi = (x, y, max(1, int(w * roi_w / 100)), max(1, int(h * roi_h / 100)))
    return pl.PipelineConfig(
        roi=roi,
        rotation_angle=float(rotation),
        filter_method=filter_method,
        filter_kernel=int(filter_kernel),
        illumination_correction=illumination,
        enhancement_method=enhancement,
        gamma=float(gamma),
        segmentation_method=seg_method,
        invert_foreground=invert_fg,
        manual_threshold=int(manual_thresh),
        hsv_lower=(0, int(hsv_s_min), int(hsv_v_min)),
        morph_kernel=int(morph_kernel),
        morph_iterations=int(morph_iter),
        min_component_area=int(min_component),
        min_object_area=int(min_object),
        use_watershed=use_watershed,
        watershed_peak_kernel=int(peak_kernel),
        exclude_border_objects=exclude_borders,
    )


tabs = st.tabs([
    "🩸 Single Image Analysis",
    "📁 Batch Processing",
    "📊 Evaluation",
    "📖 Method and Limitations",
])

# ===========================================================================
# TAB 1 -- SINGLE IMAGE ANALYSIS
# ===========================================================================
with tabs[0]:
    uploaded = st.file_uploader(
        "Upload a stained blood-smear image (PNG/JPG/JPEG)",
        type=["png", "jpg", "jpeg"],
        key=f"single_uploader_{st.session_state['uploader_key']}",
    )

    if uploaded is not None:
        try:
            image = pl.load_image_from_bytes(uploaded.getvalue(), filename=uploaded.name)
        except (ValueError, FileNotFoundError) as exc:
            st.error(f"❌ Could not load '{uploaded.name}': {exc}")
            image = None

        if image is not None:
            cfg = build_config(image.shape)
            if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
                with st.spinner("Processing image…"):
                    try:
                        result = pl.process_smear_image(
                            image, cfg, filename=uploaded.name,
                            file_size_bytes=uploaded.size,
                        )
                        st.session_state["single_result"] = result
                        st.session_state["single_filename"] = uploaded.name
                    except Exception as exc:  # user-friendly, never a traceback
                        st.error(f"❌ Processing failed: {exc}")
                        st.session_state["single_result"] = None

    result = st.session_state.get("single_result")
    if result is not None:
        st.success(
            f"Analysed **{st.session_state.get('single_filename')}** in "
            f"{result.runtime_seconds:.2f} s (educational screening only)."
        )
        for warning in result.warnings:
            st.warning(warning)

        s = result.summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RBC-like objects", s["detected_rbc_like_count"])
        c2.metric("Morphology review flags", s["flagged_count"])
        c3.metric("Border objects excluded", s["border_excluded_count"])
        c4.metric("Reference status", s["reference_status"])
        st.caption(s["note"])

        with st.expander("ℹ️ Image metadata & ROI information", expanded=True):
            meta_df = pd.DataFrame(
                [{"Property": k, "Value": str(v)} for k, v in result.metadata.items()]
            )
            st.dataframe(meta_df, use_container_width=True, hide_index=True)

        st.subheader("Image previews")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Processed input (after ROI/rotation)**")
            st.image(bgr_to_rgb(result.processed_image), use_container_width=True)
            st.markdown("**Enhanced grayscale**")
            st.image(result.enhanced_gray, use_container_width=True)
            st.markdown("**Cleaned segmentation mask**")
            st.image(result.mask, use_container_width=True)
        with p2:
            st.markdown("**Watershed labelled regions**")
            st.image(bgr_to_rgb(result.watershed_image), use_container_width=True)
            st.markdown("**Annotated result**")
            st.image(bgr_to_rgb(result.annotated_image), use_container_width=True)

        st.subheader("🔍 Gaussian vs median filter comparison")
        st.caption(
            "Median filtering is useful for isolated speck (salt-and-pepper) "
            "noise; Gaussian filtering reduces approximately Gaussian image "
            "noise but softens edges more. The filter chosen in the sidebar "
            "is the one actually used by segmentation."
        )
        fc = result.filter_comparison
        f1, f2 = st.columns(2)
        with f1:
            st.markdown("**Gaussian**")
            st.image(fc["gaussian"], use_container_width=True)
            st.json({k: round(v, 3) for k, v in fc["gaussian_metrics"].items()})
        with f2:
            st.markdown("**Median**")
            st.image(fc["median"], use_container_width=True)
            st.json({k: round(v, 3) for k, v in fc["median_metrics"].items()})

        st.subheader("📈 Grayscale histograms (before / after enhancement)")
        hist_df = pd.DataFrame({
            "Before enhancement": result.histograms["before"],
            "After enhancement": result.histograms.get(
                "after", result.histograms["before"]
            ),
        })
        st.line_chart(hist_df)

        st.subheader("📋 Per-object features (RBC-like objects)")
        if result.records:
            df = pd.DataFrame(result.records)
            st.dataframe(df, use_container_width=True, hide_index=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button(
                    "⬇️ Annotated image (PNG)", encode_png(result.annotated_image),
                    "annotated.png", "image/png", use_container_width=True,
                )
            with d2:
                st.download_button(
                    "⬇️ Segmentation mask (PNG)", encode_png(result.mask),
                    "mask.png", "image/png", use_container_width=True,
                )
            with d3:
                st.download_button(
                    "⬇️ Cell features CSV", df.to_csv(index=False).encode("utf-8"),
                    "cell_features.csv", "text/csv", use_container_width=True,
                )
        else:
            st.info("No RBC-like objects were detected with the current parameters.")
    elif uploaded is None:
        st.info("Upload an image above, adjust parameters in the sidebar, then run the analysis.")

# ===========================================================================
# TAB 2 -- BATCH PROCESSING
# ===========================================================================
with tabs[1]:
    st.subheader("Batch processing")
    st.caption(
        "Process multiple uploaded images or every supported image in a local "
        "directory. A corrupted file is recorded and skipped — the batch continues."
    )
    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"batch_uploader_{st.session_state['uploader_key']}",
    )
    batch_dir = st.text_input("…or a local directory path", value="dataset/raw")

    if st.button("▶️ Run Batch", type="primary", use_container_width=True):
        jobs: list[tuple[str, bytes | None, Path | None]] = []
        for f in batch_files or []:
            jobs.append((f.name, f.getvalue(), None))
        if batch_dir.strip():
            directory = Path(batch_dir.strip())
            if directory.is_dir():
                for path in sorted(directory.iterdir()):
                    if path.suffix.lower() in pl.SUPPORTED_EXTENSIONS:
                        jobs.append((path.name, None, path))
            elif not jobs:
                st.error(f"Directory not found: {batch_dir}")

        if not jobs:
            st.warning("No images to process. Upload files or give a valid directory.")
        else:
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            records, errors = [], []
            progress = st.progress(0.0)
            for i, (name, data, path) in enumerate(jobs):
                started = time.perf_counter()
                try:
                    img = (
                        pl.load_image_from_bytes(data, filename=name)
                        if data is not None else pl.load_image_from_path(path)
                    )
                    res = pl.process_smear_image(
                        img, build_config(img.shape), filename=name
                    )
                    stem = Path(name).stem
                    cv2.imwrite(str(output_dir / f"annotated_{stem}.png"),
                                res.annotated_image)
                    cv2.imwrite(str(output_dir / f"mask_{stem}.png"), res.mask)
                    records.append({
                        "filename": name,
                        "predicted_count": res.summary["detected_rbc_like_count"],
                        "flagged_count": res.summary["flagged_count"],
                        "excluded_border_count": res.summary["border_excluded_count"],
                        "runtime_seconds": round(time.perf_counter() - started, 3),
                        "status": "ok",
                        "error": "",
                    })
                except Exception as exc:
                    records.append({
                        "filename": name,
                        "predicted_count": 0,
                        "flagged_count": 0,
                        "excluded_border_count": 0,
                        "runtime_seconds": round(time.perf_counter() - started, 3),
                        "status": "error",
                        "error": str(exc),
                    })
                    errors.append(f"{name}: {exc}")
                progress.progress((i + 1) / len(jobs))
            st.session_state["batch_records"] = records
            st.session_state["batch_errors"] = errors

    if st.session_state.get("batch_records"):
        batch_df = pd.DataFrame(st.session_state["batch_records"])
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        ok_count = int((batch_df["status"] == "ok").sum())
        st.success(f"Batch finished: {ok_count}/{len(batch_df)} image(s) processed. "
                   "Annotated images and masks were saved under `outputs/`.")
        for err in st.session_state.get("batch_errors", []):
            st.error(f"❌ {err}")
        st.download_button(
            "⬇️ Batch report CSV", batch_df.to_csv(index=False).encode("utf-8"),
            "batch_report.csv", "text/csv",
        )

# ===========================================================================
# TAB 3 -- EVALUATION (Phase 3)
# ===========================================================================
with tabs[2]:
    st.subheader("Evaluation against ground truth")
    st.caption(
        "Ground truth is never invented. Provide a CSV (filename + RBC count) "
        "and/or Pascal VOC XML annotations (RBC labels). Without ground truth, "
        "no performance scores are shown."
    )
    st.markdown("**Project targets (goals only — not measured results):**")
    st.table(pd.DataFrame([
        {"Target": "Mean percentage count error", "Goal": "≤ 15%"},
        {"Target": "Detection F1", "Goal": "≥ 0.75"},
        {"Target": "Average runtime per image", "Goal": "≤ 2 s on the stated test computer"},
    ]))

    if use_roi or int(rotation) != 0:
        st.warning(
            "⚠️ ROI cropping or rotation is enabled in the sidebar. Annotation "
            "evaluation is disabled until ROI is turned off and rotation is 0°, "
            "so predicted boxes stay in original image coordinates."
        )

    eval_images = st.file_uploader(
        "Evaluation images (upload)",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True,
        key=f"eval_img_uploader_{st.session_state['uploader_key']}",
    )
    eval_dir = st.text_input("…or evaluation image directory", value="dataset/raw/test")
    gt_csv_file = st.file_uploader(
        "Ground-truth CSV (filename, ground_truth_count)",
        type=["csv"], key=f"eval_csv_{st.session_state['uploader_key']}",
    )
    gt_xml_files = st.file_uploader(
        "Pascal VOC XML annotations (optional, enables precision/recall/F1)",
        type=["xml"], accept_multiple_files=True,
        key=f"eval_xml_{st.session_state['uploader_key']}",
    )
    gt_mask_files = st.file_uploader(
        "Ground-truth binary masks (optional, enables IoU/Dice; match by filename stem)",
        type=["png", "jpg", "jpeg"], accept_multiple_files=True,
        key=f"eval_masks_{st.session_state['uploader_key']}",
    )
    iou_threshold = st.slider("BBox matching IoU threshold", 0.1, 0.9, 0.5, 0.05)
    st.caption(
        "Matching rule: greedy one-to-one matching by descending IoU — each "
        "predicted and each ground-truth box can be used at most once."
    )

    has_gt_input = gt_csv_file is not None or bool(gt_xml_files)
    if not has_gt_input:
        st.info(
            "📭 **No ground truth provided.** Upload a ground-truth CSV and/or "
            "VOC XML annotations above to run evaluation. No placeholder "
            "performance scores are displayed."
        )

    run_eval = st.button(
        "▶️ Run Evaluation", type="primary", use_container_width=True,
        disabled=(not has_gt_input) or use_roi or int(rotation) != 0,
    )
    if run_eval:
        notes: list[str] = []
        try:
            gt_counts: dict[str, int] = {}
            if gt_csv_file is not None:
                gt_counts = ev.load_ground_truth_csv(gt_csv_file.getvalue())
            annotations: dict[str, dict] = {}
            if gt_xml_files:
                annotations = ev.load_voc_annotations(
                    [f.getvalue() for f in gt_xml_files],
                    filenames=[f.name for f in gt_xml_files],
                )
                for stem, ann in annotations.items():
                    gt_counts.setdefault(
                        ann["filename"] or f"{stem}.jpg", len(ev.rbc_boxes(ann))
                    )
            gt_mask_map: dict[str, bytes] = {}
            for f in gt_mask_files or []:
                gt_mask_map[Path(f.name).stem] = f.getvalue()

            jobs: list[tuple[str, bytes | None, Path | None]] = [
                (f.name, f.getvalue(), None) for f in eval_images or []
            ]
            directory = Path(eval_dir.strip()) if eval_dir.strip() else None
            if directory is not None and directory.is_dir():
                for path in sorted(directory.iterdir()):
                    if path.suffix.lower() in pl.SUPPORTED_EXTENSIONS:
                        jobs.append((path.name, None, path))
            elif directory is not None and not jobs:
                st.error(f"Evaluation directory not found: {eval_dir}")

            if not jobs:
                st.warning("No evaluation images found.")
            else:
                per_image: list[dict] = []
                progress = st.progress(0.0)
                for i, (name, data, path) in enumerate(jobs):
                    # Ground truth may key the file by stem (from XML).
                    stem = Path(name).stem
                    ann = annotations.get(stem)
                    gt_count = gt_counts.get(name)
                    if gt_count is None and ann is not None:
                        gt_count = len(ev.rbc_boxes(ann))
                    if gt_count is None:
                        notes.append(f"{name}: no ground truth — skipped.")
                        progress.progress((i + 1) / len(jobs))
                        continue
                    try:
                        img = (
                            pl.load_image_from_bytes(data, filename=name)
                            if data is not None else pl.load_image_from_path(path)
                        )
                        started = time.perf_counter()
                        res = pl.process_smear_image(
                            img, build_config(img.shape), filename=name
                        )
                        runtime = time.perf_counter() - started
                        item: dict = {
                            "filename": name,
                            # Counted objects only (border-excluded objects are
                            # not counted, matching the analysis summary).
                            "predicted_count": res.summary["detected_rbc_like_count"],
                            "ground_truth_count": int(gt_count),
                            "runtime_seconds": round(runtime, 3),
                        }
                        if ann is not None:
                            gt_boxes = ev.rbc_boxes(ann)
                            if ann.get("width") and ann.get("height"):
                                gt_boxes = [
                                    ev.scale_bbox(
                                        b, (ann["width"], ann["height"]),
                                        (img.shape[1], img.shape[0]),
                                    )
                                    for b in gt_boxes
                                ]
                            pred_boxes = [
                                (float(r["bbox_x"]), float(r["bbox_y"]),
                                 float(r["bbox_x"] + r["bbox_w"]),
                                 float(r["bbox_y"] + r["bbox_h"]))
                                for r in res.records
                                if not (exclude_borders and r["touches_border"])
                            ]
                            matched = ev.match_boxes(pred_boxes, gt_boxes, iou_threshold)
                            p, r_, f1 = ev.precision_recall_f1(
                                matched["tp"], matched["fp"], matched["fn"]
                            )
                            item.update({
                                "tp": matched["tp"], "fp": matched["fp"],
                                "fn": matched["fn"], "precision": round(p, 4),
                                "recall": round(r_, 4), "f1": round(f1, 4),
                                "predicted_count": len(pred_boxes),
                                "ground_truth_count": len(gt_boxes),
                            })
                        if stem in gt_mask_map:
                            gt_mask = pl.load_image_from_bytes(
                                gt_mask_map[stem], filename=f"{stem}.png"
                            )
                            gt_gray = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
                            if gt_gray.shape != res.mask.shape:
                                gt_gray = cv2.resize(
                                    gt_gray, (res.mask.shape[1], res.mask.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            iou_v, dice_v = ev.mask_iou_dice(res.mask, gt_gray)
                            item.update({"iou": round(iou_v, 4), "dice": round(dice_v, 4)})
                        per_image.append(item)
                    except Exception as exc:
                        notes.append(f"{name}: evaluation error — {exc}")
                    progress.progress((i + 1) / len(jobs))

                if per_image:
                    frame, summary = ev.evaluate_counts(per_image)
                    st.session_state["eval_frame"] = frame
                    st.session_state["eval_summary"] = summary
                    st.session_state["eval_notes"] = notes
                else:
                    st.warning("No images could be matched to ground truth.")
                    st.session_state["eval_notes"] = notes
        except ValueError as exc:
            st.error(f"❌ {exc}")

    for note in st.session_state.get("eval_notes") or []:
        st.warning(note)
    if st.session_state.get("eval_frame") is not None:
        frame = st.session_state["eval_frame"]
        summary = st.session_state["eval_summary"]
        st.dataframe(frame, use_container_width=True, hide_index=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Images evaluated", summary["image_count"])
        m2.metric("MAE (cells)", f"{summary['mae']:.2f}" if summary["mae"] is not None else "—")
        m3.metric("MAPE (%)", f"{summary['mape']:.1f}" if summary["mape"] is not None else "—")
        m4.metric("Count accuracy (%)",
                  f"{summary['count_accuracy']:.1f}" if summary["count_accuracy"] is not None else "—")
        if "f1" in summary:
            st.metric("Detection F1 (all images)", f"{summary['f1']:.3f}")
            st.caption(
                f"Precision {summary['precision']:.3f} · Recall {summary['recall']:.3f} "
                f"· TP {summary['tp']} · FP {summary['fp']} · FN {summary['fn']}. "
                "Reminder: published BCCD-style annotations may not label every "
                "visible RBC-like object, which affects precision."
            )
        if summary.get("mape_excluded_zero_gt"):
            st.caption(
                f"{summary['mape_excluded_zero_gt']} image(s) had ground-truth "
                "count 0 and were excluded from MAPE (listed in the table)."
            )
        st.caption(
            "Measured results above are from the ground truth you supplied. "
            "Project targets (≤15% MAPE, F1 ≥ 0.75, ≤2 s/image) are goals, "
            "not claimed achievements."
        )
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️ Per-image evaluation CSV",
                frame.to_csv(index=False).encode("utf-8"),
                "evaluation_per_image.csv", "text/csv", use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Evaluation summary CSV",
                ev.summary_frame(summary).to_csv(index=False).encode("utf-8"),
                "evaluation_summary.csv", "text/csv", use_container_width=True,
            )

    with st.expander("🗂️ Dataset manifest validation"):
        st.caption(
            "Upload a filled manifest (see dataset/manifest_template.csv) to "
            "check totals, student-created count, split counts, and missing files."
        )
        manifest_file = st.file_uploader(
            "Dataset manifest CSV", type=["csv"],
            key=f"manifest_{st.session_state['uploader_key']}",
        )
        manifest_root = st.text_input("Dataset root for manifest paths", value="dataset")
        if manifest_file is not None and st.button("Validate manifest"):
            try:
                report = ds.validate_dataset_manifest(
                    manifest_file.getvalue(), manifest_root
                )
                st.json(report)
            except Exception as exc:
                st.error(f"❌ Manifest validation failed: {exc}")

# ===========================================================================
# TAB 4 -- METHOD AND LIMITATIONS
# ===========================================================================
with tabs[3]:
    st.subheader("Method and Limitations")
    st.markdown(f"> ⚠️ {DISCLAIMER}")
    st.markdown(
        """
### Processing workflow
1. **Acquisition & validation** — images are decoded in memory
   (`cv2.imdecode`) or loaded from a path; unsupported/corrupted files are
   rejected with a user-friendly error.
2. **Geometry** — optional ROI crop and rotation define the analysed field.
3. **Colour spaces** — grayscale drives intensity-based steps; HSV carries
   stain/saturation information for colour segmentation and per-object
   staining features.
4. **Enhancement** — illumination is normalised by dividing by a smooth
   estimated background (large morphological closing + Gaussian blur), then
   contrast is enhanced with histogram equalisation or CLAHE, with optional
   gamma correction. The enhanced image is what segmentation actually uses.
5. **Noise filtering** — Gaussian and median filters are compared
   (residual noise, Laplacian sharpness); the selected filter feeds
   segmentation. Median suits isolated speck noise; Gaussian suits
   approximately Gaussian noise.
6. **Segmentation** — Otsu, HSV-range, adaptive, or manual thresholding
   (adjustable image-processing parameters, never clinical cut-offs).
7. **Morphological cleanup** — elliptical opening/closing plus removal of
   very small components as likely noise.
8. **Separation** — Sobel gradient magnitude gives boundary strength; a
   distance transform finds object cores, connected components label them,
   and watershed splits touching RBC-like objects along weak boundaries.
9. **Features & screening** — area, perimeter, circularity
   `4π·area/perimeter²`, aspect ratio, equivalent diameter, centroid, mean
   intensity/HSV and Sobel boundary strength are extracted per object.
   Review flags use field-relative median/MAD statistics.

### Why the flags are not diagnoses
Flags such as *Irregular morphology review* or *Staining outlier review*
only say an object differs from other objects **in the same image field**.
They carry no clinical meaning, require human review, and below five valid
objects the app reports **Insufficient reference objects** instead of
presenting unreliable statistics.

### Limitations
- Stain type/concentration, illumination, magnification, focus, compression
  and overlap all change pixel statistics; parameters tuned on one dataset
  may fail on another.
- Heavily overlapping clusters may remain unresolved; debris can resemble
  small cell fragments at pixel level.
- Counts are **objects per processed image field** — without physical
  calibration and a laboratory procedure they must not be converted to
  cells per microlitre.
- This is an educational prototype. It must not be used for patient care.
        """
    )
