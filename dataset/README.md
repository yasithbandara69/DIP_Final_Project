# CellScope EDU — Dataset Documentation

> Educational prototype only. Images here are used for image-processing
> development and evaluation. They are **not** clinical data and must not be
> presented as patient samples.

## Contents (as of Phase 3 — verify counts before submission)

| Location | Contents | Source |
|---|---|---|
| `dataset/raw/test/` | 27 × `BloodImage_*.jpg` (640×480) microscopy images | BCCD dataset, downloaded from https://github.com/Shenggan/BCCD_Dataset |
| `dataset/annotations/` | 27 × Pascal VOC XML files matching the images above (labels: RBC, WBC, Platelets) | Same BCCD repository |
| `dataset/raw/` (root) | 3 × `Gemini_Generated_Image_*.jpg` | Pre-existing repository files; AI-generated placeholder images from the original pill project — **not** blood-smear data and **not** part of the evaluation set |

## Known ground-truth limitation (must be stated in the report)

Visual inspection of `BloodImage_00012.jpg` against its BCCD XML (see
`outputs/validation/gt_overlay_00012.png`) shows the BCCD annotations are
**incomplete**: several clearly visible RBCs have no bounding box. Therefore:

- Count-error metrics against BCCD counts will overstate pipeline error.
- Bounding-box precision/recall/F1 must be interpreted with this caveat.
- Never claim these annotations are exhaustive manual counts.

## Assignment requirements still outstanding (do NOT fabricate)

The SE3071 brief requires ≥30 relevant images, of which ≥10 captured or
produced by the student, with source/resolution/format/purpose recorded.
Currently:

- Public (BCCD) images with annotations: 27 — usable for evaluation.
- Student-created images: **0 recorded so far.** These must be genuinely
  captured/produced by the student and added honestly; do not duplicate,
  rename, or synthesise images to reach a count.

## Manifest

Use `dataset/manifest_template.csv` with columns:
`filename, source, source_url, student_created, capture_device, magnification, resolution, format, purpose, split, ground_truth_count, annotation_file, notes`

Validate a filled manifest with `dataset_utils.validate_dataset_manifest()`
(reports totals, student-created count, calibration/training vs unseen-test
splits, and missing files/annotations). Blank/unknown fields are reported
as unknown — never guessed.
