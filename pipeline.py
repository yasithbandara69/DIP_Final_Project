import os
import cv2
import numpy as np


# ==========================================
# PART B: CORE PROCESSING PIPELINE
# ==========================================


def load_and_validate_image(image_path, target_width=1000):
    """Part B: Safely load image, extract metadata, and resize."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unsupported or corrupted image file: {image_path}")

    orig_h, orig_w, channels = img.shape
    file_size_kb = os.path.getsize(image_path) / 1024.0

    scale = target_width / float(orig_w)
    target_height = int(orig_h * scale)
    resized_img = cv2.resize(
        img, (target_width, target_height), interpolation=cv2.INTER_AREA
    )

    metadata = {
        "Dimensions": f"{orig_w} x {orig_h} (Scaled to {target_width}x{target_height})",
        "Width": orig_w,
        "Height": orig_h,
        "Channels": channels,
        "Bit Depth": str(img.dtype),
        "File Size (KB)": round(file_size_kb, 2),
    }
    return resized_img, metadata


def convert_color_spaces(img):
    """Part B & Lab 02: Convert BGR to Grayscale and HSV."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return gray, hsv


def compare_noise_filters(img):
    """Part B & Lab 06: Compare Gaussian vs Median filtering."""
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    return gaussian, median


def enhance_illumination_and_contrast(gray_img):
    """Part B & Lab 05: Apply CLAHE to balance uneven lighting."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)


# ==========================================
# PART C: ROBUST INDUSTRIAL SEGMENTATION
# ==========================================


def segment_pills(
    img,
    bg_mode="Colored Pills on White Tray",
    white_threshold=235,
    min_saturation=10,
    dark_thresh=190,
):
    """Part C: Industrial Multi-Background Segmentation Engine."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(hsv)
    h_img, w_img = gray.shape[:2]

    if bg_mode == "Colored Pills on White Tray":
        _, dark_mask = cv2.threshold(
            gray, white_threshold, 255, cv2.THRESH_BINARY_INV
        )
        _, color_mask = cv2.threshold(
            s_channel, min_saturation, 255, cv2.THRESH_BINARY
        )
        pill_mask = cv2.bitwise_and(dark_mask, color_mask)

    elif bg_mode == "White Pills on White Tray (Edge Rim)":
        # 1. CLAHE enhancement to bring out faint rims on the lit side
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        blurred_gray = cv2.medianBlur(enhanced_gray, 7)

        # 2. Hough Circle Transform tuned for white circular tablets
        min_dist = int(w_img * 0.12)
        min_r = int(w_img * 0.05)
        max_r = int(w_img * 0.14)

        # Lowered param2 to 17 so even faint left-column tablets vote successfully
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=50,
            param2=17,
            minRadius=min_r,
            maxRadius=max_r,
        )

        pill_mask = np.zeros(gray.shape, dtype=np.uint8)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # Draw the detected circular tablet solid
                cv2.circle(pill_mask, (cx, cy), r, 255, -1)
        else:
            edges = cv2.Canny(blurred_gray, 15, 65)
            kernel_rim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            closed_rim = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, kernel_rim, iterations=2
            )
            cnts, _ = cv2.findContours(
                closed_rim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                if cv2.contourArea(c) > 300:
                    cv2.drawContours(pill_mask, [c], -1, 255, -1)

    else:
        # White Pills on Dark / Colored Trays
        _, bright_mask = cv2.threshold(
            gray, dark_thresh, 255, cv2.THRESH_BINARY
        )
        _, low_sat_mask = cv2.threshold(
            s_channel, 60, 255, cv2.THRESH_BINARY_INV
        )
        pill_mask = cv2.bitwise_or(bright_mask, low_sat_mask)

        blurred = cv2.medianBlur(gray, 5)
        _, otsu_bright = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        pill_mask = cv2.bitwise_and(pill_mask, otsu_bright)

    # Fill internal engravings / score lines solid
    contours, _ = cv2.findContours(
        pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    solid_mask = np.zeros(pill_mask.shape, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            cv2.drawContours(solid_mask, [cnt], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(solid_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


# Alias function so neither name ever throws an AttributeError:
def segment_pills_white_bg(img, white_threshold=235, min_saturation=10):
    return segment_pills(
        img,
        bg_mode="Colored Pills on White Tray",
        white_threshold=white_threshold,
        min_saturation=min_saturation,
    )


def separate_touching_pills_watershed(cleaned_mask, original_img, peak_dist=21):
    """Part C: Distance Transform Watershed for Large Batches."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(cleaned_mask, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
    dist_smooth = cv2.GaussianBlur(dist_transform, (7, 7), 0)

    p_size = max(5, int(peak_dist))
    if p_size % 2 == 0:
        p_size += 1
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p_size, p_size))

    local_max = cv2.dilate(dist_smooth, peak_kernel)
    peaks = (dist_smooth == local_max) & (dist_smooth > 0.20 * dist_smooth.max())
    sure_fg = np.uint8(peaks * 255)

    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents(sure_fg)

    if num_markers <= 1:
        cnts, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return cnts

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_img, markers)

    pill_contours = []
    unique_labels = np.unique(markers)

    for label in unique_labels:
        if label <= 1:
            continue

        single_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
        single_mask[markers == label] = 255
        single_mask = cv2.bitwise_and(single_mask, cleaned_mask)

        single_mask = cv2.morphologyEx(
            single_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )
        cnts, _ = cv2.findContours(
            single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            pill_contours.append(c)

    return pill_contours


def inspect_pills_and_detect_defects(
    cleaned_mask,
    original_img,
    min_area=400,
    circularity_threshold=0.26,
    area_ratio_threshold=0.52,
    use_watershed=True,
    peak_distance=21,
    ignore_borders=True,
):
    """Part C: Comprehensive Multi-Unit Tablet Inspection with Border Filtering."""
    h_img, w_img = original_img.shape[:2]

    if use_watershed:
        candidate_contours = separate_touching_pills_watershed(
            cleaned_mask, original_img, peak_dist=peak_distance
        )
    else:
        candidate_contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    annotated_img = original_img.copy()

    raw_pills = []
    for cnt in candidate_contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            raw_pills.append((cnt, area))

    if len(raw_pills) == 0:
        return annotated_img, 0, 0, [], "NO PILLS DETECTED"

    # Separate interior pills from boundary-touching pills
    interior_areas = []
    for cnt, area in raw_pills:
        x, y, w, h = cv2.boundingRect(cnt)
        is_border = (
            (x <= 4)
            or (y <= 4)
            or (x + w >= w_img - 4)
            or (y + h >= h_img - 4)
        )
        if not is_border:
            interior_areas.append(area)

    if len(interior_areas) > 0:
        median_batch_area = np.median(interior_areas)
    else:
        median_batch_area = np.median([p[1] for p in raw_pills])

    # Extract dominant batch hue
    hues = []
    for cnt, _ in raw_pills:
        pm = np.zeros(cleaned_mask.shape, dtype=np.uint8)
        cv2.drawContours(pm, [cnt], -1, 255, -1)
        m_hue = cv2.mean(hsv_img, mask=pm)[0]
        hues.append(m_hue)
    median_batch_hue = np.median(hues) if len(hues) > 0 else 0

    pill_reports = []
    defective_count = 0
    audited_count = 0

    for idx, (cnt, area) in enumerate(raw_pills, start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        is_border = (
            (x <= 4)
            or (y <= 4)
            or (x + w >= w_img - 4)
            or (y + h >= h_img - 4)
        )

        perimeter = cv2.arcLength(cnt, True)
        circularity = 0
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)

        pill_mask = np.zeros(cleaned_mask.shape, dtype=np.uint8)
        cv2.drawContours(pill_mask, [cnt], -1, 255, -1)
        mean_hsv = cv2.mean(hsv_img, mask=pill_mask)[:3]
        mean_hue = mean_hsv[0]

        defect_reasons = []

        if is_border and ignore_borders:
            status = "BORDER (Excluded from Audit)"
            box_color = (120, 120, 120)  # Neutral Grey
        else:
            audited_count += 1
            if area > (median_batch_area * 1.85):
                defect_reasons.append("Overlapping Stack")
            elif area < (median_batch_area * area_ratio_threshold):
                defect_reasons.append("Broken / Fragment")
            elif circularity < circularity_threshold:
                defect_reasons.append("Chipped / Irregular")

            hue_diff = abs(mean_hue - median_batch_hue)
            if hue_diff > 30 and hue_diff < 150:
                defect_reasons.append("Foreign Pill")

            if len(defect_reasons) > 0:
                defective_count += 1
                status = "DEFECT: " + ", ".join(defect_reasons)
                box_color = (0, 0, 255)  # Red
            else:
                status = "PASS (Normal)"
                box_color = (0, 255, 0)  # Green

        cv2.drawContours(annotated_img, [cnt], -1, box_color, 2)
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 1)

        label = (
            f"#{idx}"
            if len(raw_pills) > 25
            else f"#{idx}: {status.split(':')[0]}"
        )
        cv2.putText(
            annotated_img,
            label,
            (x, max(y - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            box_color,
            1,
        )

        pill_reports.append(
            {
                "Pill_ID": idx,
                "Status": status,
                "Area_px": round(area, 1),
                "Circularity": round(circularity, 2),
                "Mean_Hue": round(mean_hue, 1),
            }
        )

    effective_total = audited_count if ignore_borders else len(raw_pills)
    batch_status = (
        "BATCH PASS"
        if defective_count == 0
        else "BATCH REJECTED (Defects Found)"
    )
    banner_color = (0, 255, 0) if defective_count == 0 else (0, 0, 255)

    cv2.putText(
        annotated_img,
        f"Audited: {effective_total} | Defects: {defective_count} | {batch_status}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        banner_color,
        2,
    )

    return (
        annotated_img,
        effective_total,
        defective_count,
        pill_reports,
        batch_status,
    )