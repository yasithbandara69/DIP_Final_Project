import importlib
import os
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as dip

importlib.reload(dip)

# -------------------------------------------------------------------
# SAAS DASHBOARD THEME CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PharmaVision OS | Automated Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0b0f17; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .saas-header { display: flex; justify-content: space-between; align-items: center; padding: 1rem 1.5rem; background: #111827; border: 1px solid #1f2937; border-radius: 12px; margin-bottom: 1.5rem; }
    .saas-title { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0; display: flex; align-items: center; gap: 0.6rem; }
    .saas-badge-live { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #059669; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 1rem 1.2rem; }
    .kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 0.25rem; }
    .kpi-val { font-size: 1.5rem; font-weight: 700; color: #f8fafc; }
    .status-pass { background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; color: #34d399; padding: 0.85rem 1.2rem; border-radius: 8px; font-weight: 600; margin: 1rem 0; }
    .status-reject { background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; color: #f87171; padding: 0.85rem 1.2rem; border-radius: 8px; font-weight: 600; margin: 1rem 0; }
    section[data-testid="stSidebar"] { background-color: #0d131f; border-right: 1px solid #1f2937; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="saas-header">
    <div class="saas-title">
        <span>🔬</span> PharmaVision OS
        <span style="font-size: 0.9rem; font-weight: 400; color: #64748b;">| Vision Inspection Core v2.5</span>
    </div>
    <div><span class="saas-badge-live">● System Active</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Calibration Controls")

    with st.expander("🎚️ Detection Thresholds", expanded=True):
        min_area = st.slider("Min Tablet Area (px)", 200, 3000, 400, 50)
        circularity_thresh = st.slider(
            "Circularity Tolerance", 0.15, 0.85, 0.26, 0.02
        )
        area_ratio_thresh = st.slider(
            "Batch Size Tolerance", 0.35, 0.95, 0.52, 0.05
        )
        ignore_borders = st.checkbox(
            "Ignore Partial Border Tablets",
            value=True,
            help="Excludes tablets cut off by the edge of the image from triggering false defect alarms.",
        )

    with st.expander("🎨 Segmentation & Shadows", expanded=True):
        bg_mode = st.selectbox(
            "Tray Background Style",
            [
                "Colored Pills on White Tray",
                "White Pills on White Tray (Edge Rim)",
                "White Pills on Dark/Colored Tray",
            ],
            help="Select 'White Pills on White Tray (Edge Rim)' when inspecting white tablets on white paper.",
        )
        white_thresh = st.slider("White Paper Cutoff", 200, 254, 235, 1)
        shadow_rejection = st.slider(
            "Shadow Rejection (Sat)",
            min_value=5,
            max_value=60,
            value=10,
            step=1,
            help="Lower value (e.g. 8-12) detects shaded pills; higher value removes dark shadows.",
        )

    with st.expander("🧩 Watershed Tuning", expanded=False):
        enable_ws = st.checkbox("Enable Watershed", value=True)
        peak_dist = st.slider("Peak Separation Distance", 7, 45, 17, 2)

tab_single, tab_batch, tab_analytics = st.tabs(
    ["📷 Single Batch Audit", "📁 Multi-Batch Runner", "📊 Batch Analytics"]
)

# ===================================================================
# TAB 1: SINGLE IMAGE AUDIT
# ===================================================================
with tab_single:
    uploaded_file = st.file_uploader(
        "Ingest Tablet Array",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        temp_path = "temp_single.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        t_start = time.time()
        img, metadata = dip.load_and_validate_image(temp_path)

        # Segment with selected mode
        mask = dip.segment_pills(
            img,
            bg_mode=bg_mode,
            white_threshold=white_thresh,
            min_saturation=shadow_rejection,
        )

        # Inspect and evaluate defects
        result_img, total, defects, report_data, verdict = (
            dip.inspect_pills_and_detect_defects(
                mask,
                img,
                min_area=min_area,
                circularity_threshold=circularity_thresh,
                area_ratio_threshold=area_ratio_thresh,
                use_watershed=enable_ws,
                peak_distance=peak_dist,
                ignore_borders=ignore_borders,
            )
        )
        runtime_ms = round((time.time() - t_start) * 1000, 2)

        # SaaS Metric KPI Cards
        st.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Audited Units</div>
                <div class="kpi-val">{total} <span style="font-size:0.9rem; color:#94a3b8;">tablets</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Defects Flagged</div>
                <div class="kpi-val" style="color: {'#34d399' if defects == 0 else '#f87171'};">{defects} <span style="font-size:0.9rem; color:#94a3b8;">units</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Inference Latency</div>
                <div class="kpi-val">{runtime_ms} <span style="font-size:0.9rem; color:#94a3b8;">ms</span></div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Batch Pass Rate</div>
                <div class="kpi-val">{round(((total-defects)/max(total, 1))*100, 1)}%</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if verdict == "BATCH PASS":
            st.markdown(
                f'<div class="status-pass">✔ <strong>{verdict}</strong> — All audited units comply with geometric specifications.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-reject">✖ <strong>{verdict}</strong> — Outliers detected. Secondary review recommended.</div>',
                unsafe_allow_html=True,
            )

        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.markdown("##### 📥 Raw Ingestion Viewport")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True
            )
        with col_proc:
            st.markdown("##### 🔬 Annotated Geometric Segmentation")
            st.image(
                cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

        with st.expander("👁️ View Intermediate Binary Segmentation Mask"):
            st.image(
                mask,
                caption="Cleaned Morphological Mask",
                use_container_width=True,
            )

        if len(report_data) > 0:
            df = pd.DataFrame(report_data)
            st.markdown("---")
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                st.markdown("##### 📋 Tablet Telemetry Log")
            with col_t2:
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Telemetry CSV",
                    csv_data,
                    "batch_telemetry.csv",
                    "text/csv",
                    use_container_width=True,
                )
            st.dataframe(df, use_container_width=True, hide_index=True)

        if os.path.exists(temp_path):
            os.remove(temp_path)

# ===================================================================
# TAB 2: MULTI-BATCH RUNNER (FOLDER AUDIT)
# ===================================================================
with tab_batch:
    st.markdown("##### 📁 Industrial Automated Batch Runner")
    batch_dir = st.text_input("Dataset Directory Path", value="dataset/raw")
    if st.button("🚀 Execute Full Directory Audit", use_container_width=True):
        if not os.path.exists(batch_dir):
            st.error(f"Path '{batch_dir}' does not exist.")
        else:
            exts = (".png", ".jpg", ".jpeg")
            files = [f for f in os.listdir(batch_dir) if f.lower().endswith(exts)]
            if len(files) == 0:
                st.warning("No valid images discovered.")
            else:
                progress = st.progress(0)
                records = []
                os.makedirs("outputs", exist_ok=True)
                for idx, fname in enumerate(files):
                    fpath = os.path.join(batch_dir, fname)
                    ts = time.time()
                    try:
                        im, meta = dip.load_and_validate_image(fpath)
                        m = dip.segment_pills(
                            im,
                            bg_mode=bg_mode,
                            white_threshold=white_thresh,
                            min_saturation=shadow_rejection,
                        )
                        res_im, tot, defs, _, vrd = (
                            dip.inspect_pills_and_detect_defects(
                                m,
                                im,
                                min_area=min_area,
                                circularity_threshold=circularity_thresh,
                                area_ratio_threshold=area_ratio_thresh,
                                use_watershed=enable_ws and (bg_mode != "White Pills on White Tray (Edge Rim)"),
                                peak_distance=peak_dist,
                                ignore_borders=ignore_borders,
                            )
                        )
                        lat = round((time.time() - ts) * 1000, 2)
                        cv2.imwrite(
                            os.path.join("outputs", f"audited_{fname}"), res_im
                        )
                        records.append(
                            {
                                "Filename": fname,
                                "Audited_Units": tot,
                                "Defects": defs,
                                "Verdict": vrd,
                                "Latency_ms": lat,
                            }
                        )
                    except Exception as err:
                        records.append(
                            {
                                "Filename": fname,
                                "Audited_Units": 0,
                                "Defects": 0,
                                "Verdict": f"ERR: {str(err)}",
                                "Latency_ms": 0,
                            }
                        )
                    progress.progress((idx + 1) / len(files))

                df_batch = pd.DataFrame(records)
                st.success(f"Audit completed: {len(files)} batches processed.")
                st.dataframe(
                    df_batch, use_container_width=True, hide_index=True
                )
                b_csv = df_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Master Batch Report",
                    b_csv,
                    "master_batch_audit.csv",
                    "text/csv",
                )

# ===================================================================
# TAB 3: BATCH ANALYTICS
# ===================================================================
with tab_analytics:
    st.markdown("##### 📊 Production Quality Control Analytics")
    if "df" in locals() and len(report_data) > 0:
        fig = px.scatter(
            df,
            x="Area_px",
            y="Circularity",
            color="Status",
            hover_data=["Pill_ID", "Mean_Hue"],
            title="Morphometric Distribution: Tablet Area vs Circularity",
            color_discrete_map={
                "PASS (Normal)": "#10b981",
                "DEFECT: Chipped / Irregular": "#ef4444",
                "DEFECT: Broken / Fragment": "#f59e0b",
                "BORDER (Excluded from Audit)": "#6b7280",
            },
        )
        fig.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#0b0f17",
            font=dict(color="#cbd5e1"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload an image in 'Single Batch Audit' to view scatter data.")