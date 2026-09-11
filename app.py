"""
Deep Road Analysis - Interactive Streamlit Web Dashboard
Run via:
    streamlit run app.py
or:
    python main.py --web
"""

import os
import io
import json
import tempfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from road_analyzer import RoadAnalyzer, CLASS_COLORS, CATEGORY_MAP

# Page configuration
st.set_page_config(
    page_title="Deep Road Analysis // AI Vision",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-tech aesthetic
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, #161f30 0%, #0d1522 100%);
        border: 1px solid #24324d;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-safe {
        color: #10b981;
    }
    .status-caution {
        color: #f59e0b;
    }
    .status-critical {
        color: #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_analyzer(model_path="best.pt"):
    return RoadAnalyzer(model_path=model_path, device="cpu")


def main():
    st.title("🚗 Deep Road Analysis & Perception System")
    st.caption("Custom YOLO-Powered Scene Intelligence, Traffic Flow & Collision Risk Assessment")

    analyzer = load_analyzer("best.pt")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Perception Controls")
        conf_thresh = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
        iou_thresh = st.slider("IoU NMS Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

        st.subheader("HUD Display Options")
        show_hud = st.checkbox("Show HUD Telemetry Bar", value=True)
        show_danger_zone = st.checkbox("Show Ego-Vehicle Hazard Corridor", value=True)

        st.divider()
        st.subheader("Sample Media")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            use_sample = st.button("📷 Sample Image")
        with col_s2:
            use_sample_vid = st.button("🎥 Sample Video")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Road Image or Video",
        type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
    )

    image_to_process = None
    video_to_process = None

    if use_sample:
        if os.path.exists("sample_road.jpg"):
            image_to_process = cv2.imread("sample_road.jpg")
    elif use_sample_vid:
        if os.path.exists("sample_traffic.mp4"):
            video_to_process = "sample_traffic.mp4"
    elif uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_to_process = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif uploaded_file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_to_process = tfile.name
    elif os.path.exists("sample_road.jpg"):
        image_to_process = cv2.imread("sample_road.jpg")

    if video_to_process is not None:
        st.subheader("🎥 Video Analysis Pipeline")
        st.write(f"Source: `{video_to_process}`")
        if st.button("⚡ Run Full AI Video Analysis", type="primary"):
            cap = cv2.VideoCapture(video_to_process)
            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_vid_path = os.path.join("output", "web_analyzed_video.mp4")
            os.makedirs("output", exist_ok=True)
            writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            pbar = st.progress(0)
            status_text = st.empty()
            f_idx = 0
            unique_ids = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                f_idx += 1
                an = analyzer.analyze(frame, conf=conf_thresh, iou=iou_thresh, track=True)
                ann = analyzer.render_hud(frame, an, show_danger_zone=show_danger_zone, show_hud_banner=show_hud)
                writer.write(ann)
                for det in an["detections"]:
                    if det.get("track_id") is not None:
                        unique_ids.add(det["track_id"])
                pbar.progress(min(1.0, f_idx / tot_frames))
                status_text.text(f"Processed frame {f_idx}/{tot_frames} ({int(f_idx/tot_frames*100)}%) - Tracked Objects: {len(unique_ids)}")

            cap.release()
            writer.release()
            status_text.success(f"✅ Video processing complete! {f_idx} frames analyzed, {len(unique_ids)} unique objects tracked.")

            if os.path.exists(out_vid_path):
                st.video(out_vid_path)
                with open(out_vid_path, "rb") as vf:
                    st.download_button("📥 Download Analyzed Video", vf.read(), file_name="analyzed_road_traffic.mp4", mime="video/mp4")
        return

    if image_to_process is not None:
        # Run deep analysis
        analytics = analyzer.analyze(image_to_process, conf=conf_thresh, iou=iou_thresh)
        annotated_bgr = analyzer.render_hud(
            image_to_process,
            analytics,
            show_danger_zone=show_danger_zone,
            show_hud_banner=show_hud,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        summary = analytics["summary"]

        # Status Banner Alert
        status_color = "status-safe"
        if summary["safety_status"] == "CRITICAL ALERT":
            status_color = "status-critical"
            st.error(f"🚨 **CRITICAL COLLISION HAZARD DETECTED**: {summary['critical_hazards']} object(s) in direct ego trajectory!")
        elif summary["safety_status"] == "PROXIMITY WARNING":
            status_color = "status-caution"
            st.warning(f"⚠️ **PROXIMITY CAUTION**: {summary['caution_hazards']} nearby road participant(s) detected.")
        else:
            st.success("✅ **ROAD TRAFFIC CONDITION**: Clear & Safe Forward Path.")

        # Top KPI Metrics Cards
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        with k1:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Total Objects</div>"
                f"<div class='metric-value'>{summary['total_objects']}</div></div>",
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Motor Vehicles</div>"
                f"<div class='metric-value'>{summary['total_vehicles']}</div></div>",
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Pedestrians / VRU</div>"
                f"<div class='metric-value'>{summary['vulnerable_road_users']}</div></div>",
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Signals & Signs</div>"
                f"<div class='metric-value'>{summary['infrastructure_elements']}</div></div>",
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Traffic Density</div>"
                f"<div class='metric-value'>{summary['traffic_density']}</div></div>",
                unsafe_allow_html=True,
            )
        with k6:
            st.markdown(
                f"<div class='metric-card'><div class='metric-title'>Safety Score</div>"
                f"<div class='metric-value {status_color}'>{summary['road_safety_score']}/100</div></div>",
                unsafe_allow_html=True,
            )

        # Image Display
        tab1, tab2, tab3 = st.tabs(["👁️ Annotated Perception HUD", "📊 Analytics & Graphs", "📋 Detections Table"])

        with tab1:
            st.image(annotated_rgb, use_container_width=True, caption="Deep Road Analysis HUD")

            # Download Buttons
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                is_success, buffer = cv2.imencode(".jpg", annotated_bgr)
                if is_success:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name="road_analysis_annotated.jpg",
                        mime="image/jpeg",
                    )
            with col_d2:
                json_str = json.dumps(analytics, indent=2)
                st.download_button(
                    label="📄 Download Analytics JSON",
                    data=json_str,
                    file_name="road_analytics.json",
                    mime="application/json",
                )

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Class Distribution")
                if analytics["counts_by_class"]:
                    df_classes = pd.DataFrame(
                        list(analytics["counts_by_class"].items()),
                        columns=["Class", "Count"],
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(df_classes.set_index("Class"))
                else:
                    st.write("No objects detected.")

            with c2:
                st.subheader("Category Distribution")
                df_cat = pd.DataFrame(
                    list(analytics["counts_by_category"].items()),
                    columns=["Category", "Count"],
                )
                st.bar_chart(df_cat.set_index("Category"))

        with tab3:
            st.subheader("Detailed Detections List")
            if analytics["detections"]:
                df_dets = pd.DataFrame([
                    {
                        "ID": d["id"],
                        "Class": d["class"].title(),
                        "Category": d["category"],
                        "Confidence": f"{int(d['confidence']*100)}%",
                        "Hazard Level": d["hazard_level"],
                        "BBox (x1, y1, x2, y2)": str(d["bbox"]),
                        "Proximity Index": d["proximity_score"],
                    }
                    for d in analytics["detections"]
                ])
                st.dataframe(df_dets, use_container_width=True)
            else:
                st.info("No detections at this confidence threshold.")

    else:
        st.info("Please upload an image or video to begin analysis.")


if __name__ == "__main__":
    main()
