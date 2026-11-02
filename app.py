"""
🚧 AI-Based Intelligent Road Damage Detection & Risk Assessment Dashboard
==========================================================================
Premium Streamlit dashboard with dark theme, analytics, and GPS mapping.

Usage:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
import math
from datetime import datetime

from ultralytics import YOLO
from config import (
    get_model_path, CONFIDENCE_THRESHOLD, DEVICE, THEME,
    CLASS_NAMES, DEFAULT_SPEED_KMH
)
from risk_assessment import (
    calculate_area_ratio, calculate_severity, calculate_impact_score,
    get_risk_level, generate_road_health_report, DamageReport
)

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="🚧 Road Damage Detection — AI Dashboard",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS — Premium Dark Glassmorphism Theme
# ==============================================================================
st.markdown(f"""
<style>
    /* --- Global --- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {{
        background: linear-gradient(135deg, {THEME['bg_primary']} 0%, {THEME['bg_secondary']} 100%);
        font-family: 'Inter', sans-serif;
    }}

    /* --- Sidebar --- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0d0d2b 0%, #1a1a3e 100%) !important;
        border-right: 1px solid rgba(108, 92, 231, 0.2);
    }}
    [data-testid="stSidebar"] * {{
        color: #e0e0f0 !important;
    }}

    /* --- Headers --- */
    h1, h2, h3 {{
        background: linear-gradient(90deg, {THEME['gradient_start']}, {THEME['gradient_end']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }}

    /* --- Glass Cards --- */
    .glass-card {{
        background: rgba(25, 25, 60, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }}
    .glass-card:hover {{
        border-color: rgba(108, 92, 231, 0.5);
        box-shadow: 0 8px 32px rgba(108, 92, 231, 0.15);
        transform: translateY(-2px);
    }}

    /* --- Metric Cards --- */
    .metric-card {{
        background: linear-gradient(135deg, rgba(25, 25, 60, 0.8), rgba(40, 40, 80, 0.6));
        border: 1px solid rgba(108, 92, 231, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108, 92, 231, 0.2);
    }}
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {THEME['gradient_start']}, {THEME['gradient_end']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    .metric-label {{
        color: {THEME['text_secondary']};
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }}

    /* --- Risk Badges --- */
    .badge-severe {{
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }}
    .badge-moderate {{
        background: linear-gradient(135deg, #ffa502, #ffbe76);
        color: #1a1a2e;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }}
    .badge-low {{
        background: linear-gradient(135deg, #2ed573, #7bed9f);
        color: #1a1a2e;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }}

    /* --- Health Index Bar --- */
    .health-bar-container {{
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        margin: 10px 0;
    }}
    .health-bar {{
        height: 100%;
        border-radius: 12px;
        transition: width 1s ease;
    }}

    /* --- Streamlit element overrides --- */
    .stButton > button {{
        background: linear-gradient(135deg, {THEME['accent']}, {THEME['accent_light']}) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108, 92, 231, 0.4) !important;
    }}

    [data-testid="stMetric"] {{
        background: rgba(25, 25, 60, 0.6);
        border: 1px solid rgba(108, 92, 231, 0.2);
        border-radius: 12px;
        padding: 16px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(25, 25, 60, 0.6) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(108, 92, 231, 0.2) !important;
        color: {THEME['text_secondary']} !important;
        padding: 10px 20px !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {THEME['accent']}, {THEME['accent_light']}) !important;
        color: white !important;
    }}

    /* --- File Uploader --- */
    [data-testid="stFileUploader"] {{
        background: rgba(25, 25, 60, 0.4);
        border: 2px dashed rgba(108, 92, 231, 0.3);
        border-radius: 16px;
        padding: 20px;
    }}

    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Model Loading
# ==============================================================================
@st.cache_resource
def load_model():
    """Load the YOLO model with caching."""
    model_path = get_model_path()
    return YOLO(model_path)


model = load_model()


# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 3rem;">🚧</div>
        <h2 style="margin: 10px 0 5px 0;">Road Damage AI</h2>
        <p style="color: #b0b0d0; font-size: 0.85rem;">Intelligent Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # System Status
    st.markdown("### ⚡ System Status")
    device_str = "🟢 GPU (CUDA)" if isinstance(DEVICE, int) else "🟡 CPU"
    st.markdown(f"**Device:** {device_str}")
    model_file = get_model_path()
    st.markdown(f"**Model:** `{os.path.basename(model_file)}`")
    # Show actual classes from loaded model
    model_classes = list(model.names.values()) if hasattr(model, 'names') else CLASS_NAMES
    st.markdown(f"**Classes:** {', '.join(model_classes)}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    assumed_speed = st.number_input("Vehicle Speed (km/h)", 10.0, 120.0, DEFAULT_SPEED_KMH, 5.0)

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#666; font-size:0.75rem;'>"
        f"Built with YOLOv8x + PyTorch<br>"
        f"© 2026 Road Damage Detection"
        f"</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# Main Content — Tabbed Layout
# ==============================================================================

# Hero Section
st.markdown("""
<div class="glass-card" style="text-align: center; padding: 30px;">
    <h1 style="font-size: 2.2rem; margin-bottom: 8px;">
        🚧 AI-Based Intelligent Road Damage Detection
    </h1>
    <p style="color: #b0b0d0; font-size: 1.05rem; max-width: 700px; margin: 0 auto;">
        Real-time road damage detection with severity classification,
        impact scoring, and road health assessment powered by YOLOv8x.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_detect, tab_analytics, tab_gps, tab_about = st.tabs([
    "🔍 Detection", "📊 Analytics", "🗺️ GPS Mapping", "ℹ️ About"
])


# ==============================================================================
# TAB 1: Detection
# ==============================================================================
with tab_detect:
    upload = st.file_uploader(
        "Upload an image or video for analysis",
        type=['jpg', 'png', 'jpeg', 'mp4', 'avi'],
        help="Supported formats: JPG, PNG, JPEG, MP4, AVI"
    )

    if upload is not None:
        file_ext = upload.name.split('.')[-1].lower()

        # ── Image Processing ──
        if file_ext in ['jpg', 'png', 'jpeg']:
            file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            col_orig, col_result = st.columns(2)
            with col_orig:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 📷 Original Image")
                st.image(img_rgb, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🚀 Run Detection", use_container_width=True):
                with st.spinner("🔍 Analyzing road conditions..."):
                    results = model.predict(img, conf=confidence, device=DEVICE, verbose=False)
                    result = results[0]
                    annotated_img = result.plot()
                    img_h, img_w = img.shape[:2]

                    # Build report
                    boxes_list = []
                    confs_list = []
                    cls_ids_list = []
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes_list.append((x1, y1, x2, y2))
                        confs_list.append(float(box.conf[0]))
                        cls_ids_list.append(int(box.cls[0]))

                    report = generate_road_health_report(
                        boxes_list, confs_list, cls_ids_list,
                        model.names, img_w, img_h, assumed_speed
                    )

                    # Draw risk labels
                    for damage in report.damages:
                        x1, y1, x2, y2 = damage.bbox
                        risk_colors = {"Severe": (0,0,255), "Moderate": (0,165,255), "Low": (0,255,0)}
                        color = risk_colors.get(damage.risk_level, (255,255,255))
                        label = f"Impact: {damage.impact_score:.2f} ({damage.risk_level})"
                        cv2.putText(annotated_img, label, (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

                    with col_result:
                        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                        st.markdown("#### 🎯 Detection Results")
                        st.image(annotated_rgb, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # ── Dashboard Metrics ──
                    st.markdown("---")
                    st.markdown("### 📋 Road Health Report")

                    # Top metrics row
                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{report.total_damages}</div>
                            <div class="metric-label">Total Damages</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="background: linear-gradient(90deg, #ff4757, #ff6b81); -webkit-background-clip: text;">{report.severe_count}</div>
                            <div class="metric-label">🔴 Severe</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="background: linear-gradient(90deg, #ffa502, #ffbe76); -webkit-background-clip: text;">{report.moderate_count}</div>
                            <div class="metric-label">🟡 Moderate</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="background: linear-gradient(90deg, #2ed573, #7bed9f); -webkit-background-clip: text;">{report.low_count}</div>
                            <div class="metric-label">🟢 Low</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m5:
                        hi = report.health_index
                        hi_color = "#2ed573" if hi > 0.7 else "#ffa502" if hi > 0.4 else "#ff4757"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="background: {hi_color}; -webkit-background-clip: text;">{hi:.2f}</div>
                            <div class="metric-label">🏥 Health Index</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Health bar
                    hi_pct = int(report.health_index * 100)
                    bar_color = "#2ed573" if hi > 0.7 else "#ffa502" if hi > 0.4 else "#ff4757"
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="color: {THEME['text_secondary']}; font-weight: 600;">Road Health Index</span>
                            <span class="badge-{'low' if hi > 0.7 else 'moderate' if hi > 0.4 else 'severe'}">{report.health_category}</span>
                        </div>
                        <div class="health-bar-container">
                            <div class="health-bar" style="width: {hi_pct}%; background: linear-gradient(90deg, {bar_color}, {bar_color}99);"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Damage details table
                    if report.damages:
                        st.markdown("### 🔎 Detection Details")
                        for i, d in enumerate(report.damages):
                            badge_class = "badge-severe" if d.risk_level == "Severe" else "badge-moderate" if d.risk_level == "Moderate" else "badge-low"
                            st.markdown(f"""
                            <div class="glass-card" style="padding: 16px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong style="color: white; font-size: 1.05rem;">#{i+1} {d.class_name}</strong>
                                        <span class="{badge_class}" style="margin-left: 12px;">{d.risk_level}</span>
                                    </div>
                                    <div style="color: {THEME['text_secondary']};">
                                        Conf: {d.confidence:.2f} | Area: {d.area_ratio:.4f} | Severity: {d.severity:.4f} | Impact: {d.impact_score:.2f}
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    # Store report in session for analytics tab
                    st.session_state['last_report'] = report

        # ── Video Processing ──
        elif file_ext in ['mp4', 'avi']:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎥 Video Upload")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}')
            tfile.write(upload.read())
            tfile.flush()

            st.video(tfile.name)

            if st.button("🚀 Process Video (Sample Frames)", use_container_width=True):
                with st.spinner("🎥 Processing video frames... This may take a while."):
                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    st.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")

                    # Sample every Nth frame
                    sample_interval = max(1, total_frames // 10)
                    sampled_results = []
                    progress_bar = st.progress(0)

                    frame_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_idx % sample_interval == 0:
                            results = model.predict(frame, conf=confidence, device=DEVICE, verbose=False)
                            result = results[0]
                            annotated = result.plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                            num_detections = len(result.boxes)
                            sampled_results.append({
                                "frame": frame_idx,
                                "detections": num_detections,
                                "image": annotated_rgb
                            })
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))

                        frame_idx += 1

                    cap.release()
                    progress_bar.progress(1.0)

                    # Show sampled frames
                    if sampled_results:
                        st.markdown("### 📸 Sampled Detection Frames")
                        cols = st.columns(min(len(sampled_results), 3))
                        for idx, sr in enumerate(sampled_results[:9]):
                            col = cols[idx % 3]
                            with col:
                                st.image(sr["image"], caption=f"Frame {sr['frame']} — {sr['detections']} detections", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">📸</div>
            <h3 style="margin-bottom: 8px;">Upload an Image or Video</h3>
            <p style="color: #b0b0d0;">
                Drop a road image (JPG/PNG) or dashcam video (MP4) above to begin AI-powered damage detection.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: Analytics
# ==============================================================================
with tab_analytics:
    st.markdown("### 📊 Detection Analytics")

    if 'last_report' in st.session_state:
        report = st.session_state['last_report']

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Risk Distribution Pie Chart
                risk_data = {
                    "Risk Level": ["Severe", "Moderate", "Low"],
                    "Count": [report.severe_count, report.moderate_count, report.low_count]
                }
                fig_pie = go.Figure(data=[go.Pie(
                    labels=risk_data["Risk Level"],
                    values=risk_data["Count"],
                    hole=0.5,
                    marker_colors=["#ff4757", "#ffa502", "#2ed573"],
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white'),
                )])
                fig_pie.update_layout(
                    title="Risk Distribution",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#b0b0d0'),
                    showlegend=True,
                    legend=dict(font=dict(color='#b0b0d0')),
                    height=400,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                # Damage Type Bar Chart
                type_counts = {}
                for d in report.damages:
                    type_counts[d.class_name] = type_counts.get(d.class_name, 0) + 1

                if type_counts:
                    fig_bar = go.Figure(data=[go.Bar(
                        x=list(type_counts.keys()),
                        y=list(type_counts.values()),
                        marker_color=['#6c5ce7', '#00cec9', '#fdcb6e', '#e17055'][:len(type_counts)],
                        text=list(type_counts.values()),
                        textposition='outside',
                        textfont=dict(color='white'),
                    )])
                    fig_bar.update_layout(
                        title="Damage Type Distribution",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#b0b0d0'),
                        xaxis=dict(gridcolor='rgba(108,92,231,0.1)'),
                        yaxis=dict(gridcolor='rgba(108,92,231,0.1)', title='Count'),
                        height=400,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            # Severity vs Impact Scatter
            if report.damages:
                severities = [d.severity for d in report.damages]
                impacts = [d.impact_score for d in report.damages]
                labels = [d.class_name for d in report.damages]
                risk_colors_map = {"Severe": "#ff4757", "Moderate": "#ffa502", "Low": "#2ed573"}
                colors = [risk_colors_map.get(d.risk_level, "#6c5ce7") for d in report.damages]

                fig_scatter = go.Figure(data=[go.Scatter(
                    x=severities,
                    y=impacts,
                    mode='markers+text',
                    marker=dict(size=15, color=colors, line=dict(width=1, color='white')),
                    text=labels,
                    textposition='top center',
                    textfont=dict(color='#b0b0d0', size=11),
                )])
                fig_scatter.update_layout(
                    title="Severity vs Impact Score",
                    xaxis_title="Severity",
                    yaxis_title="Impact Score",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#b0b0d0'),
                    xaxis=dict(gridcolor='rgba(108,92,231,0.15)'),
                    yaxis=dict(gridcolor='rgba(108,92,231,0.15)'),
                    height=400,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        except ImportError:
            st.warning("Install plotly for charts: `pip install plotly`")

    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px;">
            <div style="font-size: 3rem; margin-bottom: 16px;">📊</div>
            <h3>No Detection Data Yet</h3>
            <p style="color: #b0b0d0;">Run a detection in the Detection tab first to see analytics.</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 3: GPS Mapping
# ==============================================================================
with tab_gps:
    st.markdown("### 🗺️ GPS-Based Damage Mapping")

    st.markdown("""
    <div class="glass-card">
        <p style="color: #b0b0d0;">
            GPS mapping allows you to track road damage locations geographically.
            When integrated with a GPS-enabled camera or mobile device, detected damages
            can be pinned on a map with severity information.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Demo map
    col_map, col_form = st.columns([2, 1])

    with col_form:
        st.markdown("#### 📍 Add Manual Location")
        lat = st.number_input("Latitude", value=16.4419, format="%.4f")
        lon = st.number_input("Longitude", value=80.6225, format="%.4f")
        damage_type = st.selectbox("Damage Type", CLASS_NAMES)
        severity_input = st.selectbox("Severity", ["Low", "Moderate", "Severe"])

        if st.button("📌 Pin Location"):
            if 'gps_pins' not in st.session_state:
                st.session_state['gps_pins'] = []
            st.session_state['gps_pins'].append({
                "lat": lat, "lon": lon,
                "type": damage_type, "severity": severity_input,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success(f"📌 Pinned {damage_type} at ({lat:.4f}, {lon:.4f})")

    with col_map:
        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[16.4419, 80.6225], zoom_start=14,
                           tiles="CartoDB dark_matter")

            # Add pinned locations
            if 'gps_pins' in st.session_state:
                severity_colors = {"Low": "green", "Moderate": "orange", "Severe": "red"}
                for pin in st.session_state['gps_pins']:
                    folium.CircleMarker(
                        location=[pin["lat"], pin["lon"]],
                        radius=10,
                        popup=f"{pin['type']} - {pin['severity']}<br>{pin['time']}",
                        color=severity_colors.get(pin["severity"], "blue"),
                        fill=True,
                        fill_opacity=0.7,
                    ).add_to(m)

            st_folium(m, width=700, height=500)

        except ImportError:
            st.info("Install folium and streamlit-folium for map visualization:\n`pip install folium streamlit-folium`")

            # Fallback display
            if 'gps_pins' in st.session_state and st.session_state['gps_pins']:
                import pandas as pd
                df = pd.DataFrame(st.session_state['gps_pins'])
                st.dataframe(df, use_container_width=True)


# ==============================================================================
# TAB 4: About
# ==============================================================================
with tab_about:
    st.markdown("""
    <div class="glass-card">
        <h3>🏗️ System Architecture</h3>
        <p style="color: #b0b0d0;">
            This system uses <strong>YOLOv8x</strong> (Extra-Large) for real-time object detection,
            combined with custom mathematical models for risk assessment and road health indexing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("""
        <div class="glass-card">
            <h4>🧠 Detection Pipeline</h4>
            <ul style="color: #b0b0d0;">
                <li><strong>Backbone:</strong> CSPDarknet53 feature extraction</li>
                <li><strong>Neck:</strong> PANet multi-scale feature fusion</li>
                <li><strong>Head:</strong> Decoupled head with anchor-free detection</li>
                <li><strong>NMS-Free:</strong> End-to-end inference pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="glass-card">
            <h4>📐 Mathematical Models</h4>
            <ul style="color: #b0b0d0;">
                <li><strong>Area Ratio:</strong> DamageArea / ImageArea</li>
                <li><strong>Severity:</strong> AreaRatio × Confidence × ClassWeight</li>
                <li><strong>Impact:</strong> AreaRatio × Speed × Confidence × Scale</li>
                <li><strong>Health Index:</strong> 1 - Σ(severity), clamped [0,1]</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4>⚡ Key Features</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; color: #b0b0d0;">
            <div>✅ Static image detection</div>
            <div>✅ Real-time video tracking</div>
            <div>✅ Vehicle speed estimation</div>
            <div>✅ Impact score calculation</div>
            <div>✅ Severity classification</div>
            <div>✅ Road Health Index</div>
            <div>✅ GPS damage mapping</div>
            <div>✅ Analytics dashboard</div>
            <div>✅ GPU-accelerated inference</div>
            <div>✅ JSON report export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Performance benchmark
    st.markdown("""
    <div class="glass-card">
        <h4>📈 Performance Benchmarks</h4>
        <table style="width: 100%; color: #b0b0d0; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid rgba(108,92,231,0.2);">
                <th style="padding: 10px; text-align: left;">Model</th>
                <th style="padding: 10px; text-align: center;">mAP50</th>
                <th style="padding: 10px; text-align: center;">FPS (T4)</th>
                <th style="padding: 10px; text-align: center;">Parameters</th>
            </tr>
            <tr style="border-bottom: 1px solid rgba(108,92,231,0.1);">
                <td style="padding: 10px;">YOLOv8n</td>
                <td style="padding: 10px; text-align: center;">-</td>
                <td style="padding: 10px; text-align: center;">~170</td>
                <td style="padding: 10px; text-align: center;">3.2M</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(108,92,231,0.1);">
                <td style="padding: 10px;">YOLOv8s</td>
                <td style="padding: 10px; text-align: center;">-</td>
                <td style="padding: 10px; text-align: center;">~130</td>
                <td style="padding: 10px; text-align: center;">11.2M</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(108,92,231,0.1);">
                <td style="padding: 10px;">YOLOv8m</td>
                <td style="padding: 10px; text-align: center;">-</td>
                <td style="padding: 10px; text-align: center;">~100</td>
                <td style="padding: 10px; text-align: center;">25.9M</td>
            </tr>
            <tr style="background: rgba(108,92,231,0.15);">
                <td style="padding: 10px;"><strong>YOLOv8x ★</strong></td>
                <td style="padding: 10px; text-align: center;"><strong>Trained</strong></td>
                <td style="padding: 10px; text-align: center;">~50</td>
                <td style="padding: 10px; text-align: center;">68.2M</td>
            </tr>
        </table>
        <p style="color: #666; font-size: 0.75rem; margin-top: 8px;">★ = Your trained model</p>
    </div>
    """, unsafe_allow_html=True)
