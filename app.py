import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import tempfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PlayVision Analytics", 
    page_icon="1.png", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Persistent storage
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 2. PREMIUM DARK UI STYLING ---
accent_color = "#FF3131"  # Neon Red
success_green = "#00FF41" # Matrix Green

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

    .stApp {{ background-color: #050505; color: #E0E0E0; }}
    .stDeployButton, footer {{visibility: hidden;}}

    /* Sidebar Arrow for Mobile */
    header {{ background-color: transparent !important; }}
    button[data-testid="bundle--menubar-button"] {{
        background-color: {accent_color} !important;
        color: white !important;
        border-radius: 50% !important;
        box-shadow: 0 0 15px {accent_color};
    }}

    /* Hero Section */
    .hero-text {{
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        padding: 40px 10px;
        background: linear-gradient(180deg, #1a1a1a 0%, #050505 100%);
        border-radius: 20px;
        border: 1px solid #333;
        margin-bottom: 30px;
    }}

    /* Glassmorphism Report Cards */
    .report-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 20px;
        transition: 0.3s;
    }}
    .report-card:hover {{ border: 1px solid {accent_color}; transform: translateY(-5px); }}
    
    .metric-value {{ 
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(32px, 6vw, 54px); 
        color: {accent_color}; 
        text-shadow: 0 0 10px {accent_color}44;
    }}

    /* Feature Badges */
    .badge {{
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: bold;
        background: #333;
        margin-right: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("4.png", width=500)
    st.markdown("<h2 style='font-family:Orbitron;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    st.write("Analyze movement, speed, and reaction times using Computer Vision.")
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Drop Match Footage Here", type=["mp4", "mov", "avi"])
    
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.analysis_results = None
        st.rerun()

# --- 4. HOME PAGE CONTENT ---
if uploaded_file is None:
    # --- HERO SECTION ---
    st.markdown(f"""
    <div class='hero-text'>
        <h1 style='color:{accent_color}; margin-bottom:0;'>PLAYVISION ANALYTICS</h1>
        <p style='letter-spacing: 3px; color: #888;'>SMART PLAYER PERFORMANCE ANALYZER FOR KHO-KHO</p>
    </div>
    """, unsafe_allow_html=True)

    # --- FEATURE GRID ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='report-card'><h3>🏃 Velocity</h3><p>Real-time tracking of runner evasion speeds.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='report-card'><h3>🛡️ Agility</h3><p>Measuring transition times between chasers.</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='report-card'><h3>📊 Reports</h3><p>AI-generated feedback for coaching staff.</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("👈 **GET STARTED:** Open the sidebar and upload a match video to generate live AI metrics.")
    
    # # Visual Filler
    # st.image("kho-kho.png", use_container_width=True)

# --- 5. ANALYSIS ENGINE ---
else:
    st.markdown(f"<h2 style='font-family:Orbitron; color:{accent_color};'>LIVE SESSION</h2>", unsafe_allow_html=True)
    
    col_vid, col_inst = st.columns([2, 1])
    
    with col_vid:
        st.video(uploaded_file)
    
    with col_inst:
        st.markdown(f"""
        <div class='report-card'>
            <span class='badge' style='background:{accent_color};'>YOLOv8 Engine</span>
            <h3>Match Ready</h3>
            <p style='color:#bbb;'>Click start to begin frame-by-frame player tracking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 INITIATE AI TRACKING", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name

            stats = defaultdict(lambda: {'pos': [], 'dist': 0})
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("🛰️ Processing Neural Network...")

            model = YOLO('yolov8n.pt')
            results = model.track(source=temp_path, stream=True, persist=True, imgsz=160, conf=0.4, classes=[0], verbose=False, vid_stride=30)

            for i, r in enumerate(results):
                if r.boxes.id is not None:
                    boxes = r.boxes.xywh.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        cx, cy = box[0], box[1]
                        p_data = stats[track_id]
                        if p_data['pos']:
                            prev_x, prev_y = p_data['pos'][-1]
                            p_data['dist'] += np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                        p_data['pos'].append((cx, cy))
                progress_bar.progress(min((i + 1) / 40, 1.0))

            all_speeds = [v['dist']/len(v['pos']) for v in stats.values() if len(v['pos']) > 2]
            r_speeds = [s for s in all_speeds if s > 30]
            c_speeds = [s for s in all_speeds if s <= 30]
            
            st.session_state.analysis_results = {
                "runner_speed": np.mean(r_speeds) if r_speeds else 0,
                "chaser_speed": np.mean(c_speeds) if c_speeds else 0
            }
            status_text.success("✅ Match Data Compiled!")
            if os.path.exists(temp_path): os.remove(temp_path)

    # --- 6. RESULTS DASHBOARD ---
    if st.session_state.analysis_results:
        st.markdown("---")
        res = st.session_state.analysis_results
        r_col, c_col = st.columns(2)
        
        with r_col:
            is_good_r = res['runner_speed'] > 60
            st.markdown(f"""
            <div class='report-card'>
                <h3 style='margin:0;'>🏃 RUNNER TEAM</h3>
                <div class='metric-value'>{res['runner_speed']:.1f}</div>
                <p style='color:#888;'>VELOCITY INDEX</p>
                <p class='{"status-good" if is_good_r else "status-warn"}'>
                    REPORT: {"EXCELLENT AGILITY DETECTED" if is_good_r else "LOW EVASION SPEED"}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with c_col:
            is_good_c = res['chaser_speed'] > 20
            st.markdown(f"""
            <div class='report-card'>
                <h3 style='margin:0;'>🛡️ CHASER TEAM</h3>
                <div class='metric-value'>{res['chaser_speed']:.1f}</div>
                <p style='color:#888;'>TRANSITION SCORE</p>
                <p class='{"status-good" if is_good_c else "status-warn"}'>
                    REPORT: {"STRONG ATTACK PACE" if is_good_c else "PASSIVE CHASE DETECTED"}
                </p>
            </div>
            """, unsafe_allow_html=True)