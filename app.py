import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PlayVision Analytics", 
    page_icon="🏃", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

# Persistent storage
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- 2. OPTIMIZED MODEL LOADING (Memory Saver) ---
@st.cache_resource
def load_yolo_model():
    # Load the smallest possible model and force it to CPU for Render stability
    model = YOLO('yolov8n.pt') 
    return model

# --- 3. PREMIUM DARK UI STYLING ---
accent_color = "#FF3131"  # Neon Red

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp {{ background-color: #050505; color: #E0E0E0; }}
    .stDeployButton, footer {{visibility: hidden;}}
    header {{ background-color: transparent !important; }}
    
    button[data-testid="bundle--menubar-button"] {{
        background-color: {accent_color} !important;
        color: white !important;
        border-radius: 50% !important;
        box-shadow: 0 0 15px {accent_color};
    }}

    .hero-text {{
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        padding: 40px 10px;
        background: linear-gradient(180deg, #1a1a1a 0%, #050505 100%);
        border-radius: 20px;
        border: 1px solid #333;
        margin-bottom: 30px;
    }}

    .report-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 20px;
    }}
    
    .metric-value {{ 
        font-family: 'Orbitron', sans-serif;
        font-size: clamp(32px, 6vw, 54px); 
        color: {accent_color}; 
        text-shadow: 0 0 10px {accent_color}44;
    }}

    .status-good {{ color: #00FF41; font-weight: bold; }}
    .status-warn {{ color: #FFA000; font-weight: bold; }}
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    # Use a placeholder if local images fail on Render
    st.markdown(f"<h1 style='color:{accent_color}; font-family:Orbitron;'>SPAK</h1>", unsafe_allow_html=True)
    st.markdown("### CONTROL PANEL")
    st.write("AI-Powered Computer Vision for Kho-Kho Athletics.")
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Drop Match Footage", type=["mp4", "mov", "avi"])
    
    if st.button("🗑️ Reset All Data", use_container_width=True):
        st.session_state.analysis_results = None
        st.rerun()

# --- 5. HOME PAGE & ENGINE ---
if uploaded_file is None:
    st.markdown(f"""
    <div class='hero-text'>
        <h1 style='color:{accent_color}; margin-bottom:0;'>PLAYVISION ANALYTICS</h1>
        <p style='letter-spacing: 3px; color: #888;'>AI PERFORMANCE ANALYZER</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div class='report-card'><h3>🏃 Velocity</h3><p>Runner evasion metrics.</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='report-card'><h3>🛡️ Agility</h3><p>Chaser transition speeds.</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='report-card'><h3>📊 Reports</h3><p>Coaching insights.</p></div>", unsafe_allow_html=True)
    st.info("👈 Open the sidebar to upload a match video.")

else:
    st.markdown(f"<h2 style='font-family:Orbitron; color:{accent_color};'>LIVE SESSION</h2>", unsafe_allow_html=True)
    col_vid, col_inst = st.columns([2, 1])
    
    with col_vid:
        st.video(uploaded_file)
    
    with col_inst:
        st.markdown("<div class='report-card'><h3>Match Ready</h3><p>Click below to track players.</p></div>", unsafe_allow_html=True)
        
        if st.button("🚀 INITIATE AI TRACKING", use_container_width=True):
            model = load_yolo_model()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name

            stats = defaultdict(lambda: {'pos': [], 'dist': 0})
            progress_bar = st.progress(0)
            
            # --- CRITICAL: Memory-Efficient Tracking ---
            results = model.track(
                source=temp_path, 
                stream=True, 
                persist=True, 
                imgsz=160,     # Low res for RAM savings
                conf=0.4, 
                classes=[0], 
                vid_stride=40,  # Skip frames to prevent CPU timeout/crash
                device='cpu'    # Force CPU for Render
            )

            for i, r in enumerate(results):
                if r.boxes.id is not None:
                    boxes = r.boxes.xywh.cpu().numpy()
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        cx, cy = box[0], box[1]
                        p_data = stats[track_id]
                        if p_data['pos']:
                            prev_x, prev_y = p_data['pos'][-1]
                            p_data['dist'] += np.sqrt((cx-prev_x)**2 + (cy-prev_y)**2)
                        p_data['pos'].append((cx, cy))
                progress_bar.progress(min((i + 1) / 30, 1.0))

            # Calculation logic
            all_speeds = [v['dist']/len(v['pos']) for v in stats.values() if len(v['pos']) > 2]
            r_speeds = [s for s in all_speeds if s > 30]
            c_speeds = [s for s in all_speeds if s <= 30]
            
            st.session_state.analysis_results = {
                "runner_speed": np.mean(r_speeds) if r_speeds else 0,
                "chaser_speed": np.mean(c_speeds) if c_speeds else 0
            }
            if os.path.exists(temp_path): os.remove(temp_path)
            st.rerun()

    # --- 6. RESULTS DASHBOARD ---
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        r_col, c_col = st.columns(2)
        
        with r_col:
            is_good = res['runner_speed'] > 60
            st.markdown(f"""<div class='report-card'><h3>🏃 RUNNER TEAM</h3><div class='metric-value'>{res['runner_speed']:.1f}</div>
            <p class='{"status-good" if is_good else "status-warn"}'>{"EXCELLENT AGILITY" if is_good else "LOW SPEED"}</p></div>""", unsafe_allow_html=True)

        with c_col:
            is_good = res['chaser_speed'] > 20
            st.markdown(f"""<div class='report-card'><h3>🛡️ CHASER TEAM</h3><div class='metric-value'>{res['chaser_speed']:.1f}</div>
            <p class='{"status-good" if is_good else "status-warn"}'>{"STRONG ATTACK" if is_good else "PASSIVE CHASE"}</p></div>""", unsafe_allow_html=True)