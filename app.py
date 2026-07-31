import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="3D Room Spatial AI",
    page_icon="🧊",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b2559;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">AI-Powered 3D Spatial Room Reconstruction & Layout Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BCA Capstone Project — Spatial Depth Perception & Boundary Mapping Engine</p>', unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR CONTROLS & MODEL CACHING
# ==========================================
st.sidebar.header("🕹️ System Controls")
uploaded_file = st.sidebar.file_uploader(
    "Upload 2D Room Photograph", 
    type=["jpg", "jpeg", "png"]
)

conf_threshold = st.sidebar.slider(
    "YOLOv8 Object Detection Confidence",
    min_value=0.15,
    max_value=0.80,
    value=0.35,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Framework Models")
st.sidebar.info(
    "**1. Depth Engine:** Depth Anything V2 (Vision Transformer)\n\n"
    "**2. Boundary Engine:** YOLOv8 Nano (Indoor Object & Boundary Detector)"
)

@st.cache_resource
def load_depth_model():
    return pipeline(
        task="depth-estimation", 
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

with st.spinner("Initializing Spatial Vision AI Engines..."):
    depth_pipe = load_depth_model()
    yolo_model = load_yolo_model()

# ==========================================
# 3. MAIN PROCESSING & RECONSTRUCTION FLOW
# ==========================================
if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Raw 2D Input Photo")
        st.image(raw_image, use_container_width=True)

    # 1. Monocular Depth Estimation
    with st.spinner("Calculating Monocular Depth Geometry..."):
        processing_size = (160, 160)
        img_resized = raw_image.resize(processing_size)
        depth_result = depth_pipe(img_resized)
        depth_map = np.array(depth_result["depth"])

    with col2:
        st.subheader("2. Monocular Depth Map")
        st.image(depth_map, clamp=True, use_container_width=True)

    # 2. YOLOv8 Object Detection
    with st.spinner("Scanning Spatial Boundaries & Objects..."):
        yolo_results = yolo_model(raw_image, conf=conf_threshold)[0]
        annotated_bgr = yolo_results.plot()
        annotated_rgb = annotated_bgr[..., ::-1]

        detected_objects = []
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            conf = float(box.conf[0])
            detected_objects.append(f"{label.capitalize()} ({conf*100:.0f}%)")

    with col3:
        st.subheader("3. YOLOv8 Object Detection")
        st.image(annotated_rgb, use_container_width=True)

    st.divider()

    # ==========================================
    # 4. DIRECT 2D TO 3D MATRIX TRANSFORMATION
    # ==========================================
    st.subheader("4. Interactive 3D Spatial Reconstruction (Real-World Texture)")
    st.caption("Rotate and inspect the 3D room depth space directly. Clean, intuitive 1:1 pixel depth mapping.")

    img_np = np.array(img_resized)
    height, width, _ = img_np.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = depth_map.flatten()

    colors = img_np.reshape(-1, 3)
    color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x_flat,
            y=-y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=2.5,
                color=color_strings,
                opacity=0.9
            )
        )
    ])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Width (X)"),
            yaxis=dict(title="Height (Y)"),
            zaxis=dict(title="Depth Distance (Z)"),
            aspectmode='data',
            camera=dict(
                eye=dict(x=0, y=-1.6, z=0.6)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=580
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # 5. DETECTED SPATIAL ANCHORS TABLE
    # ==========================================
    st.markdown("### 🏷️ Spatial Anchors & Detected Obstacles")
    if detected_objects:
        st.success(f"Successfully identified {len(detected_objects)} spatial boundary items: **{', '.join(detected_objects)}**")
    else:
        st.warning("No furniture/boundary objects detected at current confidence threshold. Try adjusting the slider in the sidebar.")

else:
    st.info("👈 Upload an indoor room photograph from the left sidebar to trigger live 3D reconstruction & object detection.")