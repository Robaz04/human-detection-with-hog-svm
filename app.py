import streamlit as st
import cv2
import numpy as np
import joblib
import time
from pathlib import Path
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import exposure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HOG+SVM Human Detector",
    page_icon="🧍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0D1117;
    color: #E6EDF3;
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #161B22 0%, #0D1117 60%, #0D2137 100%);
    border: 1px solid #21262D;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-title span {
    color: #00D4AA;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #7D8590;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.3);
    color: #00D4AA;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Pipeline steps */
.pipeline-container {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0;
    overflow-x: auto;
    padding: 0.5rem 0;
}
.pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 110px;
}
.step-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.85rem;
    border: 2px solid #30363D;
    background: #161B22;
    color: #7D8590;
    transition: all 0.3s;
}
.step-circle.active {
    border-color: #00D4AA;
    background: rgba(0,212,170,0.15);
    color: #00D4AA;
    box-shadow: 0 0 16px rgba(0,212,170,0.2);
}
.step-circle.done {
    border-color: #00D4AA;
    background: #00D4AA;
    color: #0D1117;
}
.step-label {
    font-size: 0.68rem;
    color: #7D8590;
    margin-top: 6px;
    text-align: center;
    max-width: 90px;
    line-height: 1.3;
}
.step-connector {
    flex: 1;
    height: 2px;
    background: #21262D;
    min-width: 30px;
    margin-bottom: 20px;
}
.step-connector.done {
    background: linear-gradient(90deg, #00D4AA, #00A884);
}

/* Cards */
.viz-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}
.viz-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.step-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.25);
    color: #00D4AA;
    padding: 2px 8px;
    border-radius: 4px;
}
.viz-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0;
}
.viz-card-desc {
    font-size: 0.82rem;
    color: #7D8590;
    margin: 0.25rem 0 0 0;
    line-height: 1.5;
}

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-box {
    flex: 1;
    min-width: 130px;
    background: #0D1117;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #00D4AA;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-box .lbl {
    font-size: 0.75rem;
    color: #7D8590;
}
.metric-box.amber .val { color: #F5A623; }
.metric-box.blue .val  { color: #58A6FF; }
.metric-box.red .val   { color: #FF7B72; }

/* Result box */
.result-banner {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(0,168,132,0.04));
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.result-banner.no-detect {
    background: rgba(255,123,114,0.06);
    border-color: rgba(255,123,114,0.3);
}
.result-banner h3 {
    margin: 0 0 4px 0;
    font-size: 1.1rem;
    color: #00D4AA;
}
.result-banner.no-detect h3 { color: #FF7B72; }
.result-banner p { margin: 0; font-size: 0.85rem; color: #7D8590; }

/* Sidebar */
.sidebar-section {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7D8590;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 0.75rem 0;
}
.param-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 4px 0;
    border-bottom: 1px solid #21262D;
    color: #C9D1D9;
}
.param-row:last-child { border-bottom: none; }
.param-row span { color: #00D4AA; font-family: 'JetBrains Mono', monospace; }

/* Detection confidence badge */
.conf-badge {
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.25);
    color: #00D4AA;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}
.conf-badge.high { background: rgba(0,212,170,0.2); border-color: rgba(0,212,170,0.5); }
.conf-badge.med  { background: rgba(245,166,35,0.15); border-color: rgba(245,166,35,0.4); color: #F5A623; }
.conf-badge.low  { background: rgba(88,166,255,0.12); border-color: rgba(88,166,255,0.3); color: #58A6FF; }

/* Upload area */
.upload-hint {
    text-align: center;
    padding: 1rem;
    color: #7D8590;
    font-size: 0.85rem;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #21262D;
    margin: 2rem 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #00D4AA, #00A884);
    color: #0D1117;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.stSlider > div { color: #C9D1D9; }
[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #21262D; }
[data-testid="stSidebar"] * { color: #C9D1D9; }
.stSelectbox label, .stSlider label, .stCheckbox label { color: #C9D1D9 !important; font-size: 0.85rem !important; }
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    pkg = joblib.load(model_path)
    return pkg

MODEL_FILE = "hog_svm_human_model_hnm.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def extract_hog_features(image, hog_params, window_size):
    resized = cv2.resize(image, window_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY) if len(resized.shape) == 3 else resized
    return hog(gray, **hog_params)


def sliding_window(image, step_size, window_size):
    win_w, win_h = window_size
    h, w = image.shape[:2]
    for y in range(0, h - win_h + 1, step_size):
        for x in range(0, w - win_w + 1, step_size):
            yield (x, y, image[y:y + win_h, x:x + win_w])


def detect_humans_multiscale(image, model, window_size, hog_params,
                              step_size=10, threshold=1.1,
                              downscale=1.25, max_levels=None):
    detections = []
    has_df = hasattr(model, "decision_function")
    for level, resized in enumerate(pyramid_gaussian(image, downscale=downscale, channel_axis=-1)):
        if max_levels and level >= max_levels:
            break
        if resized.shape[0] < window_size[1] or resized.shape[1] < window_size[0]:
            break
        resized_u8 = (resized * 255).astype(np.uint8) if resized.max() <= 1.0 else resized.astype(np.uint8)
        for (x, y, window) in sliding_window(resized_u8, step_size, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
            feats = extract_hog_features(window, hog_params, window_size).reshape(1, -1)
            pred = model.predict(feats)[0]
            if pred == 1:
                score = float(model.decision_function(feats)[0]) if has_df else 1.0
                if score >= threshold:
                    sf = downscale ** level
                    detections.append((int(x * sf), int(y * sf), score,
                                       int(window_size[0] * sf), int(window_size[1] * sf)))
    return detections


def nms(detections, iou_threshold=0.35):
    if not detections:
        return []
    rects  = np.array([[x, y, x+w, y+h] for (x,y,_,w,h) in detections], dtype=float)
    scores = np.array([s for (_,_,s,_,_) in detections], dtype=float)
    x1,y1,x2,y2 = rects[:,0],rects[:,1],rects[:,2],rects[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[np.where(iou<=iou_threshold)[0]+1]
    return [(int(rects[i][0]),int(rects[i][1]),int(rects[i][2]),int(rects[i][3]),float(scores[i])) for i in keep]


def make_hog_visualization(image_gray, hog_params):
    """Return HOG image rescaled for display."""
    params = {k: v for k, v in hog_params.items() if k != "feature_vector"}
    _, hog_img = hog(image_gray, visualize=True, feature_vector=True, **params)
    return exposure.rescale_intensity(hog_img, in_range=(0, 0.02))


def draw_boxes_on_image(image, boxes_scores, color=(0, 212, 170), thickness=2, show_score=True):
    img = image.copy()
    for item in boxes_scores:
        x1, y1, x2, y2, score = item
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if show_score:
            label = f"{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (13, 17, 23), 1, cv2.LINE_AA)
    return img


def draw_raw_detections(image, detections):
    img = image.copy()
    for (x1, y1, score, bw, bh) in detections:
        cv2.rectangle(img, (x1, y1), (x1+bw, y1+bh), (245, 166, 35), 1)
    return img


def make_pyramid_visualization(image, downscale=1.25):
    """Visualize image pyramid levels side by side."""
    levels = []
    for level, resized in enumerate(pyramid_gaussian(image, downscale=downscale, channel_axis=-1)):
        if level >= 5:
            break
        if resized.shape[0] < 64 or resized.shape[1] < 64:
            break
        img_u8 = (resized * 255).astype(np.uint8) if resized.max() <= 1.0 else resized.astype(np.uint8)
        levels.append((level, img_u8))
    return levels


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.75rem 0 1.25rem'>
        <p style='font-family:JetBrains Mono,monospace;font-size:1rem;
                  font-weight:600;color:#00D4AA;margin:0'>⚡ HOG+SVM</p>
        <p style='font-size:0.75rem;color:#7D8590;margin:4px 0 0'>Human Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    # Model path
    st.markdown('<p style="font-size:0.8rem;color:#7D8590;margin-bottom:4px">Model File (.pkl)</p>', unsafe_allow_html=True)
    model_path_input = st.text_input("", value=MODEL_FILE, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#21262D;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#7D8590;text-transform:uppercase;letter-spacing:1px">Detection Parameters</p>', unsafe_allow_html=True)

    detection_threshold = st.slider("Confidence Threshold", 0.5, 2.5, 1.1, 0.1,
        help="Semakin tinggi → semakin sedikit false positive")
    nms_threshold = st.slider("NMS IoU Threshold", 0.1, 0.6, 0.35, 0.05,
        help="Semakin rendah → box yang overlap lebih banyak dibuang")
    step_size = st.slider("Sliding Window Step", 4, 32, 10, 2,
        help="Lebih kecil = lebih akurat tapi lebih lambat")
    downscale = st.slider("Pyramid Downscale", 1.1, 1.8, 1.25, 0.05,
        help="Lebih kecil = lebih banyak level pyramid")
    max_width = st.slider("Max Image Width (px)", 400, 1000, 700, 50,
        help="Gambar akan di-resize ke lebar ini sebelum deteksi")

    st.markdown("<hr style='border-color:#21262D;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#7D8590;text-transform:uppercase;letter-spacing:1px">Visualization Options</p>', unsafe_allow_html=True)
    show_preprocessing  = st.checkbox("Preprocessing", value=True)
    show_hog_viz        = st.checkbox("HOG Visualization", value=True)
    show_pyramid        = st.checkbox("Image Pyramid", value=True)
    show_raw_detections = st.checkbox("Raw Detections (before NMS)", value=True)

    st.markdown("<hr style='border-color:#21262D;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#7D8590;line-height:1.7'>
        <b style='color:#C9D1D9'>Pengolahan & Analisis Citra Digital</b><br>
        HOG + LinearSVC + Hard Negative Mining<br>
        Dataset: INRIA Person Dataset
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">Object Detection · HOG/SVM 318</div>
    <h1 class="hero-title">Human <span>Detection</span> Pipeline</h1>
    <p class="hero-subtitle">
        Upload gambar JPG / PNG / JPEG → visualisasi lengkap setiap tahap pipeline →
        hasil deteksi dengan bounding box
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INDICATOR
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_html(active_step=0):
    steps = [
        ("01", "Upload\nGambar"),
        ("02", "Preprocessing\n& Resize"),
        ("03", "HOG\nFeatures"),
        ("04", "Image\nPyramid"),
        ("05", "Sliding\nWindow"),
        ("06", "NMS &\nHasil"),
    ]
    html = '<div class="pipeline-container">'
    for i, (num, label) in enumerate(steps):
        if i < active_step:
            cls = "done"
            icon = "✓"
        elif i == active_step:
            cls = "active"
            icon = num
        else:
            cls = ""
            icon = num
        conn_cls = "done" if i < active_step else ""
        if i > 0:
            html += f'<div class="step-connector {conn_cls}"></div>'
        html += f'''
        <div class="pipeline-step">
            <div class="step-circle {cls}">{icon}</div>
            <div class="step-label">{label.replace(chr(10),"<br>")}</div>
        </div>'''
    html += '</div>'
    return html

pipeline_placeholder = st.empty()
pipeline_placeholder.markdown(pipeline_html(0), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload gambar untuk dideteksi",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-hint">
        <div style="font-size:2.5rem;margin-bottom:0.5rem">🖼️</div>
        <p style="font-size:0.95rem;color:#C9D1D9;margin:0 0 4px">
            Drag & drop atau klik untuk upload gambar
        </p>
        <p style="font-size:0.8rem;color:#7D8590;margin:0">
            Format yang didukung: JPG, JPEG, PNG
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
if not Path(model_path_input).exists():
    st.error(f"⚠️ Model tidak ditemukan: `{model_path_input}`  \nPastikan file `.pkl` ada di direktori yang sama dengan `app.py`.")
    st.stop()

pkg         = load_model(model_path_input)
model       = pkg["model"]
WINDOW_SIZE = tuple(pkg["window_size"])
HOG_PARAMS  = pkg["hog_params"]
rec_params  = pkg.get("recommended_detection_params", {})
training_info = pkg.get("training_info", {})


# ─────────────────────────────────────────────────────────────────────────────
# DECODE IMAGE
# ─────────────────────────────────────────────────────────────────────────────
file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
orig_h, orig_w = img_rgb.shape[:2]

pipeline_placeholder.markdown(pipeline_html(1), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: ORIGINAL IMAGE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="viz-card">
    <div class="viz-card-header">
        <span class="step-tag">STEP 01</span>
        <div>
            <p class="viz-card-title">Gambar Original</p>
            <p class="viz-card-desc">Gambar asli yang di-upload sebelum diproses apapun</p>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.image(img_rgb, caption=f"Original — {uploaded_file.name}", use_container_width=True)
with col2:
    st.markdown(f"""
    <div class="metric-row" style="flex-direction:column">
        <div class="metric-box">
            <div class="val">{orig_w}×{orig_h}</div>
            <div class="lbl">Resolusi (px)</div>
        </div>
        <div class="metric-box amber">
            <div class="val">{orig_w * orig_h // 1000}K</div>
            <div class="lbl">Total Piksel</div>
        </div>
        <div class="metric-box blue">
            <div class="val">{img_rgb.shape[2] if len(img_rgb.shape)==3 else 1}</div>
            <div class="lbl">Channel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
if max_width and orig_w > max_width:
    scale_r    = max_width / orig_w
    proc_w     = int(orig_w * scale_r)
    proc_h     = int(orig_h * scale_r)
    img_resized = cv2.resize(img_rgb, (proc_w, proc_h))
else:
    img_resized = img_rgb.copy()
    proc_w, proc_h = orig_w, orig_h

img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

pipeline_placeholder.markdown(pipeline_html(2), unsafe_allow_html=True)

if show_preprocessing:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-card-header">
            <span class="step-tag">STEP 02</span>
            <div>
                <p class="viz-card-title">Preprocessing</p>
                <p class="viz-card-desc">
                    Gambar di-resize agar deteksi tidak terlalu lambat,
                    lalu dikonversi ke grayscale untuk ekstraksi HOG
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_resized, caption=f"Resized — {proc_w}×{proc_h}px", use_container_width=True)
    with col2:
        st.image(img_gray, caption="Grayscale", use_container_width=True, clamp=True)
    with col3:
        # Histogram
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#161B22")
        ax.set_facecolor("#0D1117")
        ax.hist(img_gray.ravel(), bins=64, color="#00D4AA", alpha=0.8, edgecolor="none")
        ax.set_title("Pixel Intensity Distribution", color="#C9D1D9", fontsize=9)
        ax.set_xlabel("Intensity", color="#7D8590", fontsize=8)
        ax.set_ylabel("Count", color="#7D8590", fontsize=8)
        ax.tick_params(colors="#7D8590", labelsize=7)
        for spine in ax.spines.values(): spine.set_color("#21262D")
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown(f"""
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:0.5rem'>
        <span class='conf-badge'>Original: {orig_w}×{orig_h}</span>
        <span style='color:#7D8590;font-size:0.8rem;align-self:center'>→</span>
        <span class='conf-badge high'>Resized: {proc_w}×{proc_h}</span>
        <span style='color:#7D8590;font-size:0.8rem;align-self:center'>→</span>
        <span class='conf-badge med'>Grayscale 1ch</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: HOG VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
pipeline_placeholder.markdown(pipeline_html(3), unsafe_allow_html=True)

if show_hog_viz:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-card-header">
            <span class="step-tag">STEP 03</span>
            <div>
                <p class="viz-card-title">HOG Feature Visualization</p>
                <p class="viz-card-desc">
                    Histogram of Oriented Gradients — setiap cell menampilkan arah dan kekuatan edge.
                    Ini adalah "fingerprint" yang digunakan SVM untuk mengenali manusia.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    hog_viz_params = {k: v for k, v in HOG_PARAMS.items() if k != "feature_vector"}
    hog_image = make_hog_visualization(img_gray, hog_viz_params)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#161B22")
        ax.set_facecolor("#0D1117")
        ax.imshow(img_gray, cmap="gray")
        ax.set_title("Input Grayscale", color="#C9D1D9", fontsize=10, pad=10)
        ax.axis("off")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#161B22")
        ax.set_facecolor("#0D1117")
        im = ax.imshow(hog_image, cmap="inferno")
        ax.set_title("HOG Descriptor Visualization", color="#C9D1D9", fontsize=10, pad=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(colors="#7D8590", labelsize=7)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # HOG parameter info
    st.markdown(f"""
    <div style='background:#0D1117;border:1px solid #21262D;border-radius:8px;padding:0.75rem 1rem;margin-top:0.5rem'>
        <div style='display:flex;gap:6px;flex-wrap:wrap'>
            <span class='conf-badge'>orientations={HOG_PARAMS.get('orientations',9)}</span>
            <span class='conf-badge'>pixels_per_cell={HOG_PARAMS.get('pixels_per_cell',(8,8))}</span>
            <span class='conf-badge'>cells_per_block={HOG_PARAMS.get('cells_per_block',(2,2))}</span>
            <span class='conf-badge med'>block_norm={HOG_PARAMS.get('block_norm','L2-Hys')}</span>
            <span class='conf-badge med'>transform_sqrt={HOG_PARAMS.get('transform_sqrt',True)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: IMAGE PYRAMID
# ─────────────────────────────────────────────────────────────────────────────
pipeline_placeholder.markdown(pipeline_html(3), unsafe_allow_html=True)

if show_pyramid:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-card-header">
            <span class="step-tag">STEP 04</span>
            <div>
                <p class="viz-card-title">Image Pyramid (Gaussian)</p>
                <p class="viz-card-desc">
                    Gambar di-downsample bertahap dengan Gaussian smoothing.
                    Sliding window dijalankan di setiap level agar orang besar maupun kecil
                    bisa terdeteksi dengan window 64×128 yang tetap.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    pyramid_levels = make_pyramid_visualization(img_resized, downscale=downscale)
    n_levels = len(pyramid_levels)
    cols = st.columns(min(n_levels, 5))

    for col, (level, lvl_img) in zip(cols, pyramid_levels):
        lh, lw = lvl_img.shape[:2]
        with col:
            st.image(lvl_img, caption=f"Level {level}\n{lw}×{lh}px", use_container_width=True)
            ratio = round((1 / (downscale ** level)) * 100, 1)
            st.markdown(f'<p style="text-align:center;font-family:JetBrains Mono,monospace;font-size:0.7rem;color:#00D4AA">{ratio}% skala</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:0.5rem;font-size:0.82rem;color:#7D8590'>
        🔢 Total <b style='color:#C9D1D9'>{n_levels} level</b> pyramid dengan downscale factor
        <b style='color:#00D4AA'>{downscale}×</b> per level
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: DETECTION
# ─────────────────────────────────────────────────────────────────────────────
pipeline_placeholder.markdown(pipeline_html(4), unsafe_allow_html=True)

with st.spinner("🔍 Menjalankan sliding window + SVM classifier..."):
    t_start = time.time()
    detections_raw = detect_humans_multiscale(
        image      = img_resized,
        model      = model,
        window_size= WINDOW_SIZE,
        hog_params = HOG_PARAMS,
        step_size  = step_size,
        threshold  = detection_threshold,
        downscale  = downscale,
    )
    t_detect = time.time() - t_start

    t_nms_start = time.time()
    final_boxes = nms(detections_raw, iou_threshold=nms_threshold)
    t_nms = time.time() - t_nms_start

pipeline_placeholder.markdown(pipeline_html(5), unsafe_allow_html=True)

# Raw detections visualization
if show_raw_detections:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-card-header">
            <span class="step-tag">STEP 05</span>
            <div>
                <p class="viz-card-title">Raw Detections (Sebelum NMS)</p>
                <p class="viz-card-desc">
                    Semua window yang diprediksi SVM sebagai "manusia" di semua level pyramid.
                    Banyak box yang tumpuk karena setiap posisi sliding window diproses independen.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    raw_img = draw_raw_detections(img_resized, detections_raw)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(raw_img, caption=f"Raw detections — {len(detections_raw)} boxes", use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="metric-row" style="flex-direction:column">
            <div class="metric-box amber">
                <div class="val">{len(detections_raw)}</div>
                <div class="lbl">Raw boxes</div>
            </div>
            <div class="metric-box">
                <div class="val">{t_detect:.2f}s</div>
                <div class="lbl">Detection time</div>
            </div>
        </div>
        <p style="font-size:0.78rem;color:#7D8590;margin-top:0.5rem">
            Box kuning = semua kandidat sebelum NMS.
            Setiap orang biasanya menghasilkan banyak box yang overlap.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: FINAL RESULT (NMS)
# ─────────────────────────────────────────────────────────────────────────────
pipeline_placeholder.markdown(pipeline_html(6), unsafe_allow_html=True)

st.markdown("""
<div class="viz-card">
    <div class="viz-card-header">
        <span class="step-tag">STEP 06</span>
        <div>
            <p class="viz-card-title">Hasil Deteksi Final (Setelah NMS)</p>
            <p class="viz-card-desc">
                Non-Maximum Suppression membuang box yang redundant berdasarkan IoU.
                Hanya box dengan confidence tertinggi yang dipertahankan per orang.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

n_detected = len(final_boxes)

# Result banner
if n_detected > 0:
    st.markdown(f"""
    <div class="result-banner">
        <h3>🧍 {n_detected} Manusia Terdeteksi</h3>
        <p>Threshold: {detection_threshold} · NMS IoU: {nms_threshold} · Step: {step_size}px</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="result-banner no-detect">
        <h3>❌ Tidak Ada Manusia Terdeteksi</h3>
        <p>Coba turunkan Confidence Threshold di sidebar, atau pastikan ada orang di gambar.</p>
    </div>
    """, unsafe_allow_html=True)

col_main, col_side = st.columns([3, 1])

with col_main:
    result_img = draw_boxes_on_image(img_resized, final_boxes, show_score=True)
    st.image(result_img, caption="Final Detection Result", use_container_width=True)

with col_side:
    st.markdown(f"""
    <div class="metric-row" style="flex-direction:column">
        <div class="metric-box">
            <div class="val">{n_detected}</div>
            <div class="lbl">Manusia terdeteksi</div>
        </div>
        <div class="metric-box amber">
            <div class="val">{len(detections_raw)}</div>
            <div class="lbl">Raw → NMS</div>
        </div>
        <div class="metric-box blue">
            <div class="val">{(t_detect + t_nms):.2f}s</div>
            <div class="lbl">Total waktu</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if final_boxes:
        st.markdown('<p style="font-size:0.78rem;color:#7D8590;margin:0.5rem 0 0.3rem">Confidence scores:</p>', unsafe_allow_html=True)
        badges = ""
        for i, (_, _, _, _, score) in enumerate(final_boxes):
            cls = "high" if score >= 1.5 else "med" if score >= 1.0 else "low"
            badges += f'<span class="conf-badge {cls}">#{i+1} {score:.2f}</span>'
        st.markdown(badges, unsafe_allow_html=True)

# Side-by-side comparison
if n_detected > 0:
    st.markdown("<hr style='border-color:#21262D;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem;color:#7D8590;margin-bottom:0.75rem">📊 Perbandingan Before vs After NMS</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        raw_small = draw_raw_detections(img_resized, detections_raw)
        st.image(raw_small, caption=f"Before NMS — {len(detections_raw)} boxes", use_container_width=True)
    with c2:
        st.image(result_img, caption=f"After NMS — {n_detected} boxes", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
if final_boxes:
    st.markdown("""
    <div class="viz-card">
        <div class="viz-card-header">
            <span class="step-tag">SUMMARY</span>
            <div>
                <p class="viz-card-title">Detection Details</p>
                <p class="viz-card-desc">Koordinat dan confidence score setiap manusia yang terdeteksi</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    import pandas as pd
    rows = []
    for i, (x1, y1, x2, y2, score) in enumerate(final_boxes):
        rows.append({
            "ID"          : f"Person #{i+1}",
            "X1"          : x1,
            "Y1"          : y1,
            "X2"          : x2,
            "Y2"          : y2,
            "Width (px)"  : x2 - x1,
            "Height (px)" : y2 - y1,
            "Confidence"  : round(score, 4),
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=3, format="%.4f"
            )
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📦 Model & Training Info", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="sidebar-section">
            <p class="sidebar-title">Model Package</p>
            <div class="param-row">Window Size <span>{WINDOW_SIZE}</span></div>
            <div class="param-row">Method <span>HOG + LinearSVC</span></div>
            <div class="param-row">Hard Neg Count <span>{training_info.get('hard_negative_count','—')}</span></div>
            <div class="param-row">Initial C <span>{training_info.get('initial_C','—')}</span></div>
            <div class="param-row">Final C <span>{training_info.get('final_C','—')}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="sidebar-section">
            <p class="sidebar-title">Detection Session</p>
            <div class="param-row">Image (original) <span>{orig_w}×{orig_h}</span></div>
            <div class="param-row">Image (processed) <span>{proc_w}×{proc_h}</span></div>
            <div class="param-row">Threshold <span>{detection_threshold}</span></div>
            <div class="param-row">NMS IoU <span>{nms_threshold}</span></div>
            <div class="param-row">Step size <span>{step_size}px</span></div>
            <div class="param-row">Downscale <span>{downscale}×</span></div>
            <div class="param-row">Detect time <span>{t_detect:.3f}s</span></div>
            <div class="param-row">NMS time <span>{t_nms:.4f}s</span></div>
            <div class="param-row">Total time <span>{t_detect+t_nms:.3f}s</span></div>
        </div>
        """, unsafe_allow_html=True)