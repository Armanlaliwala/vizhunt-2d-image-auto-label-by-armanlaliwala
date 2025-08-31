# vizhunt_v2.py
# 🔍 Vizhunt — Auto-Label Images with AI
# Sections: Analysis → Preview → Exports (v2 update)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import json
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from collections import defaultdict

# -------------------------- Utility functions --------------------------

def load_model(model_path: str = r"vizhunt-model.pt") -> YOLO:
    """Load the YOLO model and cache it in Streamlit session state."""
    if "model" not in st.session_state:
        st.session_state.model = YOLO(model_path)
    return st.session_state.model


def read_image_bytes(file) -> (np.ndarray, Image.Image):
    """Return cv2 BGR image and PIL image from uploaded file-like object."""
    img_bytes = file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    cv_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv_img, pil


def xyxy_to_xywh_norm(x1, y1, x2, y2, width, height):
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return float(xc), float(yc), float(w), float(h)


def draw_detections(cv_img: np.ndarray, detections: list, class_colors: dict) -> Image.Image:
    """Draw only colored bounding boxes (no labels/confidences)."""
    img = cv_img.copy()
    for det in detections:
        cls = det["class"]
        x1, y1, x2, y2 = map(int, det["bbox_pixels"])
        color = class_colors.get(cls, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def make_class_colors(names):
    """Deterministic color for each class name."""
    colors = {}
    for i, n in names.items():
        h = abs(hash(n)) % (256**3)
        b = h & 255
        g = (h >> 8) & 255
        r = (h >> 16) & 255
        colors[n] = (int(b), int(g), int(r))
    return colors

# -------------------------- Processing pipeline --------------------------

def process_single_image(pil_image: Image.Image, img_cv_bgr: np.ndarray, model: YOLO, image_name: str) -> dict:
    """Run YOLO model on one image and return structured detections + summary."""
    res = model(pil_image, imgsz=1280)[0]

    names_map = model.names if hasattr(model, 'names') else {i: str(i) for i in range(1000)}
    class_colors = make_class_colors(names_map)

    detections = []
    per_class_counts = defaultdict(int)
    per_class_confs = defaultdict(list)

    width, height = pil_image.size
    boxes = getattr(res, 'boxes', None)

    if boxes is None or len(boxes) == 0:
        annotated = pil_image.copy()
        return {"detections": [], "annotated_pil": annotated, "summary": {}}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    class_instance_counters = defaultdict(int)

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        cls_name = names_map.get(cls_id, str(cls_id))
        class_instance_counters[cls_name] += 1
        inst_id = class_instance_counters[cls_name]
        per_class_counts[cls_name] += 1
        per_class_confs[cls_name].append(float(conf))

        bbox_px = [float(x1), float(y1), float(x2), float(y2)]
        xc, yc, w, h = xyxy_to_xywh_norm(x1, y1, x2, y2, width, height)

        detections.append({
            "image": image_name,
            "class": cls_name,
            "class_id": int(cls_id),
            "inst_id": int(inst_id),
            "conf": float(conf),
            "bbox_pixels": bbox_px,
            "bbox_norm": [xc, yc, w, h],
            "width_px": width,
            "height_px": height,
        })

    annotated = draw_detections(img_cv_bgr, detections, class_colors)

    summary = {}
    for c, count in per_class_counts.items():
        avg_conf = float(sum(per_class_confs[c]) / len(per_class_confs[c]))
        summary[c] = {"instances": int(count), "avg_conf": avg_conf}

    return {"detections": detections, "annotated_pil": annotated, "summary": summary}

# -------------------------- Exports --------------------------

def generate_yolo_txt(detections: list) -> str:
    lines = []
    for d in detections:
        cls_id = d['class_id']
        xc, yc, w, h = d['bbox_norm']
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def generate_json(image_name: str, detections: list) -> dict:
    return {
        "image": image_name,
        "detections": [
            {
                "class": d['class'],
                "class_id": d['class_id'],
                "inst_id": d['inst_id'],
                "conf": d['conf'],
                "bbox_pixels": d['bbox_pixels'],
                "bbox_norm": d['bbox_norm']
            }
            for d in detections
        ]
    }


def generate_csv_rows(detections: list) -> pd.DataFrame:
    rows = []
    for d in detections:
        xc, yc, w, h = d['bbox_norm']
        rows.append({
            'image': d['image'],
            'class': d['class'],
            'inst_id': d['inst_id'],
            'conf': d['conf'],
            'xc': xc,
            'yc': yc,
            'w': w,
            'h': h,
            'width_px': d['width_px'],
            'height_px': d['height_px']
        })
    return pd.DataFrame(rows)

# -------------------------- Streamlit UI --------------------------

st.set_page_config(page_title="Vizhunt", layout="wide")

st.title("🔍 Vizhunt – Auto-Label Images with AI")
st.subheader("Upload. Analyze. Export.")
st.markdown("---")

with st.sidebar:
    st.header("Upload & Model")
    uploaded_files = st.file_uploader(
        "Upload images (.jpg .jpeg .png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    st.caption("Model: vizhunt-model.pt — will be loaded once (make sure vizhunt-model.pt is present)")
    model_path = st.text_input("Model path", value="vizhunt-model.pt")
    if st.button("Load model"):
        try:
            with st.spinner("Loading model..."):
                model = load_model(model_path)
            st.success("Model loaded")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

if uploaded_files:
    st.sidebar.markdown("**Files queued:**")
    for f in uploaded_files:
        st.sidebar.write(f.name)

if 'results' not in st.session_state:
    st.session_state['results'] = {}

process_now = st.button("Run Auto-Labeling")

if process_now and uploaded_files:
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    progress_bar = st.progress(0)
    all_csv_rows = []

    for idx, file in enumerate(uploaded_files, start=1):
        file.seek(0)
        cv_img, pil = read_image_bytes(file)
        image_name = os.path.splitext(file.name)[0]
        with st.spinner(f"Processing {file.name} ({idx}/{len(uploaded_files)})"):
            try:
                res = process_single_image(pil, cv_img, model, image_name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                res = {"detections": [], "annotated_pil": pil, "summary": {}}

        st.session_state.results[file.name] = res
        df = generate_csv_rows(res['detections'])
        if not df.empty:
            all_csv_rows.append(df)

        progress_bar.progress(int(idx / len(uploaded_files) * 100))

    st.session_state['combined_csv'] = pd.concat(all_csv_rows, ignore_index=True) if all_csv_rows else pd.DataFrame()
    st.success("Auto-labeling completed")

# ---------------- Results Display ----------------
if st.session_state['results']:
    # ---------------- Analysis ----------------
    st.header("📊 Analysis (All Images)")
    combined_summary = defaultdict(lambda: {"instances": 0, "conf_list": []})
    total_labels = 0

    for res in st.session_state['results'].values():
        for cls, data in res['summary'].items():
            combined_summary[cls]["instances"] += data["instances"]
            combined_summary[cls]["conf_list"].append(data["avg_conf"])
            total_labels += data["instances"]

    if combined_summary:
        classes = list(combined_summary.keys())
        counts = [combined_summary[c]["instances"] for c in classes]

        fig, ax = plt.subplots(figsize=(6, 3))  # smaller chart
        ax.bar(classes, counts)
        ax.set_ylabel('Instances')
        ax.set_title('Class distribution (all images)')
        st.pyplot(fig)

        df_summary = pd.DataFrame(
            [{'class': c,
              'instances': combined_summary[c]["instances"],
              'avg_conf': sum(combined_summary[c]["conf_list"]) / len(combined_summary[c]["conf_list"])}
             for c in classes])

        st.dataframe(df_summary)

        st.markdown(f"**Total labels detected across all images: {total_labels}**")
    else:
        st.info("No detections found.")

    st.markdown("---")

    # ---------------- Preview ----------------
    st.header("🖼️ Preview (All Images)")
    result_items = list(st.session_state['results'].items())
    cols = st.columns(2)
    for idx, (fname, res) in enumerate(result_items):
        with cols[idx % 2]:
            st.subheader(fname)
            st.image(res['annotated_pil'], use_column_width=True)

    st.markdown("---")

    # ---------------- Exports ----------------
    st.header("📂 Exports")
    export_format = st.selectbox(
        "Choose export format",
        options=[".zip (combine)", ".json", ".csv", ".txt"],
        index=0
    )

    if export_format == ".zip (combine)":
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, res in st.session_state['results'].items():
                name_no_ext = os.path.splitext(fname)[0]
                yolo_txt = generate_yolo_txt(res['detections'])
                zf.writestr(f"labels/{name_no_ext}.txt", yolo_txt)
                js = generate_json(name_no_ext, res['detections'])
                zf.writestr(f"json/{name_no_ext}.json", json.dumps(js, indent=2))
            if not st.session_state['combined_csv'].empty:
                csv_bytes = st.session_state['combined_csv'].to_csv(index=False).encode('utf-8')
                zf.writestr('combined.csv', csv_bytes)
        zbuf.seek(0)
        st.download_button(
            "Download ZIP (labels + json + combined.csv)",
            data=zbuf.getvalue(),
            file_name="vizhunt_labels.zip",
            mime="application/zip"
        )

    elif export_format == ".csv":
        if not st.session_state['combined_csv'].empty:
            csv_bytes = st.session_state['combined_csv'].to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download CSV (combined)",
                data=csv_bytes,
                file_name="vizhunt_combined.csv",
                mime='text/csv'
            )
        else:
            st.info("No CSV data available.")

    elif export_format == ".json":
        for fname, res in st.session_state['results'].items():
            js = generate_json(os.path.splitext(fname)[0], res['detections'])
            st.download_button(
                label=f"Download {fname} JSON",
                data=json.dumps(js, indent=2),
                file_name=f"{os.path.splitext(fname)[0]}.json",
                mime='application/json',
                key=f"json_{fname}"
            )

    elif export_format == ".txt":
        for fname, res in st.session_state['results'].items():
            yolo_txt = generate_yolo_txt(res['detections'])
            st.download_button(
                label=f"Download {fname} TXT",
                data=yolo_txt,
                file_name=f"{os.path.splitext(fname)[0]}.txt",
                mime='text/plain',
                key=f"txt_{fname}"
            )

else:
    st.info("Upload images and click 'Run Auto-Labeling' to start.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("**Dependencies**: streamlit, ultralytics, opencv-python, pillow, matplotlib, pandas")
st.sidebar.write("Install with: `pip install streamlit ultralytics opencv-python pillow matplotlib pandas`")
