"""
Streamlit demo — YOLOv8 MedVision
Chest X-ray pathology detection + optional MedGemma-4B clinical report.

Launch:
    streamlit run app.py
"""

import gc
import io
import os
import tempfile

import streamlit as st

# ── page config (must come before any other st call) ──────────────────────────
st.set_page_config(
    page_title="YOLOv8 MedVision",
    page_icon="🫁",
    layout="wide",
)

# ── optional heavy imports — UI loads regardless, error shown on inference ─────
try:
    import cv2
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    _DEPS_OK = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError as _e:
    _DEPS_OK = False
    GPU_AVAILABLE = False

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Aortic_enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung_Opacity", "Nodule/Mass", "Other_lesion",
    "Pleural_effusion", "Pleural_thickening", "Pneumothorax",
    "Pulmonary_fibrosis", "No finding",
]
CONF_COLORS = {"HAUTE": "#00C853", "MOYENNE": "#FFD600", "FAIBLE": "#FF1744"}


def _conf_level(conf: float) -> tuple[str, str]:
    if conf >= 0.70:
        return "HAUTE", "🟢"
    elif conf >= 0.45:
        return "MOYENNE", "🟡"
    else:
        return "FAIBLE", "🔴"


# ── YOLO loader (cached across reruns) ────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Model")
    model_file = st.file_uploader("Upload best.pt", type=["pt"])
    model_path_text = st.text_input("…or enter local path to best.pt")

    st.subheader("Detection")
    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.10, max_value=0.90,
        value=0.25, step=0.05,
    )

    st.subheader("LLM Report")
    use_medgemma = st.checkbox("🧠 Generate MedGemma report", value=False)

    hf_token_input = ""
    if use_medgemma:
        hf_token_input = st.text_input(
            "HuggingFace token (HF_TOKEN)",
            type="password",
            help="Required — get your token at huggingface.co/settings/tokens",
        )
        st.caption("Required — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)")

    st.info(
        "⚡ YOLO requires ~2 GB VRAM. "
        "MedGemma-4B requires ~6 GB VRAM (loaded sequentially)."
    )

    if not GPU_AVAILABLE:
        st.warning("⚠️ No GPU detected. YOLO will run on CPU (slower). MedGemma disabled.")
        use_medgemma = False


# ── main header ────────────────────────────────────────────────────────────────
st.title("🫁 YOLOv8 MedVision")
st.markdown(
    "![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) "
    "![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF) "
    "![HuggingFace](https://img.shields.io/badge/HuggingFace-MedGemma--4B-FFD700) "
    "![License](https://img.shields.io/badge/License-MIT-green)"
)
st.markdown(
    "> Automated pipeline: YOLOv8m detects pulmonary pathologies on chest X-rays, "
    "MedGemma-4B generates a structured clinical report."
)
st.divider()

# ── resolve model path ─────────────────────────────────────────────────────────
resolved_model_path = None
_tmp_model_file = None

if model_file is not None:
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    _tmp.write(model_file.read())
    _tmp.close()
    resolved_model_path = _tmp.name
elif model_path_text.strip():
    resolved_model_path = model_path_text.strip()

if resolved_model_path is None:
    st.info("👆 Please upload or specify a YOLO model in the sidebar to get started.")
    st.stop()

if not _DEPS_OK:
    st.error(
        "**Missing ML dependencies.** Run the following then restart Streamlit:\n\n"
        "```bash\npip install -r requirements.txt\n```\n\n"
        "PyTorch, OpenCV, Pillow and Matplotlib are required for inference."
    )
    st.stop()

# ── load model ─────────────────────────────────────────────────────────────────
try:
    yolo_model = load_yolo(resolved_model_path)
except Exception as e:
    st.error(f"Failed to load YOLO model: {e}")
    st.stop()

# ── image upload ───────────────────────────────────────────────────────────────
uploaded_image = st.file_uploader(
    "Upload a chest X-ray", type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_image is None:
    st.stop()

# ── run YOLO ───────────────────────────────────────────────────────────────────
img_bytes = uploaded_image.read()
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
img_np = np.array(pil_img)

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
    pil_img.save(tmp_img.name)
    tmp_img_path = tmp_img.name

device = 0 if GPU_AVAILABLE else "cpu"

with st.spinner("Running YOLOv8 detection…"):
    results = yolo_model.predict(
        source=tmp_img_path, conf=conf_threshold,
        imgsz=640, device=device, verbose=False,
    )

detections = []
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        level, emoji = _conf_level(conf)
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": conf,
            "conf_level": level,
            "emoji": emoji,
            "bbox": [x1, y1, x2, y2],
        })
detections.sort(key=lambda d: d["confidence"], reverse=True)

# ── draw annotated image ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_np)
for det in detections:
    if det["class_name"] == "No finding":
        continue
    x1, y1, x2, y2 = det["bbox"]
    color = CONF_COLORS[det["conf_level"]]
    ax.add_patch(
        patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
    )
    ax.text(
        x1, y1 - 5,
        f"{det['emoji']} {det['class_name']} {det['confidence']:.0%}",
        color="white", fontsize=8, fontweight="bold",
        bbox=dict(facecolor=color, alpha=0.85, pad=2, edgecolor="none"),
    )
ax.axis("off")
ax.set_title("YOLOv8m Detections", fontsize=12, fontweight="bold")
plt.tight_layout()

annotated_buf = io.BytesIO()
fig.savefig(annotated_buf, format="png", dpi=120, bbox_inches="tight")
plt.close(fig)
annotated_buf.seek(0)

# ── display side by side ───────────────────────────────────────────────────────
col_orig, col_pred = st.columns(2)
with col_orig:
    st.image(pil_img, caption="Original X-ray", use_container_width=True)
with col_pred:
    st.image(annotated_buf, caption="YOLOv8m Detections", use_container_width=True)

# ── detections table ───────────────────────────────────────────────────────────
st.subheader(f"Detections — {len(detections)} finding(s)")

if detections:
    import pandas as pd
    df = pd.DataFrame([
        {
            "Pathology":   d["class_name"],
            "Confidence":  f"{d['confidence']:.1%}",
            "Level":       d["conf_level"],
            "Emoji":       d["emoji"],
        }
        for d in detections
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No pathologies detected above the confidence threshold.")

# ── MedGemma report ────────────────────────────────────────────────────────────
if use_medgemma:
    hf_token = hf_token_input.strip() or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        st.error("HF_TOKEN required to load MedGemma. Enter it in the sidebar.")
        st.stop()

    st.subheader("## Clinical Report")

    with st.spinner("Releasing YOLO from GPU memory…"):
        del yolo_model
        gc.collect()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    with st.spinner("Loading MedGemma-4B (4-bit quantized)…"):
        from huggingface_hub import login
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        login(token=hf_token)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it")
        llm_model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-4b-it",
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        llm_model.eval()

    with st.spinner("Generating clinical report…"):
        from src.utils import build_llm_prompt
        prompt = build_llm_prompt(detections)

        annotated_buf.seek(0)
        pil_annotated = Image.open(annotated_buf).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_annotated},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = llm_model.generate(
                **inputs, max_new_tokens=600, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        report_text = processor.tokenizer.decode(
            output[0][input_len:], skip_special_tokens=True
        ).strip()

    st.text_area("Clinical Report", value=report_text, height=400, disabled=True)
    st.download_button(
        label="⬇️ Download report (.txt)",
        data=report_text.encode("utf-8"),
        file_name="radiology_report.txt",
        mime="text/plain",
    )

# ── footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ For research purposes only. "
    "Not a substitute for professional medical diagnosis."
)
