# Running the Demo

## Prerequisites

- Python 3.10+
- CUDA GPU recommended (CPU works but is significantly slower)
- YOLO weights: download from [Releases](https://github.com/Ksyl25/yolov8-medvision/releases) or train from the notebook

## Launch

```bash
pip install -r requirements.txt
streamlit run app.py
```

The demo opens at `http://localhost:8501` in your browser.

## Usage

1. **Upload `best.pt`** in the sidebar (or enter the local path to your weights file)
2. **Upload a chest X-ray** (JPG or PNG) in the main area
3. **Adjust the confidence threshold** if needed (default: 0.25)
4. **Optionally enable MedGemma report** — requires:
   - A HuggingFace token with access to `google/medgemma-4b-it`
   - ~6 GB VRAM (model loaded in 4-bit quantization)
   - YOLO is automatically released from GPU before MedGemma loads

## Environment Variables

Copy `.env.example` to `.env` and set your HuggingFace token:

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_your_token_here
```

The app reads `HF_TOKEN` from the environment **or** from the sidebar input field.

## GPU Memory Notes

| Component       | VRAM required | Notes                                  |
|-----------------|---------------|----------------------------------------|
| YOLOv8m         | ~2 GB         | Released before LLM loads             |
| MedGemma-4B 4-bit | ~6 GB       | bfloat16, NF4 quantization (bitsandbytes) |
| Both together   | not supported | Sequential loading required on T4 GPU  |
