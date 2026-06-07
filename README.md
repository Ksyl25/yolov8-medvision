# YOLOv8 MedVision — Chest X-ray Pathology Detection & Reporting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?logo=github)
![HuggingFace](https://img.shields.io/badge/HuggingFace-MedGemma--4B-FFD700?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

> **Automated pipeline that detects pulmonary pathologies on chest radiographs with YOLOv8 and generates a structured clinical report with MedGemma-4B.**

---

## Pipeline Overview

```
Chest X-ray (DICOM/JPG)
        │
        ▼
┌──────────────────┐
│  YOLOv8m         │  ← detects & localises anomalies (15 classes)
│  best.pt         │
└────────┬─────────┘
         │  bounding boxes + confidence scores
         ▼
┌──────────────────┐
│  Prompt builder  │  ← structures detections by confidence level
└────────┬─────────┘
         │  structured clinical prompt
         ▼
┌──────────────────┐
│  MedGemma-4B-IT  │  ← generates radiology report (4-bit quantized)
│  (Google)        │
└────────┬─────────┘
         │
         ▼
  Structured radiology report
  (summary · findings · recommendations · disclaimer)
```

---

## Dataset — VinBigData

- **15 000** chest X-rays annotated by professional radiologists
- **15 pathology classes** in YOLO format with Weighted Box Fusion (WBF)
- Source: [`buithanhxuan/vinbigdata-yolo-dataset-with-wbf-3x-downscaled`](https://www.kaggle.com/datasets/buithanhxuan/vinbigdata-yolo-dataset-with-wbf-3x-downscaled) (Kaggle)

| Class ID | Pathology              | Class ID | Pathology            |
|:--------:|------------------------|:--------:|----------------------|
| 0        | Aortic enlargement     | 8        | Nodule/Mass          |
| 1        | Atelectasis            | 9        | Other lesion         |
| 2        | Calcification          | 10       | Pleural effusion     |
| 3        | Cardiomegaly           | 11       | Pleural thickening   |
| 4        | Consolidation          | 12       | Pneumothorax         |
| 5        | ILD                    | 13       | Pulmonary fibrosis   |
| 6        | Infiltration           | 14       | No finding           |
| 7        | Lung Opacity           |          |                      |

Training subsets were built with a custom balanced-sampling strategy (class caps + rare-class prioritisation) to mitigate the severe class imbalance in the original distribution.

---

## Training Results

Three progressive experiments were run on Google Colab (Tesla T4 GPU):

| Version | Architecture | Epochs | Training Set | mAP@50 | mAP@50-95 |
|---------|-------------|--------|--------------|--------|-----------|
| **V1s** | YOLOv8s     | 10     | 5 000 images | 0.251  | 0.161     |
| **V1m** | YOLOv8m     | 50     | 5 000 images | 0.338  | 0.197     |
| **V2**  | YOLOv8m     | 50     | 7 500 images | **0.390**  | **0.223** |

Key takeaways:
- Upgrading from YOLOv8**s** → YOLOv8**m** brought +8.7 mAP50 points (+35%)
- Extending the training set from 5k → 7.5k images brought another +5.2 points (+15%)
- V2 was fine-tuned from V1m weights (transfer learning), reducing training time

Training curves and confusion matrices are available in [`results/`](results/).

---

## Project Structure

```
yolov8-medvision/
├── notebooks/
│   └── MEDICALCAPTIONING_CLEAN.ipynb   # Full training & inference notebook
├── src/
│   ├── predict.py                       # Standalone inference script
│   ├── dataset.py                       # Subset creation utilities
│   └── utils.py                         # draw_boxes, build_llm_prompt, helpers
├── results/
│   ├── v1s/                             # YOLOv8s 10ep curves & samples
│   ├── v1m/                             # YOLOv8m 50ep 5k curves & samples
│   └── v2/                              # YOLOv8m 50ep 7.5k curves & samples
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** Model weights (`*.pt`) are excluded from the repository due to size.  
> Download them from the [Releases](#) tab or train from the notebook.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/yolov8-medvision.git
cd yolov8-medvision
pip install -r requirements.txt
```

> GPU strongly recommended (CUDA). The full pipeline (YOLO + MedGemma-4B 4-bit) requires ~6 GB VRAM.  
> On a T4 GPU, YOLO and the LLM must be loaded sequentially — the notebook handles this automatically.

---

## Quick Start

### Run inference on a single X-ray

```bash
# Set your HuggingFace token (required for MedGemma access)
export HF_TOKEN=hf_your_token_here

python src/predict.py \
    --image path/to/xray.jpg \
    --model path/to/best.pt \
    --output report.txt
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | — | Path to the chest X-ray image (JPG/PNG) |
| `--model` | — | Path to YOLO `best.pt` weights |
| `--conf`  | `0.25` | YOLO detection confidence threshold |
| `--output` | — | Optional path to save the text report |
| `--no-llm` | `False` | Run YOLO only, skip MedGemma generation |

### YOLO-only (no LLM)

```bash
python src/predict.py \
    --image path/to/xray.jpg \
    --model path/to/best.pt \
    --no-llm
```

### Build a training subset

```python
from src.dataset import build_balanced_subset

stats = build_balanced_subset(
    source_dir="path/to/vinbigdata-yolo-dataset-with-wbf-3x-downscaled",
    output_dir="path/to/my-subset",
    total_images=5000,
)
print(stats)
```

---

## Notebook

Open [`notebooks/MEDICALCAPTIONING_CLEAN.ipynb`](notebooks/MEDICALCAPTIONING_CLEAN.ipynb) in Google Colab for the full end-to-end workflow:

1. Mount Google Drive & unzip the dataset
2. Explore and visualise class distributions
3. Build balanced subsets (5k and 7.5k)
4. Train V1s, V1m, and V2 models
5. Evaluate on the test set
6. Run the YOLO + MedGemma inference pipeline

> Set your HuggingFace token as a Colab secret named `HF_TOKEN` before running the LLM cells.

---

## Acknowledgements

- **VinBigData** — dataset and original annotations: [Nguyen et al., 2022](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
- **Ultralytics** — YOLOv8 implementation
- **Google DeepMind** — MedGemma-4B-IT medical vision-language model
- Dataset preprocessing inspired by the Kaggle community kernel by **buithanhxuan**

---

## Medical Disclaimer

> **This project is for research and educational purposes only.**  
> The AI-generated reports must **never** be used as a substitute for professional medical diagnosis.  
> All outputs must be reviewed and validated by a qualified radiologist before any clinical use.  
> The authors assume no liability for any medical decisions made based on this tool.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
