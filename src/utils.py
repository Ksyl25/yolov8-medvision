"""
Utility functions for chest X-ray analysis pipeline.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = [
    "Aortic_enlargement", "Atelectasis", "Calcification",
    "Cardiomegaly", "Consolidation", "ILD", "Infiltration",
    "Lung_Opacity", "Nodule/Mass", "Other_lesion",
    "Pleural_effusion", "Pleural_thickening", "Pneumothorax",
    "Pulmonary_fibrosis", "No finding",
]

CONFIDENCE_THRESHOLDS = {"high": 0.70, "medium": 0.45, "low": 0.25}

COLORS = {"HAUTE": "#00C853", "MOYENNE": "#FFD600", "FAIBLE": "#FF1744"}


def get_confidence_level(conf: float) -> tuple[str, str, str]:
    """Return (level_label, emoji, description) for a confidence score."""
    if conf >= CONFIDENCE_THRESHOLDS["high"]:
        return "HAUTE", "🟢", "Le modèle est très sûr"
    elif conf >= CONFIDENCE_THRESHOLDS["medium"]:
        return "MOYENNE", "🟡", "Confiance modérée"
    else:
        return "FAIBLE", "🔴", "Incertain, interpréter avec précaution"


def draw_boxes(image_path: str, detections: list, output_path: str = "annotated_xray.png") -> str:
    """
    Draw bounding boxes on an X-ray image and save the result.

    Args:
        image_path: Path to the input image.
        detections: List of detection dicts from run_yolo_detection().
        output_path: Where to save the annotated image.

    Returns:
        Path to the saved annotated image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)

    for det in detections:
        if det["class_name"] == "No finding":
            continue
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS[det["conf_level"]]
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
            bbox=dict(facecolor=color, alpha=0.8, pad=2, edgecolor="none"),
        )

    ax.axis("off")
    ax.set_title("Détections YOLOv8 — Radiographie thoracique", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_llm_prompt(detections: list) -> str:
    """
    Build the structured prompt to send to MedGemma from YOLO detections.

    Args:
        detections: List of detection dicts from run_yolo_detection().

    Returns:
        Formatted prompt string.
    """
    high_conf   = [d for d in detections if d["conf_level"] == "HAUTE"   and d["class_name"] != "No finding"]
    medium_conf = [d for d in detections if d["conf_level"] == "MOYENNE" and d["class_name"] != "No finding"]
    low_conf    = [d for d in detections if d["conf_level"] == "FAIBLE"  and d["class_name"] != "No finding"]
    no_finding  = [d for d in detections if d["class_name"] == "No finding"]

    prompt = (
        "You are an expert radiologist assistant analyzing a chest X-ray.\n"
        "A YOLO detection model has identified the following findings:\n\n"
        "=== DETECTIONS ===\n"
    )

    if high_conf:
        prompt += "\n🟢 HIGH CONFIDENCE (model is very certain):\n"
        for d in high_conf:
            prompt += f"  - {d['class_name']} (confidence: {d['confidence']:.1%})\n"
    if medium_conf:
        prompt += "\n🟡 MEDIUM CONFIDENCE (moderate caution):\n"
        for d in medium_conf:
            prompt += f"  - {d['class_name']} (confidence: {d['confidence']:.1%})\n"
    if low_conf:
        prompt += "\n🔴 LOW CONFIDENCE (high caution):\n"
        for d in low_conf:
            prompt += f"  - {d['class_name']} (confidence: {d['confidence']:.1%})\n"
    if no_finding:
        prompt += f"\n✅ No pathological finding (confidence: {no_finding[0]['confidence']:.1%})\n"
    if not any([high_conf, medium_conf, low_conf, no_finding]):
        prompt += "\nNo findings detected.\n"

    prompt += (
        "\n=== YOUR TASK ===\n"
        "1. SUMMARY: Brief clinical summary of findings\n"
        "2. HIGH PRIORITY: Detail high confidence findings and their clinical significance\n"
        "3. UNCERTAIN: Comment on medium/low confidence detections\n"
        "4. RECOMMENDATIONS: Suggest follow-up or additional imaging if needed\n"
        "5. DISCLAIMER: Remind this is AI-assisted and must be confirmed by a radiologist\n"
    )
    return prompt
