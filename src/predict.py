"""
Standalone inference script: YOLO detection + MedGemma report generation.

Usage:
    python src/predict.py --image path/to/xray.jpg --model path/to/best.pt
    python src/predict.py --image path/to/xray.jpg --model path/to/best.pt --conf 0.30 --output report.txt
"""

import argparse
import gc
import os

import torch
from PIL import Image
from ultralytics import YOLO

from utils import CLASS_NAMES, build_llm_prompt, draw_boxes, get_confidence_level


def run_yolo_detection(yolo_model: YOLO, image_path: str, conf_threshold: float = 0.25) -> list:
    """Run YOLO inference and return a list of detection dicts."""
    device = 0 if torch.cuda.is_available() else "cpu"
    results = yolo_model.predict(
        source=image_path, conf=conf_threshold, imgsz=640, device=device, verbose=False
    )
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_level, emoji, warning = get_confidence_level(conf)
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": conf,
                    "conf_level": conf_level,
                    "emoji": emoji,
                    "warning": warning,
                    "bbox": [x1, y1, x2, y2],
                }
            )
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    return detections


def load_medgemma(model_id: str = "google/medgemma-4b-it"):
    """Load MedGemma-4B with 4-bit quantization."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def run_llm_analysis(annotated_path: str, prompt: str, processor, llm_model) -> str:
    """Send annotated image + prompt to MedGemma and return the generated report."""
    pil_image = Image.open(annotated_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = llm_model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = output[0][input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Chest X-ray analysis: YOLO + MedGemma")
    parser.add_argument("--image",  required=True, help="Path to the chest X-ray image")
    parser.add_argument("--model",  required=True, help="Path to YOLO best.pt weights")
    parser.add_argument("--conf",   type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--output", default=None, help="Optional path to save the text report")
    parser.add_argument("--no-llm", action="store_true", help="Run YOLO only, skip MedGemma")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    # Authenticate HuggingFace if token is set
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token and not args.no_llm:
        from huggingface_hub import login
        login(token=hf_token)

    # --- YOLO detection ---
    print("\n[1/4] Loading YOLO model...")
    yolo_model = YOLO(args.model)

    print("[2/4] Running detection...")
    detections = run_yolo_detection(yolo_model, args.image, args.conf)

    annotated_path = args.image.rsplit(".", 1)[0] + "_annotated.png"
    draw_boxes(args.image, detections, output_path=annotated_path)
    print(f"      Annotated image saved to: {annotated_path}")

    print(f"\n      {len(detections)} detection(s):")
    for d in detections:
        print(f"        {d['emoji']} {d['class_name']:<25} — {d['confidence']:.1%} ({d['conf_level']})")

    if args.no_llm:
        return

    # Free YOLO before loading the LLM (T4 GPU memory constraint)
    print("\n[3/4] Releasing YOLO from GPU memory...")
    del yolo_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- MedGemma report ---
    print("[4/4] Loading MedGemma-4B and generating report...")
    processor, llm_model = load_medgemma()

    prompt   = build_llm_prompt(detections)
    response = run_llm_analysis(annotated_path, prompt, processor, llm_model)

    separator = "=" * 60
    report = f"\n{separator}\nRADIOLOGICAL ANALYSIS REPORT\n{separator}\n{response}\n{separator}\n"
    report += "\nWARNING: This analysis is AI-generated and does NOT replace a qualified radiologist.\n"

    print(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
