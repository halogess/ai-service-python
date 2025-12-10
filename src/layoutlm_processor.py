import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image, ImageDraw
import os
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

MODEL_PATH = "/app/models/layoutlmv3"

model = None
processor = None


def load_model():
    global model, processor
    if model is None:
        print("Loading LayoutLMv3 model...")
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
    return model, processor


def process_image_with_layoutlm(image_path, text_data):
    """
    Proses satu halaman (image) dengan LayoutLMv3 menggunakan PyMuPDF text data.
    """
    model, processor = load_model()
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if not text_data:
        raise ValueError("text_data is required - PyMuPDF only mode")

    # PyMuPDF processing - normalize bbox to 0-1000
    pdf_width = text_data.get("width", image.size[0])
    pdf_height = text_data.get("height", image.size[1])
    scale_x, scale_y = 1000 / pdf_width, 1000 / pdf_height

    words = []
    boxes = []
    original_boxes = []  # Store original bbox for visualization

    for word_data in text_data.get("words", []):
        text = word_data["text"].strip()
        if text:
            x0, y0, x1, y1 = word_data["bbox"]
            words.append(text)
            # Normalized for LayoutLMv3
            boxes.append([
                int(x0 * scale_x),
                int(y0 * scale_y),
                int(x1 * scale_x),
                int(y1 * scale_y),
            ])
            # Original for visualization
            original_boxes.append([x0, y0, x1, y1])

    if not words:
        return {
            "image": image_path,
            "predictions": [],
            "boxes": [],
            "image_token_map": {},
        }

    logger.info(f"Processing {len(words)} words with sliding window")

    # Sliding window encoding
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
        stride=1,
        return_overflowing_tokens=True
    )

    # Move to device
    for key in ("input_ids", "attention_mask", "bbox"):
        if key in encoding and isinstance(encoding[key], torch.Tensor):
            encoding[key] = encoding[key].to(device)

    pv = encoding.get("pixel_values")
    if isinstance(pv, torch.Tensor):
        pixel_values_tensor = pv.to(device)
    elif isinstance(pv, list) and len(pv) > 0:
        pixel_values_tensor = pv[0].unsqueeze(0).to(device)
    else:
        raise TypeError("Unexpected type for pixel_values")

    num_windows = len(encoding["input_ids"])
    logger.info(f"Processing {len(words)} words in {num_windows} windows")

    token_predictions = {}
    debug_predictions = []  # For HTML debug

    for window_idx in range(num_windows):
        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"][window_idx:window_idx+1],
                attention_mask=encoding["attention_mask"][window_idx:window_idx+1],
                bbox=encoding["bbox"][window_idx:window_idx+1],
                pixel_values=pixel_values_tensor,
            )

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1)
        word_ids = encoding.word_ids(batch_index=window_idx)

        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None:
                label_id = pred_ids[token_idx].item()
                label_prob = probs[token_idx, label_id].item()
                
                # Store for debug HTML (all predictions)
                debug_predictions.append({
                    'word_idx': word_id,
                    'window_idx': window_idx,
                    'label': model.config.id2label.get(label_id, "UNKNOWN"),
                    'label_id': label_id,
                    'confidence': label_prob
                })

                # Keep best prediction for final result
                if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                    token_predictions[word_id] = (label_id, label_prob)

    all_predictions = [
        {
            "token_id": word_id,
            "label": model.config.id2label.get(pred[0], "UNKNOWN"),
            "label_id": pred[0],
            "confidence": pred[1],
        }
        for word_id, pred in sorted(token_predictions.items())
    ]

    logger.info(f"Processed {len(all_predictions)} words in {num_windows} windows")

    return {
        "image": image_path,
        "predictions": all_predictions,
        "boxes": original_boxes,  # Use original bbox for visualization
        "image_token_map": {},
        "debug_predictions": debug_predictions,
        "words": words,
        "final_predictions": {word_id: {"label": model.config.id2label.get(pred[0], "UNKNOWN"), "confidence": pred[1]} for word_id, pred in token_predictions.items()}
    }



def draw_boxes_on_image(image_path, boxes, labels, output_path, pdf_dimensions=None):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    # Boxes already in PDF coordinates, need to scale to image coordinates
    # Image is 300 DPI (300/72 = 4.166x scale from PDF)
    pdf_width = pdf_dimensions["width"] if pdf_dimensions else img_width
    pdf_height = pdf_dimensions["height"] if pdf_dimensions else img_height
    scale_x, scale_y = (300/72), (300/72)  # PDF to image scale

    colors = {
        "Title": "red",
        "Text": "blue",
        "List-item": "green",
        "Table": "orange",
        "Picture": "purple",
        "Caption": "cyan",
        "Section-header": "magenta",
        "Page-header": "yellow",
        "Page-footer": "pink",
        "Footnote": "brown",
        "Formula": "teal"
    }

    for box, label in zip(boxes, labels):
        if box and len(box) == 4:
            # Scale from PDF coordinates to image coordinates
            x1, y1, x2, y2 = (
                int(box[0] * scale_x),
                int(box[1] * scale_y),
                int(box[2] * scale_x),
                int(box[3] * scale_y),
            )
            color = colors.get(label, "gray")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(y1 - 15, 0)), label, fill=color)

    image.save(output_path)


def process_document(image_paths, output_dir=None, pdf_text_data=None):
    """
    Proses banyak halaman sekaligus dan (opsional) simpan visualisasi.
    Raw LayoutLMv3 predictions per word.
    """
    results = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, image_path in enumerate(image_paths, start=1):
        print(f"Processing page {i}/{len(image_paths)}: {os.path.basename(image_path)}")

        text_data = pdf_text_data[i - 1] if pdf_text_data and i <= len(pdf_text_data) else None

        page_result = process_image_with_layoutlm(image_path, text_data)
        page_result["page_number"] = i
        results.append(page_result)
        
        # Save debug JSON for HTML generation
        if output_dir and "debug_predictions" in page_result:
            import json
            debug_json_path = os.path.join(output_dir, f"debug_page_{i}.json")
            with open(debug_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'words': page_result.get('words', []),
                    'debug_predictions': page_result.get('debug_predictions', []),
                    'final_predictions': page_result.get('final_predictions', {})
                }, f, indent=2, ensure_ascii=False)
            print(f"Saved debug JSON: {debug_json_path}")

        if not output_dir:
            continue

        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_filename)

        # Raw predictions per word
        all_boxes = page_result.get("boxes", [])
        all_labels = [pred.get("label", "Text") for pred in page_result.get("predictions", [])]

        if all_boxes:
            pdf_dims = {"width": text_data.get("width"), "height": text_data.get("height")}
            draw_boxes_on_image(image_path, all_boxes, all_labels, output_path, pdf_dims)
            print(f"Saved visualization with {len(all_boxes)} words ({len(page_result.get('predictions', []))} tokens processed)")
            
            # Generate debug HTML
            if "debug_predictions" in page_result:
                debug_html_path = os.path.join(output_dir, f"debug_page_{i}.html")
                os.system(f'python3.11 generate_debug_html.py "{os.path.join(output_dir, f"debug_page_{i}.json")}" "{debug_html_path}"')
                print(f"Saved debug HTML: {debug_html_path}")

    return results
