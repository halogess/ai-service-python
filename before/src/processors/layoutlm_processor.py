import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image, ImageDraw
import os
import time

MODEL_PATH = "models/layoutlmv3"
MODEL_NAME = "Kwan0/layoutlmv3-base-finetune-DocLayNet-100k"

model = None
processor = None

def clamp_bbox(bbox, min_val=0, max_val=1000):
    """Clamp bounding box coordinates to valid range for LayoutLM.
    
    LayoutLM expects normalized bounding boxes in range 0-1000.
    Negative values or values > 1000 cause CUDA device-side assert errors.
    """
    return [
        max(min_val, min(max_val, int(bbox[0]))),
        max(min_val, min(max_val, int(bbox[1]))),
        max(min_val, min(max_val, int(bbox[2]))),
        max(min_val, min(max_val, int(bbox[3])))
    ]

def load_model():
    global model, processor
    if model is None:
        print("Loading LayoutLMv3 model...")
        
        # Auto-download if model not exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found. Downloading from {MODEL_NAME}...")
            os.makedirs(MODEL_PATH, exist_ok=True)
            model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
            processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)
            model.save_pretrained(MODEL_PATH)
            processor.save_pretrained(MODEL_PATH)
            print(f"Model downloaded and saved to {MODEL_PATH}")
        else:
            model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
            processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
    return model, processor

def process_image_with_layoutlm(image_path, text_data):
    model, processor = load_model()
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if not text_data:
        raise ValueError("text_data is required")

    pdf_width = text_data.get("width", image.size[0])
    pdf_height = text_data.get("height", image.size[1])
    scale_x, scale_y = 1000 / pdf_width, 1000 / pdf_height
    
    # Convert PDF coordinates (72 DPI) to 300 DPI
    dpi_scale = 300.0 / 72.0

    words = []
    boxes = []
    original_boxes = []
    is_image_flags = []

    for word_data in text_data.get("words", []):
        text = word_data["text"].strip()
        if text:
            x0, y0, x1, y1 = word_data["bbox"]
            words.append(text)
            boxes.append(clamp_bbox([
                x0 * scale_x,
                y0 * scale_y,
                x1 * scale_x,
                y1 * scale_y,
            ]))
            # Convert to 300 DPI for storage
            original_boxes.append([x0 * dpi_scale, y0 * dpi_scale, x1 * dpi_scale, y1 * dpi_scale])
            is_image_flags.append(word_data.get("is_image", False))

    if not words:
        return {"predictions": [], "boxes": [], "words": []}

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True
    )

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
    token_predictions = {}

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

                if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                    token_predictions[word_id] = (label_id, label_prob)

    all_predictions = []
    for word_id, pred in sorted(token_predictions.items()):
        is_image = is_image_flags[word_id]
        
        # Get label and convert to Docling format (lowercase with underscore)
        if is_image:
            label = "picture"
        else:
            raw_label = model.config.id2label.get(pred[0], "UNKNOWN")
            # Convert "Title" -> "title", "List-item" -> "list_item", etc
            label = raw_label.lower().replace("-", "_")
        
        all_predictions.append({
            "token_id": word_id,
            "word": words[word_id],
            "label": label,
            "label_id": pred[0],
            "confidence": 1.0 if is_image else pred[1],
            "bbox": original_boxes[word_id]
        })

    merge_start = time.time()
    merged_blocks = merge_adjacent_boxes(all_predictions)
    merge_time = time.time() - merge_start
    print(f"Merge time: {merge_time:.3f}s for {len(all_predictions)} words → {len(merged_blocks)} blocks")
    
    return {
        "predictions": all_predictions,
        "blocks": merged_blocks,
        "boxes": original_boxes,
        "words": words,
        "merge_time": merge_time
    }

def merge_adjacent_boxes(predictions):
    if not predictions:
        return []
    
    merged = []
    current_group = [predictions[0]]
    
    for i in range(1, len(predictions)):
        prev = predictions[i-1]
        curr = predictions[i]
        
        same_label = prev['label'] == curr['label']
        
        # Calculate distances
        prev_height = prev['bbox'][3] - prev['bbox'][1]
        vertical_gap = curr['bbox'][1] - prev['bbox'][3]
        horizontal_gap = curr['bbox'][0] - prev['bbox'][2]
        
        # Same line: vertical position similar, horizontal gap small
        same_line = abs(curr['bbox'][1] - prev['bbox'][1]) < prev_height * 0.5 and horizontal_gap < prev_height * 2
        
        # Next line in same paragraph: vertical gap < 1.5x line height, starts from left
        next_line = vertical_gap > 0 and vertical_gap < prev_height * 1.5 and curr['bbox'][0] < prev['bbox'][0] + 50
        
        # Merge if same label and (same line OR next line in paragraph)
        if same_label and (same_line or next_line):
            current_group.append(curr)
        else:
            # Merge current group
            merged.append(merge_group(current_group))
            current_group = [curr]
    
    # Merge last group
    if current_group:
        merged.append(merge_group(current_group))
    
    return merged

def merge_group(group):
    if len(group) == 1:
        return group[0]
    
    # Merge bounding boxes
    x0 = min(item['bbox'][0] for item in group)
    y0 = min(item['bbox'][1] for item in group)
    x1 = max(item['bbox'][2] for item in group)
    y1 = max(item['bbox'][3] for item in group)
    
    # Merge words
    merged_word = ' '.join(item['word'] for item in group)
    
    # Average confidence
    avg_confidence = sum(item['confidence'] for item in group) / len(group)
    
    return {
        'token_id': group[0]['token_id'],
        'word': merged_word,
        'label': group[0]['label'],
        'label_id': group[0]['label_id'],
        'confidence': avg_confidence,
        'bbox': [x0, y0, x1, y1]
    }

def draw_boxes_on_image(image_path, predictions, output_path, pdf_width=None, pdf_height=None):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Calculate scale factor from PDF coordinates to image coordinates
    scale = 300/72  # DPI scale factor used when converting PDF to image

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

    for pred in predictions:
        box = pred["bbox"]
        label = pred["label"]
        # Scale PDF coordinates to image coordinates
        x1, y1, x2, y2 = int(box[0] * scale), int(box[1] * scale), int(box[2] * scale), int(box[3] * scale)
        color = colors.get(label, "gray")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(y1 - 15, 0)), label, fill=color)

    image.save(output_path)
