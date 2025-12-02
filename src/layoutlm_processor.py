import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image, ImageDraw
import os
import logging
import re
from collections import Counter, defaultdict

CAPTION_PATTERN = re.compile(r"^(tabel|table|gambar|figure)\s+\d+(\.\d+)*\s*[:.)\s]?", re.IGNORECASE)

logger = logging.getLogger(__name__)

MODEL_PATH = "/app/models/layoutlmv3"

model = None
processor = None


def load_model(apply_ocr=False):
    global model, processor
    if model is None:
        print("Loading LayoutLMv3 model...")
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
    
    # Processor bisa berbeda tergantung apply_ocr
    processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=apply_ocr)
    return model, processor


def process_image_with_layoutlm(image_path, text_data=None, use_builtin_ocr=False):
    """
    Proses satu halaman (image) dengan LayoutLMv3.

    Args:
        image_path (str): path ke file image halaman.
        text_data (dict | None): data teks & layout dari PDF:
            {
              "width": ...,
              "height": ...,
              "words": [
                {"text": "...", "bbox": [x0,y0,x1,y1], "block_no": int},
                ...
              ],
              "blocks": [
                {"text": "...", "bbox": [x0,y0,x1,y1], "type": "text"|"image"},
                ...
              ],
              "images": [
                {"bbox": [x0,y0,x1,y1], "block_no": int},
                ...
              ]
            }

    Returns:
        dict:
            {
              "image": image_path,
              "predictions": [ {token_id, label, label_id, confidence}, ... ],
              "boxes": [bbox per word/token],
              "image_token_map": {word_idx: image_idx}
            }
    """
    # Jika use_builtin_ocr=True, gunakan OCR internal LayoutLMv3
    model, processor = load_model(apply_ocr=use_builtin_ocr)
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if text_data and not use_builtin_ocr:
        # Single pass seperti test_image_raw - TANPA sliding window
        words = []
        boxes = []

        for word_data in text_data.get("words", []):
            text = word_data["text"].strip()
            if text:
                bbox = word_data["bbox"]
                words.append(text)
                boxes.append(bbox)  # Sudah normalized 0-1000 dari ocr_processor

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
            stride=128,
            return_overflowing_tokens=True
        )

        # Handle pixel_values (shared across windows)
        if isinstance(encoding.pixel_values, list):
            pixel_values_tensor = encoding.pixel_values[0].unsqueeze(0).to(device)
        else:
            pixel_values_tensor = encoding.pixel_values.to(device)

        num_windows = len(encoding["input_ids"])
        logger.info(f"Processing {len(words)} words in {num_windows} windows")

        # Store all predictions with probabilities
        token_predictions = {}  # word_idx -> list of (label_id, probability)

        # Process each window
        for window_idx in range(num_windows):
            window_encoding = {
                "input_ids": encoding["input_ids"][window_idx:window_idx+1].to(device),
                "attention_mask": encoding["attention_mask"][window_idx:window_idx+1].to(device),
                "bbox": encoding["bbox"][window_idx:window_idx+1].long().to(device),
                "pixel_values": pixel_values_tensor
            }

            with torch.no_grad():
                outputs = model(**window_encoding)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                pred_ids = logits.argmax(-1)

            # Map tokens to words
            word_ids = encoding.word_ids(batch_index=window_idx)

            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                label_id = pred_ids[token_idx].item()
                label_prob = probs[token_idx, label_id].item()

                if word_id not in token_predictions:
                    token_predictions[word_id] = []

                token_predictions[word_id].append((label_id, label_prob))

        # Aggregate: use highest probability
        final_predictions = {}
        for word_id, preds in token_predictions.items():
            best_pred, best_prob = max(preds, key=lambda x: x[1])
            final_predictions[word_id] = (best_pred, best_prob)

        all_predictions = [
            {
                "token_id": word_id,
                "label": model.config.id2label.get(pred[0], "UNKNOWN"),
                "label_id": pred[0],
                "confidence": pred[1],
            }
            for word_id, pred in sorted(final_predictions.items())
        ]

        logger.info(f"Processed {len(all_predictions)} words in {num_windows} windows")

        return {
            "image": image_path,
            "predictions": all_predictions,
            "boxes": boxes,
            "image_token_map": {},
        }

    # LayoutLMv3 built-in OCR - single pass seperti test_image_raw
    encoding = processor(image, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    bbox = encoding['bbox'].long().to(device)
    pixel_values = encoding['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1)
    
    word_ids = encoding.word_ids(batch_index=0)
    token_predictions = {}
    
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None:
            label_id = pred_ids[token_idx].item()
            label_prob = probs[token_idx, label_id].item()
            
            if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                token_predictions[word_id] = (label_id, label_prob)
    
    id2label = model.config.id2label
    
    all_predictions = [
        {
            "token_id": word_id,
            "label": id2label.get(pred[0], "UNKNOWN"),
            "label_id": pred[0],
            "confidence": pred[1],
        }
        for word_id, pred in sorted(token_predictions.items())
    ]
    
    boxes = encoding.get("bbox", torch.tensor([])).squeeze(0).tolist()
    
    logger.info(f"Processed with LayoutLMv3 OCR: {len(all_predictions)} words (truncated at 512 tokens)")
    
    return {
        "image": image_path,
        "predictions": all_predictions,
        "boxes": boxes,
        "image_token_map": {},
    }


def build_table_regions_from_tokens(page_result, text_data):
    """
    Bangun region tabel dari semua token yang diprediksi 'Table' oleh LayoutLM,
    tanpa threshold minimal confidence / gap.
    """
    predictions = page_result.get("predictions", [])
    words = text_data.get("words", [])

    pred_by_token = {
        p.get("token_id"): p
        for p in predictions
        if p.get("token_id") is not None
    }

    table_boxes = []
    for idx, word in enumerate(words):
        pred = pred_by_token.get(idx)
        if not pred:
            continue

        if pred.get("label") == "Table":
            bbox = word.get("bbox")
            if bbox and len(bbox) == 4:
                table_boxes.append(bbox)

    if not table_boxes:
        return []

    x0 = min(bb[0] for bb in table_boxes)
    y0 = min(bb[1] for bb in table_boxes)
    x1 = max(bb[2] for bb in table_boxes)
    y1 = max(bb[3] for bb in table_boxes)

    return [[x0, y0, x1, y1]]


def center_in_regions(bbox, regions):
    """
    Cek apakah center dari bbox berada di salah satu region.
    """
    if not regions:
        return False

    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    for rx0, ry0, rx1, ry1 in regions:
        if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
            return True
    return False


def merge_table_boxes(boxes, labels, distance_threshold=50):
    """
    Gabungkan blok-blok Table yang berdekatan menjadi satu kotak besar.
    """
    table_indices = [i for i, lab in enumerate(labels) if lab == "Table"]
    if len(table_indices) <= 1:
        return boxes, labels
    
    table_boxes = [boxes[i] for i in table_indices]
    used = [False] * len(table_boxes)
    merged_tables = []
    
    for i, box1 in enumerate(table_boxes):
        if used[i]:
            continue
        
        group = [box1]
        used[i] = True
        
        for j, box2 in enumerate(table_boxes):
            if used[j]:
                continue
            
            x0_1, y0_1, x1_1, y1_1 = box1
            x0_2, y0_2, x1_2, y1_2 = box2
            
            if (abs(y0_1 - y1_2) < distance_threshold or abs(y0_2 - y1_1) < distance_threshold):
                if not (x1_1 < x0_2 - distance_threshold or x1_2 < x0_1 - distance_threshold):
                    group.append(box2)
                    used[j] = True
        
        x0 = min(b[0] for b in group)
        y0 = min(b[1] for b in group)
        x1 = max(b[2] for b in group)
        y1 = max(b[3] for b in group)
        merged_tables.append([x0, y0, x1, y1])
    
    result_boxes = []
    result_labels = []
    
    for i, (b, lab) in enumerate(zip(boxes, labels)):
        if lab != "Table":
            result_boxes.append(b)
            result_labels.append(lab)
    
    for merged_box in merged_tables:
        result_boxes.append(merged_box)
        result_labels.append("Table")
    
    return result_boxes, result_labels


def draw_boxes_on_image(image_path, boxes, labels, output_path, pdf_dimensions=None):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    pdf_width = pdf_dimensions["width"] if pdf_dimensions else img_width
    pdf_height = pdf_dimensions["height"] if pdf_dimensions else img_height
    scale_x, scale_y = img_width / pdf_width, img_height / pdf_height

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


def process_document(image_paths, output_dir=None, pdf_text_data=None, post_processing_level="none", use_builtin_ocr=False):
    """
    Proses banyak halaman sekaligus dan (opsional) simpan visualisasi.

    Args:
        image_paths (list[str]): daftar path image halaman.
        output_dir (str|None): folder untuk simpan hasil gambar dengan bbox.
        pdf_text_data (list[dict]|None): list text_data per halaman.
        post_processing_level (str): "none", "sliding_window", "full"
            - "none": Raw LayoutLMv3 predictions tanpa post-processing
            - "sliding_window": Hanya token aggregation dari sliding window (post-processing 3)
            - "full": Semua post-processing (1,2,3,4)

    Returns:
        list[dict]: hasil LayoutLM per halaman.
    """
    results = []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, image_path in enumerate(image_paths, start=1):
        print(f"Processing page {i}/{len(image_paths)}: {os.path.basename(image_path)}")

        text_data = (
            pdf_text_data[i - 1] if pdf_text_data and i <= len(pdf_text_data) else None
        )

        page_result = process_image_with_layoutlm(image_path, text_data, use_builtin_ocr=use_builtin_ocr)
        page_result["page_number"] = i
        results.append(page_result)

        if not output_dir:
            continue

        output_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, output_filename)

        all_boxes = []
        all_labels = []

        if text_data:
            word_to_block = [w["block_no"] for w in text_data.get("words", [])]
            block_label_scores = defaultdict(Counter)
            image_token_map = page_result.get("image_token_map", {})
            img_best = {}

            for pred in page_result.get("predictions", []):
                word_idx = pred.get("token_id")
                if word_idx is None:
                    continue

                label = pred.get("label", "Text")
                conf = pred.get("confidence", 0.0)

                if word_idx < len(word_to_block):
                    block_label_scores[word_to_block[word_idx]][label] += conf
                elif word_idx in image_token_map:
                    img_idx = image_token_map[word_idx]
                    label = "Picture"
                    logger.info(
                        f"[IMG] token at word_idx={word_idx} normalized to: {label} (conf={conf:.3f})"
                    )
                    if img_idx not in img_best or conf > img_best[img_idx][1]:
                        img_best[img_idx] = (label, conf)

            # Post-processing level: none
            if post_processing_level == "none":
                # Raw predictions - gambar per word dari boxes di page_result
                boxes_from_result = page_result.get("boxes", [])
                predictions = page_result.get("predictions", [])
                
                for pred in predictions:
                    token_id = pred.get("token_id")
                    if token_id is not None and token_id < len(boxes_from_result):
                        bbox = boxes_from_result[token_id]
                        label = pred.get("label", "Text")
                        all_boxes.append(bbox)
                        all_labels.append(label)
            
            # Post-processing level: sliding_window (hanya #3)
            elif post_processing_level == "sliding_window":
                border_regions = text_data.get("table_regions_border", []) or []
                
                if border_regions:
                    table_regions = border_regions
                    logger.info(f"Using {len(table_regions)} table regions from PDF borders")
                else:
                    table_regions = build_table_regions_from_tokens(page_result, text_data)
                    logger.info(f"Using {len(table_regions)} table regions from LayoutLM tokens")
                
                blocks_list = text_data.get("blocks", [])
                images_list = text_data.get("images", [])
                
                block_infos = []
                for block_idx, block in enumerate(blocks_list):
                    bbox = block.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    block_type = block.get("type")
                    text = (block.get("text") or "").strip()
                    
                    if block_type == "image":
                        label = "Picture"
                    else:
                        scores = block_label_scores.get(block_idx)
                        if scores:
                            list_score = scores.get("List-item", 0)
                            text_score = scores.get("Text", 0)
                            if list_score > text_score * 0.3:
                                label = "List-item"
                            else:
                                label = scores.most_common(1)[0][0]
                        else:
                            label = "Text"
                    
                    block_infos.append({"idx": block_idx, "bbox": bbox, "text": text, "type": block_type, "label": label})
                
                # Post-processing #4: Table region override
                for info in block_infos:
                    if center_in_regions(info["bbox"], table_regions):
                        info["label"] = "Table"
                
                for info in block_infos:
                    if info["label"] != "Table":
                        all_boxes.append(info["bbox"])
                        all_labels.append(info["label"])
                
                for region in table_regions:
                    all_boxes.append(region)
                    all_labels.append("Table")
            
            # Post-processing level: full (semua #1,2,3,4)
            else:
                border_regions = text_data.get("table_regions_border", []) or []

                if border_regions:
                    table_regions = border_regions
                    logger.info(f"Using {len(table_regions)} table regions from PDF borders")
                else:
                    table_regions = build_table_regions_from_tokens(page_result, text_data)
                    logger.info(f"Using {len(table_regions)} table regions from LayoutLM tokens")

                blocks_list = text_data.get("blocks", [])
                images_list = text_data.get("images", [])

                block_infos = []
                for block_idx, block in enumerate(blocks_list):
                    bbox = block.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue

                    block_type = block.get("type")
                    text = (block.get("text") or "").strip()

                    if block_type == "image":
                        label = "Picture"
                        for img_idx, (pred_label, conf) in img_best.items():
                            if img_idx < len(images_list):
                                img_data = images_list[img_idx]
                                if img_data.get("bbox") == bbox:
                                    label = pred_label
                                    break
                    else:
                        scores = block_label_scores.get(block_idx)
                        if scores:
                            list_score = scores.get("List-item", 0)
                            text_score = scores.get("Text", 0)
                            if list_score > text_score * 0.3:
                                label = "List-item"
                            else:
                                label = scores.most_common(1)[0][0]
                        else:
                            label = "Text"

                    block_infos.append({"idx": block_idx, "bbox": bbox, "text": text, "type": block_type, "label": label})

                # Post-processing #2: Caption detection
                for info in block_infos:
                    if CAPTION_PATTERN.match(info["text"] or ""):
                        info["label"] = "Caption"

                # Post-processing #4: Table region override
                for info in block_infos:
                    if info["label"] == "Caption":
                        continue
                    if center_in_regions(info["bbox"], table_regions):
                        info["label"] = "Table"

                for info in block_infos:
                    if info["label"] != "Table":
                        all_boxes.append(info["bbox"])
                        all_labels.append(info["label"])

                for region in table_regions:
                    all_boxes.append(region)
                    all_labels.append("Table")
                
                # Post-processing #1: Merge table boxes
                all_boxes, all_labels = merge_table_boxes(all_boxes, all_labels)

        # Jika tidak ada text_data (LayoutLMv3 OCR), gambar dari boxes di page_result
        if not text_data and page_result.get("boxes"):
            boxes_from_result = page_result.get("boxes", [])
            predictions = page_result.get("predictions", [])
            
            # Map predictions ke boxes (sudah dalam koordinat 0-1000)
            for pred in predictions:
                token_id = pred.get("token_id")
                if token_id is not None and token_id < len(boxes_from_result):
                    bbox = boxes_from_result[token_id]
                    label = pred.get("label", "Text")
                    all_boxes.append(bbox)
                    all_labels.append(label)
        
        if all_boxes:
            pdf_dims = (
                {"width": text_data.get("width"), "height": text_data.get("height")}
                if text_data
                else {"width": 1000, "height": 1000}  # LayoutLMv3 uses 0-1000 scale
            )
            draw_boxes_on_image(
                image_path, all_boxes, all_labels, output_path, pdf_dims
            )
            print(
                f"Saved visualization with {len(all_boxes)} blocks "
                f"({len(page_result.get('predictions', []))} tokens processed) "
                f"[post-processing: {post_processing_level}]"
            )

    return results
