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


def process_image_with_layoutlm(image_path, text_data=None):
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
    model, processor = load_model()
    device = model.device

    image = Image.open(image_path).convert("RGB")

    if text_data:
        pdf_width = text_data.get("width", image.size[0])
        pdf_height = text_data.get("height", image.size[1])
        scale_x, scale_y = 1000 / pdf_width, 1000 / pdf_height

        words = []
        boxes = []

        for word_data in text_data.get("words", []):
            text = word_data["text"].strip()
            if text:
                x0, y0, x1, y1 = word_data["bbox"]
                words.append(text)
                boxes.append(
                    [
                        int(x0 * scale_x),
                        int(y0 * scale_y),
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                    ]
                )

        image_token_map = {}
        images_in_data = text_data.get("images", [])
        logger.info(f"Found {len(images_in_data)} images in text_data")

        for img_idx, img in enumerate(images_in_data):
            bbox = img.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                word_idx = len(words)
                image_token_map[word_idx] = img_idx
                words.append("[IMG]")
                boxes.append(
                    [
                        int(x0 * scale_x),
                        int(y0 * scale_y),
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                    ]
                )
                logger.info(
                    f"  Added [IMG] token at word_idx={word_idx} for image {img_idx}, bbox={bbox}"
                )

        logger.info(
            f"Total words including [IMG]: {len(words)}, image_token_map: {image_token_map}"
        )

        if not words:
            return {
                "image": image_path,
                "predictions": [],
                "boxes": [],
                "image_token_map": {},
            }

        # 3) Encoding via processor dengan sliding window (overflowing tokens)
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
        )

        # Pindahkan tensor penting ke device
        for key in ("input_ids", "attention_mask", "bbox"):
            if key in encoding and isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(device)

        # pixel_values kadang tensor, kadang list of tensor
        pv = encoding.get("pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_tensor = pv.to(device)
        elif isinstance(pv, list) and len(pv) > 0:
            pixel_values_tensor = pv[0].unsqueeze(0).to(device)
        else:
            raise TypeError("Unexpected type for pixel_values from processor")

        num_windows = len(encoding["input_ids"])
        logger.info(
            f"Processing {len(words)} words (incl. {len(image_token_map)} [IMG] tokens) in {num_windows} windows"
        )

        token_predictions = {}

        for window_idx in range(num_windows):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"][window_idx : window_idx + 1],
                    attention_mask=encoding["attention_mask"][
                        window_idx : window_idx + 1
                    ],
                    bbox=encoding["bbox"][window_idx : window_idx + 1],
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

                    if (
                        word_id not in token_predictions
                        or label_prob > token_predictions[word_id][1]
                    ):
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

        logger.info(f"Processed {len(words)} words in {num_windows} windows")

        return {
            "image": image_path,
            "predictions": all_predictions,
            "boxes": boxes,  # bbox per word (termasuk [IMG])
            "image_token_map": image_token_map,
        }

    encoding = processor(image, return_tensors="pt")
    encoding = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in encoding.items()
    }

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    id2label = model.config.id2label

    return {
        "image": image_path,
        "predictions": [
            {"token_id": idx, "label": id2label.get(pred, "UNKNOWN"), "label_id": pred}
            for idx, pred in enumerate(
                predictions if isinstance(predictions, list) else []
            )
        ],
        "boxes": encoding.get("bbox", torch.tensor([])).squeeze().tolist(),
        "image_token_map": {},
    }


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


def process_document(image_paths, output_dir=None, pdf_text_data=None):
    """
    Proses banyak halaman sekaligus dan (opsional) simpan visualisasi.

    Args:
        image_paths (list[str]): daftar path image halaman.
        output_dir (str|None): folder untuk simpan hasil gambar dengan bbox.
        pdf_text_data (list[dict]|None): list text_data per halaman.

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

        page_result = process_image_with_layoutlm(image_path, text_data)
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
                    
                    if label != "Picture":
                        label = "Picture"
                    
                    logger.info(
                        f"[IMG] token at word_idx={word_idx} normalized to: {label} (conf={conf:.3f})"
                    )
                    if img_idx not in img_best or conf > img_best[img_idx][1]:
                        img_best[img_idx] = (label, conf)

            image_blocks = [b for b in text_data.get("blocks", []) if b.get("type") == "image"]

            for block in text_data.get("blocks", []):
                bbox = block.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                block_type = block.get("type")

                if block_type == "image":
                    label = "Picture"
                    for img_idx, (pred_label, conf) in img_best.items():
                        if img_idx < len(text_data.get("images", [])):
                            img_data = text_data["images"][img_idx]
                            if img_data.get("bbox") == bbox:
                                label = pred_label
                                break
                else:
                    block_idx = text_data.get("blocks", []).index(block)
                    scores = block_label_scores.get(block_idx)
                    if scores:
                        label = scores.most_common(1)[0][0]
                    else:
                        label = "Text"
                    
                    text_lower = block.get("text", "").lower().strip()
                    is_gambar = text_lower.startswith("gambar ") or text_lower.startswith("figure ")
                    is_tabel = text_lower.startswith("tabel ") or text_lower.startswith("table ")
                    
                    if is_gambar or is_tabel:
                        y_top, y_bot = bbox[1], bbox[3]
                        
                        for img_block in image_blocks:
                            img_bbox = img_block.get("bbox")
                            if img_bbox:
                                img_y_top, img_y_bot = img_bbox[1], img_bbox[3]
                                
                                if abs(y_top - img_y_bot) < 40 or abs(img_y_top - y_bot) < 40:
                                    label = "Caption"
                                    break

                all_boxes.append(bbox)
                all_labels.append(label)

        if all_boxes:
            pdf_dims = (
                {"width": text_data.get("width"), "height": text_data.get("height")}
                if text_data
                else None
            )
            draw_boxes_on_image(
                image_path, all_boxes, all_labels, output_path, pdf_dims
            )
            print(
                f"Saved visualization with {len(all_boxes)} blocks "
                f"({len(page_result.get('predictions', []))} tokens processed)"
            )

    return results
