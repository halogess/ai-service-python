import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import os
import logging

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "/app/models/layoutlmv3"

# Load model and processor (load once)
model = None
processor = None

def load_model():
    """Load LayoutLMv3 model and processor"""
    global model, processor
    
    if model is None:
        print("Loading LayoutLMv3 model...")
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()  # Set to eval mode for stable predictions
        print(f"Model loaded on {device}")
    
    return model, processor

def process_image_with_layoutlm(image_path, text_data=None):
    """
    Process single image with LayoutLMv3
    
    Args:
        image_path: Path to image file
        text_data: Text and bounding boxes from PDF (optional)
    
    Returns:
        dict: Classification results with bounding boxes
    """
    model, processor = load_model()
    device = model.device
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare input
    if text_data:
        # Use word-level data from PDF
        words = []
        boxes = []
        
        # Get PDF page dimensions
        pdf_width = text_data.get("width", image.size[0])
        pdf_height = text_data.get("height", image.size[1])
        
        # Use word-level bboxes directly from PDF
        for word_data in text_data.get("words", []):
            text = word_data["text"].strip()
            if text:
                x0, y0, x1, y1 = word_data["bbox"]
                
                # Normalize to 0-1000 range
                normalized_box = [
                    int((x0 / pdf_width) * 1000),
                    int((y0 / pdf_height) * 1000),
                    int((x1 / pdf_width) * 1000),
                    int((y1 / pdf_height) * 1000)
                ]
                
                words.append(text)
                boxes.append(normalized_box)
        
        # Use tokenizer overflow mechanism
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
        
        # pixel_values is a list when using overflow tokens, extract the tensor
        if isinstance(encoding.pixel_values, list):
            pixel_values_tensor = encoding.pixel_values[0].unsqueeze(0).to(device)  # Add batch dimension
        else:
            pixel_values_tensor = encoding.pixel_values.to(device)
        
        encoding = encoding.to(device)
        num_windows = len(encoding["input_ids"])
        logger.info(f"Processing {len(words)} words in {num_windows} windows")
        
        # Store predictions with probabilities
        token_predictions = {}  # word_idx -> list of (label_id, probability)
        
        # Process each window
        for window_idx in range(num_windows):
            # Get window data - pixel_values is shared across all windows
            window_encoding = {
                "input_ids": encoding["input_ids"][window_idx:window_idx+1],
                "attention_mask": encoding["attention_mask"][window_idx:window_idx+1],
                "bbox": encoding["bbox"][window_idx:window_idx+1],
                "pixel_values": pixel_values_tensor
            }
            
            # Inference
            with torch.no_grad():
                outputs = model(**window_encoding)
            
            # Get predictions and probabilities
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
        
        # Aggregate predictions using highest probability
        all_predictions = []
        for word_id in sorted(token_predictions.keys()):
            preds = token_predictions[word_id]
            best_pred, best_prob = max(preds, key=lambda x: x[1])
            
            all_predictions.append({
                "token_id": word_id,
                "label": model.config.id2label.get(best_pred, "UNKNOWN"),
                "label_id": best_pred,
                "confidence": best_prob
            })
        
        logger.info(f"Processed {len(words)} words in {num_windows} windows")
        
        # Return results with all predictions
        return {
            "image": image_path,
            "predictions": all_predictions,
            "boxes": boxes  # Original boxes
        }
    
    # If no text data, process without text (layout only)
    encoding = processor(image, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    
    # Get labels
    id2label = model.config.id2label
    
    # Extract results
    results = {
        "image": image_path,
        "predictions": [],
        "boxes": encoding.bbox.squeeze().tolist() if "bbox" in encoding else []
    }
    
    # Map predictions to labels
    if isinstance(predictions, list):
        for idx, pred in enumerate(predictions):
            results["predictions"].append({
                "token_id": idx,
                "label": id2label.get(pred, "UNKNOWN"),
                "label_id": pred
            })
    
    return results

def draw_boxes_on_image(image_path, boxes, labels, output_path, pdf_dimensions=None):
    """
    Draw bounding boxes on image
    
    Args:
        image_path: Path to original image
        boxes: List of bounding boxes [x1, y1, x2, y2] or normalized [0-1000]
        labels: List of labels for each box
        output_path: Path to save result image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)
    
    # Use PDF dimensions if provided, otherwise use image dimensions
    if pdf_dimensions:
        pdf_width = pdf_dimensions["width"]
        pdf_height = pdf_dimensions["height"]
    else:
        pdf_width = img_width
        pdf_height = img_height
    
    # Color map for different labels
    colors = {
        "Title": "red",
        "Text": "blue",
        "List": "green",
        "Table": "orange",
        "Figure": "purple",
        "Caption": "cyan",
        "Section-header": "magenta",
        "Page-header": "yellow",
        "Page-footer": "pink"
    }
    
    # Draw boxes
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        if len(box) == 4:
            # PDF boxes are absolute coordinates, scale to image
            scale_x = img_width / pdf_width
            scale_y = img_height / pdf_height
            
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            # Debug first box
            if idx == 0:
                logger.info(f"PDF dims: {pdf_width}x{pdf_height}, Image dims: {img_width}x{img_height}")
                logger.info(f"Scale: {scale_x:.2f}x{scale_y:.2f}")
                logger.info(f"Box PDF: {box}, Box Image: [{x1},{y1},{x2},{y2}]")
            
            color = colors.get(label, "gray")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label text
            text_y = max(y1 - 15, 0)
            draw.text((x1, text_y), label, fill=color)
    
    # Save
    image.save(output_path)

def process_document(image_paths, output_dir=None, pdf_text_data=None):
    """
    Process multiple images (full document)
    
    Args:
        image_paths: List of image paths
        output_dir: Directory to save visualization (optional)
        pdf_text_data: Text data extracted from PDF (optional)
    
    Returns:
        list: Results for each page
    """
    results = []
    
    # Create output directory for visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths, start=1):
        print(f"Processing page {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Get text data for this page
        text_data = pdf_text_data[i-1] if pdf_text_data and i <= len(pdf_text_data) else None
        
        page_result = process_image_with_layoutlm(image_path, text_data)
        page_result["page_number"] = i
        results.append(page_result)
        
        # Save visualization
        if output_dir:
            output_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, output_filename)
            
            # Use original PDF word boxes with LayoutLM labels
            all_boxes = []
            all_labels = []
            
            if text_data:
                # Map word predictions back to blocks using block_no
                word_to_block = [w["block_no"] for w in text_data.get("words", [])]
                
                # Aggregate predictions per block with confidence scores
                from collections import Counter, defaultdict
                block_label_scores = defaultdict(Counter)
                
                if page_result.get("predictions"):
                    for pred in page_result["predictions"]:
                        word_idx = pred.get("token_id")
                        if word_idx is not None and word_idx < len(word_to_block):
                            block_idx = word_to_block[word_idx]
                            label = pred.get("label", "Text")
                            conf = pred.get("confidence", 0.0)
                            block_label_scores[block_idx][label] += conf
                
                # Draw blocks with confidence-based label
                for block_idx, block in enumerate(text_data.get("blocks", [])):
                    if block["text"].strip():
                        all_boxes.append(block["bbox"])
                        
                        scores = block_label_scores.get(block_idx)
                        if not scores:
                            all_labels.append("Text")
                            continue
                        
                        # Prioritize Section-header and Title
                        text_score = scores.get("Text", 0)
                        section_score = scores.get("Section-header", 0)
                        title_score = scores.get("Title", 0)
                        
                        if section_score > text_score * 0.7:
                            label = "Section-header"
                        elif title_score > text_score * 0.7:
                            label = "Title"
                        else:
                            label = scores.most_common(1)[0][0]
                        
                        all_labels.append(label)
            
            # Draw all boxes with confidence-based aggregation
            if all_boxes:
                pdf_dims = {"width": text_data.get("width"), "height": text_data.get("height")} if text_data else None
                draw_boxes_on_image(image_path, all_boxes, all_labels, output_path, pdf_dims)
                print(f"Saved visualization with {len(all_boxes)} blocks ({len(page_result.get('predictions', []))} words processed)")
    
    return results