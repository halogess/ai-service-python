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
        # Use text from PDF
        words = []
        boxes = []
        
        # Get PDF page dimensions from text_data
        pdf_width = text_data.get("width", image.size[0])
        pdf_height = text_data.get("height", image.size[1])
        
        # Use block-level data from PDF
        for block in text_data.get("blocks", []):
            text = block["text"].strip()
            if text:
                # Get bbox from PDF (absolute coordinates)
                x0, y0, x1, y1 = block["bbox"]
                
                # Split into words and distribute bbox horizontally
                words_in_block = text.split()
                n = len(words_in_block)
                
                for i, word in enumerate(words_in_block):
                    # Distribute bbox horizontally (rough approximation)
                    word_x0 = x0 + (x1 - x0) * i / n
                    word_x1 = x0 + (x1 - x0) * (i + 1) / n
                    word_bbox = [word_x0, y0, word_x1, y1]
                    
                    # Normalize to 0-1000 range
                    normalized_box = [
                        int((word_bbox[0] / pdf_width) * 1000),
                        int((word_bbox[1] / pdf_height) * 1000),
                        int((word_bbox[2] / pdf_width) * 1000),
                        int((word_bbox[3] / pdf_height) * 1000)
                    ]
                    
                    words.append(word)
                    boxes.append(normalized_box)
        
        # Sliding window with overlap
        max_seq_len = 450  # Reduced to account for subword tokenization
        stride = 64  # Overlap tokens
        total_words = len(words)
        
        # Store predictions with probabilities for voting
        token_predictions = {}  # token_id -> list of (label_id, probability)
        
        # Process with sliding window
        start_idx = 0
        window_num = 0
        
        while start_idx < total_words:
            end_idx = min(start_idx + max_seq_len, total_words)
            window_words = words[start_idx:end_idx]
            window_boxes = boxes[start_idx:end_idx]
            
            window_num += 1
            logger.info(f"Processing window {window_num}: tokens {start_idx}-{end_idx} ({len(window_words)} words)")
            
            # Process window
            encoding = processor(image, window_words, boxes=window_boxes, return_tensors="pt", padding="max_length", truncation=True)
            
            # Move to device
            for k, v in encoding.items():
                encoding[k] = v.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**encoding)
            
            # Get predictions and probabilities
            logits = outputs.logits[0]  # [seq_len, num_labels]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)  # [seq_len]
            
            # Map token predictions to word predictions using word_ids
            word_ids = encoding.word_ids(batch_index=0)
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue  # Skip special tokens, padding
                
                if word_id >= len(window_words):
                    continue  # Safety check
                
                global_word_idx = start_idx + word_id
                
                label_id = pred_ids[token_idx].item()
                label_prob = probs[token_idx, label_id].item()
                
                if global_word_idx not in token_predictions:
                    token_predictions[global_word_idx] = []
                
                token_predictions[global_word_idx].append((label_id, label_prob))
            
            # Move window with stride
            if end_idx >= total_words:
                break
            start_idx += (max_seq_len - stride)
        
        # Aggregate predictions using highest probability
        all_predictions = []
        for token_id in sorted(token_predictions.keys()):
            preds = token_predictions[token_id]
            # Choose prediction with highest probability
            best_pred, best_prob = max(preds, key=lambda x: x[1])
            
            all_predictions.append({
                "token_id": token_id,
                "label": model.config.id2label.get(best_pred, "UNKNOWN"),
                "label_id": best_pred,
                "confidence": best_prob
            })
        
        logger.info(f"Processed {total_words} tokens in {window_num} windows with stride {stride}")
        
        # Return results with all predictions
        return {
            "image": image_path,
            "predictions": all_predictions,
            "boxes": boxes  # Original boxes
        }
    
    # If no text data, process without text (layout only)
    encoding = processor(image, return_tensors="pt")
    
    # Move to device
    for k, v in encoding.items():
        encoding[k] = v.to(device)
    
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
                # Map word predictions back to blocks
                word_to_block = []  # word_idx -> block_idx
                word_idx = 0
                
                for block_idx, block in enumerate(text_data.get("blocks", [])):
                    words_in_block = len(block["text"].split())
                    for _ in range(words_in_block):
                        word_to_block.append(block_idx)
                        word_idx += 1
                
                # Aggregate predictions per block
                block_labels = {}  # block_idx -> list of labels
                if page_result.get("predictions"):
                    for pred in page_result["predictions"]:
                        token_id = pred.get("token_id")
                        if token_id is not None and token_id < len(word_to_block):
                            block_idx = word_to_block[token_id]
                            if block_idx not in block_labels:
                                block_labels[block_idx] = []
                            block_labels[block_idx].append(pred.get("label", "Text"))
                
                # Draw blocks with majority vote label
                for block_idx, block in enumerate(text_data.get("blocks", [])):
                    if block["text"].strip():
                        all_boxes.append(block["bbox"])
                        
                        # Use majority vote from word predictions
                        if block_idx in block_labels:
                            labels = block_labels[block_idx]
                            # Majority vote
                            label = max(set(labels), key=labels.count)
                            all_labels.append(label)
                        else:
                            all_labels.append("Text")
            
            # Draw all boxes
            if all_boxes:
                pdf_dims = {"width": text_data.get("width"), "height": text_data.get("height")} if text_data else None
                draw_boxes_on_image(image_path, all_boxes, all_labels, output_path, pdf_dims)
                print(f"Saved visualization with {len(all_boxes)} boxes ({len(page_result.get('predictions', []))} processed by LayoutLM)")
    
    return results