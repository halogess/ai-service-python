"""
Script untuk testing prediksi RAW LayoutLMv3 tanpa post-processing
Ubah IMAGE_PATH di bawah untuk testing gambar lain
"""

# ============================================
# UBAH PATH GAMBAR DI SINI UNTUK TESTING
# ============================================
IMAGE_PATH = "2.jpg"  # Ganti dengan path gambar Anda
# ============================================

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
import json

def main():
    # Validasi file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: File tidak ditemukan: {IMAGE_PATH}")
        print(f"   Silakan ubah IMAGE_PATH di baris 11 script ini")
        return
    
    print(f"📄 Memproses gambar: {IMAGE_PATH}")
    print("=" * 60)
    
    # Load image
    try:
        image = Image.open(IMAGE_PATH).convert("RGB")
        print(f"✅ Gambar berhasil dimuat: {image.size[0]}x{image.size[1]} pixels")
    except Exception as e:
        print(f"❌ Error membuka gambar: {e}")
        return
    
    # Load model
    print("\n🔄 Memuat model LayoutLMv3...")
    model_path = "models/layoutlmv3"
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path, apply_ocr=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model berhasil dimuat di {device}")
    
    # OCR dengan Tesseract
    print("\n🔄 Melakukan OCR dengan Tesseract...")
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        print("❌ Error: pytesseract tidak terinstall")
        print("   Install dengan: pip install pytesseract")
        return
    
    # OCR pada image
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='ind+eng')
    except Exception as e:
        print(f"❌ Error Tesseract: {e}")
        print("   Pastikan Tesseract OCR sudah terinstall:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Atau set path: pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        return
    
    img_width, img_height = image.size
    
    words = []
    boxes = []
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            
            words.append(text)
            boxes.append([
                int((x / img_width) * 1000),
                int((y / img_height) * 1000),
                int(((x + w) / img_width) * 1000),
                int(((y + h) / img_height) * 1000)
            ])
    
    if not words:
        print("❌ Tidak ada teks terdeteksi")
        return
    
    print(f"✅ OCR selesai: {len(words)} kata terdeteksi")
    
    # Process dengan LayoutLMv3
    print("\n🔄 Memproses dengan LayoutLMv3...")
    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    bbox = encoding['bbox'].to(device)
    pixel_values = encoding['pixel_values'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, pixel_values=pixel_values)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1)
    
    # Map predictions ke words
    word_ids = encoding.word_ids(batch_index=0)
    token_predictions = {}
    
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None:
            label_id = pred_ids[token_idx].item()
            label_prob = probs[token_idx, label_id].item()
            
            if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                token_predictions[word_id] = (label_id, label_prob)
    
    # Get label names
    id2label = model.config.id2label
    
    # Build results
    results = []
    for word_id, (pred_id, conf) in sorted(token_predictions.items()):
        if word_id < len(words):
            results.append({
                'text': words[word_id],
                'label': id2label[pred_id],
                'bbox': boxes[word_id],
                'confidence': float(conf)
            })
    
    print(f"✅ Selesai! Ditemukan {len(results)} token")
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 HASIL PREDIKSI RAW LAYOUTLMV3 (TANPA POST-PROCESSING)")
    print("=" * 60)
    
    # Group by label
    label_counts = {}
    for block in results:
        label = block['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n📈 Ringkasan per Label:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label:20s}: {count:3d} token")
    
    # Show detailed results
    print(f"\n📋 Detail Token (Total: {len(results)}):")
    print("-" * 60)
    
    for i, block in enumerate(results[:50], 1):  # Show first 50
        print(f"\n{i}. {block['label']}")
        print(f"   Text: {block['text']}")
        print(f"   BBox: {block['bbox']}")
        print(f"   Confidence: {block['confidence']:.3f}")
    
    if len(results) > 50:
        print(f"\n... dan {len(results) - 50} token lainnya")
    
    # Save to JSON
    output_file = IMAGE_PATH.rsplit('.', 1)[0] + '_raw_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Draw boxes on image
    print("\n🎨 Menggambar bounding boxes...")
    from PIL import ImageDraw, ImageFont
    
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    img_width, img_height = image.size
    
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
    
    for result in results:
        bbox = result['bbox']
        label = result['label']
        
        # Denormalize bbox dari 0-1000 ke pixel
        x0 = int((bbox[0] / 1000) * img_width)
        y0 = int((bbox[1] / 1000) * img_height)
        x1 = int((bbox[2] / 1000) * img_width)
        y1 = int((bbox[3] / 1000) * img_height)
        
        color = colors.get(label, "gray")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0, max(y0 - 15, 0)), label, fill=color)
    
    output_image = IMAGE_PATH.rsplit('.', 1)[0] + '_raw_visualization.jpg'
    draw_image.save(output_image)
    
    print("\n" + "=" * 60)
    print(f"💾 Hasil JSON disimpan ke: {output_file}")
    print(f"🖼️  Visualisasi disimpan ke: {output_image}")
    print("=" * 60)

if __name__ == "__main__":
    main()
