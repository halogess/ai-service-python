"""
OCR Processor - Extract text from images using Tesseract OCR
"""
import logging

logger = logging.getLogger(__name__)

def extract_text_with_tesseract(image):
    """
    Extract text and bounding boxes from image using Tesseract OCR
    
    Args:
        image: PIL Image object
    
    Returns:
        dict: {
            "width": image width,
            "height": image height,
            "words": [{"text": str, "bbox": [x0,y0,x1,y1], "block_no": int}],
            "blocks": []
        }
    """
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        logger.error("pytesseract not installed")
        raise
    
    img_width, img_height = image.size
    
    # OCR with Tesseract (Indonesia only)
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='ind')
    
    words = []
    block_no = 0
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if not text:
            continue
        
        x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
        
        # Normalize bbox to 0-1000 scale (same as LayoutLMv3 expects)
        words.append({
            "text": text,
            "bbox": [
                int((x / img_width) * 1000),
                int((y / img_height) * 1000),
                int(((x + w) / img_width) * 1000),
                int(((y + h) / img_height) * 1000)
            ],
            "block_no": block_no
        })
        block_no += 1
    
    logger.info(f"Tesseract OCR extracted {len(words)} words")
    
    return {
        "width": 1000,  # Normalized scale
        "height": 1000,  # Normalized scale
        "words": words,
        "blocks": [],
        "images": [],
        "table_regions_border": []
    }
