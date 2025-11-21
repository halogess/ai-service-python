import pytesseract
from dataclasses import dataclass
from typing import List
from processing.pdf_loader import PageImage
from utils.bbox import BBox

@dataclass
class OCRWord:
    text: str
    bbox: BBox
    confidence: float
    page: int

@dataclass
class OCRResult:
    words: List[OCRWord]
    page_count: int

def extract_text(page_images: List[PageImage]) -> OCRResult:
    all_words = []
    
    for page_img in page_images:
        # Get OCR data with bounding boxes
        ocr_data = pytesseract.image_to_data(page_img.image, output_type=pytesseract.Output.DICT)
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            if text:  # Skip empty text
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                conf = ocr_data['conf'][i]
                
                # Scale coordinates back to original page size
                scale_x = page_img.page_width / page_img.image.width
                scale_y = page_img.page_height / page_img.image.height
                
                bbox = BBox(
                    left=x * scale_x,
                    top=y * scale_y,
                    right=(x + w) * scale_x,
                    bottom=(y + h) * scale_y
                )
                
                all_words.append(OCRWord(
                    text=text,
                    bbox=bbox,
                    confidence=conf,
                    page=page_img.page_index
                ))
    
    return OCRResult(words=all_words, page_count=len(page_images))