from dataclasses import dataclass
from typing import List
from processing.ocr import OCRResult
from utils.bbox import BBox

@dataclass
class LayoutBlock:
    page: int
    bbox: BBox
    label: str
    text: str
    confidence: float = 0.0

class LayoutModel:
    def __init__(self):
        # Simplified implementation - in real scenario, load LayoutLMv3 model
        self.labels = ['paragraph', 'title', 'figure', 'table', 'equation', 'caption']
    
    def predict_layout(self, ocr_result: OCRResult) -> List[LayoutBlock]:
        blocks = []
        
        # Group words by proximity to form blocks (simplified heuristic)
        for page_num in range(ocr_result.page_count):
            page_words = [w for w in ocr_result.words if w.page == page_num]
            
            if not page_words:
                continue
            
            # Simple grouping by vertical proximity
            current_block_words = []
            current_y = None
            
            for word in sorted(page_words, key=lambda w: (w.bbox.top, w.bbox.left)):
                if current_y is None or abs(word.bbox.top - current_y) < 20:
                    current_block_words.append(word)
                    current_y = word.bbox.top
                else:
                    if current_block_words:
                        blocks.append(self._create_block(current_block_words, page_num))
                    current_block_words = [word]
                    current_y = word.bbox.top
            
            if current_block_words:
                blocks.append(self._create_block(current_block_words, page_num))
        
        return blocks
    
    def _create_block(self, words: List, page_num: int) -> LayoutBlock:
        if not words:
            return None
        
        # Calculate bounding box for the block
        min_left = min(w.bbox.left for w in words)
        min_top = min(w.bbox.top for w in words)
        max_right = max(w.bbox.right for w in words)
        max_bottom = max(w.bbox.bottom for w in words)
        
        bbox = BBox(min_left, min_top, max_right, max_bottom)
        text = ' '.join(w.text for w in words)
        
        # Simple heuristic for label classification
        label = 'paragraph'  # Default
        if len(text) < 50 and any(word in text.lower() for word in ['gambar', 'tabel', 'figure', 'table']):
            label = 'caption'
        elif len(text) < 30:
            label = 'title'
        
        return LayoutBlock(
            page=page_num,
            bbox=bbox,
            label=label,
            text=text
        )