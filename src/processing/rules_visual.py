from dataclasses import dataclass
from typing import List, Dict, Any
from processing.layout_model import LayoutBlock
from utils.bbox import BBox

@dataclass
class ErrorItem:
    code: str
    pesan_singkat: str
    penjelasan: str
    severity: str
    page: int
    bbox: BBox
    meta: Dict[str, Any] = None

class VisualRulesEvaluator:
    def __init__(self):
        # Default margin configuration (in points)
        self.margin_left = 72    # 1 inch
        self.margin_right = 72
        self.margin_top = 72
        self.margin_bottom = 72
        self.max_caption_distance = 50  # points
    
    def evaluate(self, blocks: List[LayoutBlock], page_sizes: List[tuple]) -> List[ErrorItem]:
        errors = []
        
        # Group blocks by page
        pages_blocks = {}
        for block in blocks:
            if block.page not in pages_blocks:
                pages_blocks[block.page] = []
            pages_blocks[block.page].append(block)
        
        for page_num, page_blocks in pages_blocks.items():
            page_width, page_height = page_sizes[page_num] if page_num < len(page_sizes) else (595, 842)  # A4 default
            
            # Check margin overflow
            errors.extend(self._check_margin_overflow(page_blocks, page_width, page_height))
            
            # Check page without paragraph
            errors.extend(self._check_page_without_paragraph(page_blocks, page_num))
            
            # Check caption distance from figures
            errors.extend(self._check_caption_distance(page_blocks, page_num))
        
        return errors
    
    def _check_margin_overflow(self, blocks: List[LayoutBlock], page_width: float, page_height: float) -> List[ErrorItem]:
        errors = []
        
        for block in blocks:
            if block.bbox.left < self.margin_left:
                errors.append(ErrorItem(
                    code="MARGIN_OVERFLOW_LEFT",
                    pesan_singkat="Konten melewati margin kiri",
                    penjelasan=f"Blok {block.label} melewati batas margin kiri yang diizinkan",
                    severity="major",
                    page=block.page,
                    bbox=block.bbox
                ))
            
            if block.bbox.right > (page_width - self.margin_right):
                errors.append(ErrorItem(
                    code="MARGIN_OVERFLOW_RIGHT",
                    pesan_singkat="Konten melewati margin kanan",
                    penjelasan=f"Blok {block.label} melewati batas margin kanan yang diizinkan",
                    severity="major",
                    page=block.page,
                    bbox=block.bbox
                ))
            
            if block.bbox.top < self.margin_top:
                errors.append(ErrorItem(
                    code="MARGIN_OVERFLOW_TOP",
                    pesan_singkat="Konten melewati margin atas",
                    penjelasan=f"Blok {block.label} melewati batas margin atas yang diizinkan",
                    severity="major",
                    page=block.page,
                    bbox=block.bbox
                ))
            
            if block.bbox.bottom > (page_height - self.margin_bottom):
                errors.append(ErrorItem(
                    code="MARGIN_OVERFLOW_BOTTOM",
                    pesan_singkat="Konten melewati margin bawah",
                    penjelasan=f"Blok {block.label} melewati batas margin bawah yang diizinkan",
                    severity="major",
                    page=block.page,
                    bbox=block.bbox
                ))
        
        return errors
    
    def _check_page_without_paragraph(self, blocks: List[LayoutBlock], page_num: int) -> List[ErrorItem]:
        errors = []
        
        has_paragraph = any(block.label == 'paragraph' for block in blocks)
        has_content = any(block.label in ['figure', 'table'] for block in blocks)
        
        if has_content and not has_paragraph:
            errors.append(ErrorItem(
                code="PAGE_WITHOUT_PARAGRAPH",
                pesan_singkat="Halaman tanpa paragraf teks",
                penjelasan="Halaman hanya berisi objek (gambar/tabel) tanpa teks paragraf",
                severity="minor",
                page=page_num,
                bbox=BBox(0, 0, 100, 100)  # Dummy bbox for page-level error
            ))
        
        return errors
    
    def _check_caption_distance(self, blocks: List[LayoutBlock], page_num: int) -> List[ErrorItem]:
        errors = []
        
        figures = [b for b in blocks if b.label == 'figure']
        captions = [b for b in blocks if b.label == 'caption']
        
        for caption in captions:
            closest_figure = None
            min_distance = float('inf')
            
            for figure in figures:
                distance = caption.bbox.distance_to(figure.bbox)
                if distance < min_distance:
                    min_distance = distance
                    closest_figure = figure
            
            if closest_figure and min_distance > self.max_caption_distance:
                errors.append(ErrorItem(
                    code="CAPTION_TOO_FAR_FROM_FIGURE",
                    pesan_singkat="Caption terlalu jauh dari gambar",
                    penjelasan=f"Jarak caption dengan gambar ({min_distance:.1f} pt) melebihi batas maksimal ({self.max_caption_distance} pt)",
                    severity="minor",
                    page=page_num,
                    bbox=caption.bbox,
                    meta={"distance": min_distance, "max_allowed": self.max_caption_distance}
                ))
        
        return errors