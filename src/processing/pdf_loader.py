import fitz
from dataclasses import dataclass
from typing import List
from PIL import Image
import numpy as np

@dataclass
class PageImage:
    page_index: int
    image: Image.Image
    page_width: float
    page_height: float

def load_pages(pdf_path: str) -> List[PageImage]:
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        
        # Convert to PIL Image
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
        
        pages.append(PageImage(
            page_index=page_num,
            image=img,
            page_width=page.rect.width,
            page_height=page.rect.height
        ))
    
    doc.close()
    return pages