"""PDF content extraction utilities"""

import fitz
from .text_utils import normalize_text
from .config import TOKEN_RE


def iter_pdf_tokens_with_bboxes(page, page_index):
    """Yield (token, bbox, page_index) dengan bbox per token"""
    raw = page.get_text("rawdict")
    blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
    blocks.sort(key=lambda b: (b.get("bbox", [0, 0])[1], b.get("bbox", [0, 0])[0]))

    for b in blocks:
        for line in b.get("lines", []):
            chars = []
            char_boxes = []
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    chars.append(ch.get("c", ""))
                    char_boxes.append(ch.get("bbox"))

            if not chars:
                continue

            line_text = "".join(chars)
            line_text_normalized = normalize_text(line_text, preserve_whitespace=True)
            
            for m in TOKEN_RE.finditer(line_text_normalized):
                token = m.group(0)
                if not token or not token.strip():
                    continue

                start, end = m.span()
                token_boxes = []
                for idx in range(start, end):
                    if idx < len(char_boxes) and char_boxes[idx]:
                        token_boxes.append(char_boxes[idx])
                
                if not token_boxes:
                    continue

                split_tokens = [token]
                
                if len(split_tokens) > 1:
                    char_idx = 0
                    for sub_token in split_tokens:
                        sub_len = len(sub_token)
                        sub_boxes = token_boxes[char_idx:char_idx + sub_len]
                        
                        if sub_boxes:
                            x0 = min(bb[0] for bb in sub_boxes)
                            y0 = min(bb[1] for bb in sub_boxes)
                            x1 = max(bb[2] for bb in sub_boxes)
                            y1 = max(bb[3] for bb in sub_boxes)
                            yield sub_token, [x0, y0, x1, y1], page_index
                        
                        char_idx += sub_len
                else:
                    x0 = min(bb[0] for bb in token_boxes)
                    y0 = min(bb[1] for bb in token_boxes)
                    x1 = max(bb[2] for bb in token_boxes)
                    y1 = max(bb[3] for bb in token_boxes)
                    yield token, [x0, y0, x1, y1], page_index


def extract_pdf_table_cells(pdf_doc, page_idx):
    """Extract table cell bboxes dari PDF page"""
    if page_idx >= len(pdf_doc):
        return []
    
    page = pdf_doc[page_idx]
    tables_result = []
    
    try:
        tables = page.find_tables()
        
        for table in tables.tables:
            cells = table.cells
            cell_texts = []
            cell_bboxes = []
            
            for cell in cells:
                if cell:
                    cell_bboxes.append(cell)
                    cell_rect = fitz.Rect(cell)
                    cell_text = page.get_text("text", clip=cell_rect).strip()
                    cell_texts.append(cell_text)
                else:
                    cell_bboxes.append(None)
                    cell_texts.append("")
            
            tables_result.append({
                'bbox': table.bbox,
                'cells': cell_bboxes,
                'cell_texts': cell_texts,
                'rows': table.row_count,
                'cols': table.col_count
            })
    except Exception:
        pass
    
    return tables_result
