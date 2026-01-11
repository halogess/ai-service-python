"""Build final alignment result"""

import re

# Decorative token patterns that should not be counted as unaligned
DECORATIVE_PATTERNS = [
    r'^[•●○◦▪▫‣⁃]$',  # Bullet characters
    r'^\d{1,3}$',  # Page numbers (1-3 digits alone)
    r'^[.\-–—:;,]$',  # Single punctuation
    r'^[ivxIVX]+$',  # Roman numerals (page numbers)
]

COMPILED_PATTERNS = [re.compile(p) for p in DECORATIVE_PATTERNS]


def is_decorative_token(text, bbox=None, page_height=842):
    """Check if token is likely decorative (page number, bullet, punctuation)
    
    Args:
        text: Token text
        bbox: Token bounding box (optional, for position-based detection)
        page_height: Page height for position checks
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Check against decorative patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.match(text):
            return True
    
    # Check position - page numbers often at top or bottom
    if bbox and len(text) <= 3 and text.isdigit():
        y_center = (bbox[1] + bbox[3]) / 2
        # If in top 50px or bottom 50px, likely page number
        if y_center < 50 or y_center > page_height - 50:
            return True
    
    return False


def build_alignment_result(final_aligned: list, pdf_tokens: list = None, pdf_bboxes: list = None, pdf_pages: list = None, used_pdf_indices: set = None) -> dict:
    """Build final result dictionary from aligned elements
    
    Args:
        final_aligned: List of aligned elements
        pdf_tokens: List of all PDF tokens
        pdf_bboxes: List of all PDF token bboxes
        pdf_pages: List of all PDF token pages
        used_pdf_indices: Set of used PDF token indices
        
    Returns:
        dict with aligned_words, unaligned_elements, unaligned_tokens, and stats
    """
    # Separate unaligned elements
    unaligned_elements = [e for e in final_aligned if e.get('unaligned')]
    aligned_only = [e for e in final_aligned if not e.get('unaligned')]
    
    # Calculate unaligned tokens (excluding decorative)
    unaligned_tokens = []
    decorative_count = 0
    
    if pdf_tokens and used_pdf_indices is not None:
        for i in range(len(pdf_tokens)):
            if i not in used_pdf_indices:
                bbox = pdf_bboxes[i]
                text = pdf_tokens[i]
                
                # Check if decorative but still include it
                is_decorative = is_decorative_token(text, bbox)
                if is_decorative:
                    decorative_count += 1
                
                # Include ALL unaligned tokens (decorative ones are flagged)
                unaligned_tokens.append({
                    "text": text,
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "page": pdf_pages[i],
                    "pdf_index": i,
                    "is_decorative": is_decorative
                })

    
    return {
        "aligned_words": aligned_only,
        "unaligned_elements": unaligned_elements,
        "unaligned_tokens": unaligned_tokens,
        "stats": {
            "total_words": len(aligned_only),
            "assigned_words": len(aligned_only),
            "unaligned_count": len(unaligned_elements),
            "unaligned_tokens_count": len(unaligned_tokens),
            "decorative_tokens_filtered": decorative_count,
            "coverage": 1.0,
        },
    }

