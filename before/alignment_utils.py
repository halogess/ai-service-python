"""
Utility tambahan untuk OpenXML Alignment
Handle edge cases dan dynamic configuration
"""

from typing import List, Dict, Tuple
from openxml_alignment import OpenXMLAligner, InsertTokenClassifier, PDFWord, TokenAligner


class DynamicPageHeightClassifier(InsertTokenClassifier):
    """Classifier dengan page height dinamis dari PDF"""
    
    def __init__(self, page_heights: Dict[int, float]):
        """
        Args:
            page_heights: {page_num: height}
        """
        super().__init__(page_heights)
    
    def _is_page_number(self, word: PDFWord) -> bool:
        """Deteksi page number dengan dynamic page height"""
        if not word.normalized.isdigit():
            return False
        
        page_height = self.page_heights.get(word.page, 842)  # Default A4
        return word.bbox[3] > page_height * 0.9


class EnhancedAligner(OpenXMLAligner):
    """Aligner dengan page height tracking"""
    
    def __init__(self, openxml_elements: List[Dict], pdf_words: List[Dict], 
                 page_heights: Dict[int, float] = None):
        super().__init__(openxml_elements, pdf_words, page_heights=page_heights)
    
    def align(self) -> List[PDFWord]:
        """Align dengan dynamic classifier"""
        # gunakan pipeline baru di base class (repaired stream + span mapping)
        aligned = super().align()

        # override classifier kalau page heights tersedia
        if self.page_heights:
            DynamicPageHeightClassifier(self.page_heights).classify_and_assign(self.pdf_words, self.element_map)

        return aligned


def extract_page_heights_from_pdf(pdf_path: str) -> Dict[int, float]:
    """Ekstrak page heights dari PDF (0-based page index)."""
    import fitz
    
    doc = fitz.open(pdf_path)
    heights = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        heights[page_num] = page.rect.height
    
    doc.close()
    return heights


def validate_alignment_quality(aligner: OpenXMLAligner, min_similarity: float = 0.8) -> Dict:
    """
    Validasi kualitas alignment per element
    Returns: {element_id: {'similarity': float, 'status': str}}
    """
    from rapidfuzz.distance import Levenshtein
    from openxml_alignment import OpenXMLFlattener
    
    results = {}
    
    for element_id, element in aligner.element_map.items():
        # Get OpenXML text
        tokens, _ = OpenXMLFlattener.flatten_element(element)
        openxml_text = ' '.join(tokens)
        
        # Get PDF text (filter out <EMPTY>)
        words = aligner.get_element_words(element_id)
        if not words:
            results[element_id] = {'similarity': 0.0, 'status': 'NO_WORDS'}
            continue
        
        pdf_text = ' '.join(
            w.normalized
            for w in sorted(words, key=lambda x: x.index)
            if w.normalized != '<EMPTY>'
        )
        
        # Calculate similarity
        similarity = Levenshtein.normalized_similarity(openxml_text, pdf_text)
        
        status = 'OK' if similarity >= min_similarity else 'LOW_SIMILARITY'
        results[element_id] = {'similarity': similarity, 'status': status}
    
    return results


def find_problematic_elements(validation_results: Dict, threshold: float = 0.8) -> List[int]:
    """Find elements dengan similarity rendah"""
    return [
        eid for eid, result in validation_results.items()
        if result['similarity'] < threshold
    ]


def export_debug_info(aligner: OpenXMLAligner, output_path: str):
    """Export debug info untuk troubleshooting"""
    import json
    
    debug_info = {
        'stats': aligner.get_coverage_stats(),
        'elements': []
    }
    
    for element_id, element in aligner.element_map.items():
        words = aligner.get_element_words(element_id)
        bbox = aligner.get_element_bbox(element_id)
        
        debug_info['elements'].append({
            'element_id': element_id,
            'type': element.get('dokumen_elemen_type'),
            'sequence': element.get('dokumen_elemen_sequence'),
            'word_count': len(words),
            'bbox': bbox,
            'words': [
                {
                    'text': w.text,
                    'normalized': w.normalized,
                    'confidence': w.confidence
                }
                for w in words[:10]  # Sample
            ]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)
    
    print(f"Debug info exported to {output_path}")
