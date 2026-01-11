"""
Production-Ready API untuk OpenXML Alignment
Siap digunakan di frontend/backend
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from openxml_alignment import OpenXMLAligner, PDFWord


class AlignmentAPI:
    """API wrapper untuk frontend integration"""
    
    def __init__(self):
        self.aligner = None
        self.last_error = None
        self._last_page_heights = None
    
    def align_document(
        self, 
        openxml_elements: List[Dict], 
        pdf_words: List[Dict],
        page_heights: Optional[Dict[int, float]] = None
    ) -> Dict:
        """
        Main API untuk alignment
        
        Args:
            openxml_elements: List dari dokumen_elemen JSON
            pdf_words: List dari PDF words [{'text': str, 'bbox': [x0,y0,x1,y1], 'page': int}] (0-based page)
            page_heights: Optional dict {page_num: height} (0-based page)
        
        Returns:
            {
                'success': bool,
                'data': {
                    'aligned_words': [...],
                    'element_bboxes': {...},
                    'stats': {...}
                },
                'error': str | None
            }
        """
        try:
            # Validate input
            if not openxml_elements:
                return self._error_response("openxml_elements is empty")
            
            if not pdf_words:
                return self._error_response("pdf_words is empty")
            
            # Create aligner with page heights
            self._last_page_heights = page_heights
            self.aligner = OpenXMLAligner(openxml_elements, pdf_words, page_heights)
            
            # Perform alignment
            aligned_words = self.aligner.align()
            
            # Get stats
            stats = self.aligner.get_coverage_stats()
            
            # Get element bboxes
            element_bboxes = {}
            for element_id in self.aligner.element_map.keys():
                bbox = self.aligner.get_element_bbox(element_id)
                if bbox:
                    element_bboxes[element_id] = {
                        'x0': bbox[0],
                        'y0': bbox[1],
                        'x1': bbox[2],
                        'y1': bbox[3]
                    }
            
            # Convert aligned words to dict
            aligned_words_dict = [
                {
                    'text': w.text,
                    'normalized': w.normalized,
                    'bbox': {
                        'x0': w.bbox[0],
                        'y0': w.bbox[1],
                        'x1': w.bbox[2],
                        'y1': w.bbox[3]
                    },
                    'page': w.page,
                    'element_id': w.assigned_element_id,
                    'confidence': w.confidence
                }
                for w in aligned_words
            ]
            
            return {
                'success': True,
                'data': {
                    'aligned_words': aligned_words_dict,
                    'element_bboxes': element_bboxes,
                    'stats': stats
                },
                'error': None
            }
            
        except Exception as e:
            self.last_error = str(e)
            return self._error_response(f"Alignment failed: {str(e)}")
    
    def get_element_details(self, element_id: int) -> Dict:
        """
        Get details untuk satu element
        
        Returns:
            {
                'success': bool,
                'data': {
                    'element_id': int,
                    'words': [...],
                    'bbox': {...},
                    'word_count': int
                },
                'error': str | None
            }
        """
        try:
            if not self.aligner:
                return self._error_response("No alignment performed yet")
            
            words = self.aligner.get_element_words(element_id)
            bbox = self.aligner.get_element_bbox(element_id)
            
            words_dict = [
                {
                    'text': w.text,
                    'normalized': w.normalized,
                    'bbox': {
                        'x0': w.bbox[0],
                        'y0': w.bbox[1],
                        'x1': w.bbox[2],
                        'y1': w.bbox[3]
                    },
                    'page': w.page,
                    'confidence': w.confidence
                }
                for w in words
            ]
            
            bbox_dict = None
            if bbox:
                bbox_dict = {
                    'x0': bbox[0],
                    'y0': bbox[1],
                    'x1': bbox[2],
                    'y1': bbox[3]
                }
            
            return {
                'success': True,
                'data': {
                    'element_id': element_id,
                    'words': words_dict,
                    'bbox': bbox_dict,
                    'word_count': len(words)
                },
                'error': None
            }
            
        except Exception as e:
            return self._error_response(f"Failed to get element details: {str(e)}")
    
    def validate_alignment(self, min_coverage: float = 0.95) -> Dict:
        """
        Validate alignment quality
        
        Returns:
            {
                'success': bool,
                'data': {
                    'is_valid': bool,
                    'coverage': float,
                    'issues': [...]
                },
                'error': str | None
            }
        """
        try:
            if not self.aligner:
                return self._error_response("No alignment performed yet")
            
            stats = self.aligner.get_coverage_stats()
            coverage = stats['coverage']
            
            issues = []
            
            # Check coverage
            if coverage < min_coverage:
                issues.append({
                    'type': 'LOW_COVERAGE',
                    'message': f"Coverage {coverage:.2%} is below threshold {min_coverage:.2%}",
                    'severity': 'ERROR'
                })
            
            # Check for unassigned words
            unassigned = [w for w in self.aligner.pdf_words if w.assigned_element_id is None]
            if unassigned:
                issues.append({
                    'type': 'UNASSIGNED_WORDS',
                    'message': f"{len(unassigned)} words are not assigned",
                    'severity': 'WARNING',
                    'details': [w.text for w in unassigned[:5]]
                })
            
            # Check for low confidence assignments
            low_conf = [w for w in self.aligner.pdf_words if w.confidence < 0.7]
            if low_conf:
                issues.append({
                    'type': 'LOW_CONFIDENCE',
                    'message': f"{len(low_conf)} words have low confidence",
                    'severity': 'INFO'
                })
            
            is_valid = coverage >= min_coverage and not any(i['severity'] == 'ERROR' for i in issues)
            
            return {
                'success': True,
                'data': {
                    'is_valid': is_valid,
                    'coverage': coverage,
                    'issues': issues,
                    'stats': stats
                },
                'error': None
            }
            
        except Exception as e:
            return self._error_response(f"Validation failed: {str(e)}")
    
    def export_results(self, output_path: str) -> Dict:
        """
        Export hasil alignment ke JSON file
        
        Returns:
            {'success': bool, 'error': str | None}
        """
        try:
            if not self.aligner:
                return self._error_response("No alignment performed yet")
            
            result = self.align_document(
                self.aligner.openxml_elements,
                self.aligner.pdf_words_raw,
                page_heights=getattr(self.aligner, 'page_heights', None) or self._last_page_heights
            )
            
            if not result['success']:
                return result
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result['data'], f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'error': None,
                'message': f"Results exported to {output_path}"
            }
            
        except Exception as e:
            return self._error_response(f"Export failed: {str(e)}")
    
    def _error_response(self, message: str) -> Dict:
        """Helper untuk error response"""
        return {
            'success': False,
            'data': None,
            'error': message
        }


# Convenience functions untuk quick usage
def align_from_files(
    openxml_json_path: str,
    pdf_words_json_path: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Quick alignment dari file paths
    
    Args:
        openxml_json_path: Path ke dokumen_elemen JSON
        pdf_words_json_path: Path ke PDF words JSON
        output_path: Optional path untuk export hasil
    
    Returns:
        Alignment result dict
    """
    # Load files
    with open(openxml_json_path, 'r', encoding='utf-8') as f:
        openxml_elements = json.load(f)
    
    with open(pdf_words_json_path, 'r', encoding='utf-8') as f:
        pdf_words = json.load(f)
    
    # Align
    api = AlignmentAPI()
    result = api.align_document(openxml_elements, pdf_words)
    
    # Export if requested
    if output_path and result['success']:
        api.export_results(output_path)
    
    return result


def align_from_pdf(
    openxml_json_path: str,
    pdf_path: str,
    output_path: Optional[str] = None
) -> Dict:
    """
    Alignment langsung dari PDF file
    
    Args:
        openxml_json_path: Path ke dokumen_elemen JSON
        pdf_path: Path ke PDF file
        output_path: Optional path untuk export hasil
    
    Returns:
        Alignment result dict
    """
    from extract_pdf_words import extract_with_merge, extract_page_heights_from_pdf
    
    # Extract PDF words
    pdf_words = extract_with_merge(pdf_path, None, merge_threshold=2.0)
    
    # Extract page heights
    page_heights = extract_page_heights_from_pdf(pdf_path)
    
    # Load OpenXML
    with open(openxml_json_path, 'r', encoding='utf-8') as f:
        openxml_elements = json.load(f)
    
    # Align
    api = AlignmentAPI()
    result = api.align_document(openxml_elements, pdf_words, page_heights)
    
    # Export if requested
    if output_path and result['success']:
        api.export_results(output_path)
    
    return result


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python alignment_api.py <openxml_json> <pdf_words_json> [output_json]")
        print("  python alignment_api.py <openxml_json> <pdf_file> [output_json] --from-pdf")
        sys.exit(1)
    
    openxml_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    from_pdf = '--from-pdf' in sys.argv
    
    if from_pdf:
        result = align_from_pdf(openxml_path, input_path, output_path)
    else:
        result = align_from_files(openxml_path, input_path, output_path)
    
    if result['success']:
        print("✓ Alignment successful!")
        print(f"  Coverage: {result['data']['stats']['coverage']:.2%}")
        print(f"  Total words: {result['data']['stats']['total_words']}")
        print(f"  Assigned: {result['data']['stats']['assigned_words']}")
    else:
        print(f"✗ Alignment failed: {result['error']}")
        sys.exit(1)
