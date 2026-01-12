"""
Debug script untuk memeriksa alignment visualization
"""

import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.alignment_service import AlignmentService
from database import SessionLocal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_alignment():
    """Debug alignment result structure"""
    
    # Test dengan dokumen ID 1 (sesuaikan dengan dokumen yang ada)
    doc_id = 1
    
    db = SessionLocal()
    try:
        # Simulasi extraction result sederhana
        extraction_results = [
            {
                'page': 1,
                'page_width': 595,
                'page_height': 842,
                'items': [
                    {
                        'type': 'group',
                        'bbox': [100, 100, 200, 120],
                        'data': {'text': 'Test text'}
                    }
                ]
            }
        ]
        
        alignment_service = AlignmentService()
        alignment_results = alignment_service.align_document(extraction_results, doc_id)
        
        print("\n" + "="*80)
        print("ALIGNMENT RESULTS STRUCTURE")
        print("="*80)
        
        for page_result in alignment_results:
            page_num = page_result.get('page', 0)
            print(f"\n--- PAGE {page_num} ---")
            print(f"Success: {page_result.get('success')}")
            print(f"Stats: {page_result.get('stats')}")
            
            alignments = page_result.get('alignments', [])
            print(f"\nTotal alignments: {len(alignments)}")
            
            if alignments:
                print("\nFirst alignment structure:")
                first = alignments[0]
                print(f"  Keys: {list(first.keys())}")
                print(f"  element_id: {first.get('element_id')}")
                print(f"  element_sequence: {first.get('element_sequence')}")
                print(f"  element_type: {first.get('element_type')}")
                print(f"  merged_bbox: {first.get('merged_bbox')}")
                
                matched_units = first.get('matched_pdf_units', [])
                print(f"  matched_pdf_units count: {len(matched_units)}")
                
                if matched_units:
                    print(f"\n  First matched unit:")
                    unit = matched_units[0]
                    print(f"    Keys: {list(unit.keys())}")
                    print(f"    text: {unit.get('text')}")
                    print(f"    bbox: {unit.get('bbox')}")
                    print(f"    item_idx: {unit.get('item_idx')}")
            
            unaligned = page_result.get('unaligned_pdf_units', [])
            print(f"\nUnaligned PDF units: {len(unaligned)}")
            if unaligned:
                print(f"  Type of first unaligned: {type(unaligned[0])}")
                if isinstance(unaligned[0], dict):
                    print(f"  First unaligned keys: {list(unaligned[0].keys())}")
                    print(f"  First unaligned bbox: {unaligned[0].get('bbox')}")
            
            header_footer = page_result.get('header_footer_units', [])
            print(f"\nHeader/Footer units: {len(header_footer)}")
            if header_footer:
                print(f"  Type of first h/f: {type(header_footer[0])}")
                if isinstance(header_footer[0], dict):
                    print(f"  First h/f keys: {list(header_footer[0].keys())}")
                    print(f"  First h/f bbox: {header_footer[0].get('bbox')}")
        
        print("\n" + "="*80)
        print("CHECKING VISUALIZER EXPECTATIONS")
        print("="*80)
        
        # Check what visualizer expects
        print("\nVisualizer expects:")
        print("  alignment_result.get('alignments', []) -> list of alignments")
        print("    Each alignment should have:")
        print("      - 'merged_bbox': [x0, y0, x1, y1]")
        print("      - 'matched_pdf_units': list of units with 'bbox'")
        print("  alignment_result.get('unaligned_pdf_units', []) -> list of dicts with 'bbox'")
        print("  alignment_result.get('header_footer_units', []) -> list of dicts with 'bbox'")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    debug_alignment()
