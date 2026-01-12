"""
Simulasi alignment visualization dengan bab2.pdf
"""

import os
import sys
import json
import fitz
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.pdf_extraction_service import PDFExtractor
from utils.alignment_visualizer import AlignmentVisualizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_with_bab2():
    """Simulasi dengan bab2.pdf tanpa database"""
    
    pdf_path = 'bab2.pdf'
    if not os.path.exists(pdf_path):
        print(f"[X] File tidak ditemukan: {pdf_path}")
        return
    
    print(f"[OK] Found: {pdf_path}")
    
    # Create output directory
    output_dir = 'test_output_bab2'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract first page
    print("\n[*] Extracting page 1...")
    
    with PDFExtractor(pdf_path) as extractor:
        page = extractor.get_page(0)
        
        # Convert to image (300 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_path = os.path.join(output_dir, 'page_001.png')
        pix.save(img_path)
        print(f"[OK] Saved image: {img_path}")
        
        # Extract merging data
        page_data = extractor.extract_merging_data(0)
        
        # Transform to items
        items = []
        
        for g in page_data.get('char_groups', []):
            items.append({
                'type': 'group',
                'bbox': g.get('merged_bbox'),
                'data': {'text': g.get('text', '')}
            })
        
        for t in page_data.get('basic_tables', []):
            items.append({
                'type': 'table',
                'bbox': t.get('bbox'),
                'data': {'rows': t.get('rows', [])}
            })
        
        for s in page_data.get('shapes', []):
            items.append({
                'type': 'shape',
                'bbox': s.get('bbox'),
                'data': {'text': s.get('text', '')}
            })
        
        for img in page_data.get('page_images', []):
            items.append({
                'type': 'image',
                'bbox': img.get('bbox'),
                'data': {}
            })
        
        items.sort(key=lambda x: (x['bbox'][1] if x.get('bbox') else 0, x['bbox'][0] if x.get('bbox') else 0))
        
        print(f"[OK] Extracted {len(items)} items")
        
        # Save extraction
        extraction_file = os.path.join(output_dir, 'extraction.json')
        with open(extraction_file, 'w', encoding='utf-8') as f:
            json.dump({
                'page': 1,
                'page_width': page.rect.width,
                'page_height': page.rect.height,
                'items': items
            }, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {extraction_file}")
        
        # Create mock alignment result (simulate successful alignment)
        print("\n[*] Creating mock alignment result...")
        
        # Take first 5 items as "aligned"
        aligned_items = items[:5]
        unaligned_items = items[5:10] if len(items) > 5 else []
        
        alignments = []
        for idx, item in enumerate(aligned_items):
            if item.get('bbox'):
                alignments.append({
                    'element_id': idx + 1,
                    'element_sequence': idx + 1,
                    'element_type': item['type'],
                    'merged_bbox': item['bbox'],
                    'matched_pdf_units': [{
                        'pdf_unit_id': f'pdf_{idx}',
                        'item_idx': idx,
                        'item_type': item['type'],
                        'text': item['data'].get('text', '[content]'),
                        'bbox': item['bbox'],
                        'score': 1.0
                    }]
                })
        
        unaligned_units = []
        for idx, item in enumerate(unaligned_items):
            if item.get('bbox'):
                unaligned_units.append({
                    'unit_id': f'pdf_{idx + 5}',
                    'item_idx': idx + 5,
                    'item_type': item['type'],
                    'text': item['data'].get('text', '[content]'),
                    'bbox': item['bbox'],
                    'is_cell': False
                })
        
        alignment_result = {
            'success': True,
            'page': 1,
            'alignments': alignments,
            'unaligned_pdf_units': unaligned_units,
            'header_footer_units': [],
            'stats': {
                'aligned_count': len(alignments),
                'unaligned_count': len(unaligned_units)
            }
        }
        
        # Save alignment result
        alignment_file = os.path.join(output_dir, 'alignment.json')
        with open(alignment_file, 'w', encoding='utf-8') as f:
            json.dump(alignment_result, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {alignment_file}")
        
        print(f"\n[*] Alignment Summary:")
        print(f"  - Alignments: {len(alignments)}")
        print(f"  - Unaligned: {len(unaligned_units)}")
        
        # Visualize
        print("\n[*] Drawing alignments on image...")
        visualizer = AlignmentVisualizer()
        output_path = os.path.join(output_dir, 'page_001_aligned.png')
        
        visualizer.draw_alignments_on_page(img_path, alignment_result, output_path)
        
        print(f"\n[OK] DONE! Check output:")
        print(f"   [DIR] {output_dir}/")
        print(f"   [IMG] page_001.png (original)")
        print(f"   [IMG] page_001_aligned.png (with alignment boxes)")
        print(f"   [JSON] extraction.json")
        print(f"   [JSON] alignment.json")

if __name__ == "__main__":
    simulate_with_bab2()
