"""
Test alignment visualization without database
"""

import os
import sys
from PIL import Image, ImageDraw

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.alignment_visualizer import AlignmentVisualizer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_visualization():
    """Test visualization with mock data"""
    
    # Create a test image (white background)
    img_width, img_height = 2480, 3508  # A4 at 300 DPI
    img = Image.new('RGB', (img_width, img_height), 'white')
    test_image_path = 'test_page.png'
    img.save(test_image_path)
    
    print(f"Created test image: {test_image_path}")
    
    # Mock alignment result with VALID bboxes
    # Bbox format: [x0, y0, x1, y1] in PDF points (72 DPI)
    # Will be scaled to 300 DPI (multiply by 300/72 = 4.166...)
    
    alignment_result = {
        'success': True,
        'page': 1,
        'alignments': [
            {
                'element_id': 1,
                'element_sequence': 1,
                'element_type': 'paragraph',
                'merged_bbox': [100, 100, 400, 150],  # Valid bbox in PDF points
                'matched_pdf_units': [
                    {
                        'pdf_unit_id': 'pdf_0',
                        'item_idx': 0,
                        'item_type': 'group',
                        'text': 'Test text 1',
                        'bbox': [100, 100, 200, 120],  # Valid bbox
                        'score': 1.0
                    },
                    {
                        'pdf_unit_id': 'pdf_1',
                        'item_idx': 1,
                        'item_type': 'group',
                        'text': 'Test text 2',
                        'bbox': [210, 100, 400, 120],  # Valid bbox
                        'score': 1.0
                    }
                ]
            },
            {
                'element_id': 2,
                'element_sequence': 2,
                'element_type': 'paragraph',
                'merged_bbox': [100, 200, 400, 250],  # Valid bbox
                'matched_pdf_units': [
                    {
                        'pdf_unit_id': 'pdf_2',
                        'item_idx': 2,
                        'item_type': 'group',
                        'text': 'Another paragraph',
                        'bbox': [100, 200, 400, 220],  # Valid bbox
                        'score': 1.0
                    }
                ]
            }
        ],
        'unaligned_pdf_units': [
            {
                'unit_id': 'pdf_3',
                'item_idx': 3,
                'item_type': 'group',
                'text': 'Unaligned text',
                'bbox': [100, 300, 300, 320],  # Valid bbox
                'is_cell': False
            }
        ],
        'header_footer_units': [
            {
                'unit_id': 'pdf_4',
                'item_idx': 4,
                'item_type': 'group',
                'text': 'Header text',
                'bbox': [100, 50, 300, 70],  # Valid bbox
                'is_cell': False,
                'zone': 'header'
            }
        ],
        'stats': {
            'aligned_count': 2,
            'unaligned_count': 1
        }
    }
    
    print("\n" + "="*80)
    print("MOCK ALIGNMENT RESULT")
    print("="*80)
    print(f"Alignments: {len(alignment_result['alignments'])}")
    print(f"Unaligned: {len(alignment_result['unaligned_pdf_units'])}")
    print(f"Header/Footer: {len(alignment_result['header_footer_units'])}")
    
    # Test visualization
    visualizer = AlignmentVisualizer()
    output_path = 'test_alignment_output.png'
    
    print(f"\nDrawing alignments...")
    visualizer.draw_alignments_on_page(test_image_path, alignment_result, output_path)
    
    print(f"\nOutput saved to: {output_path}")
    print("Check if boxes are visible!")
    
    # Cleanup
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"\nCleaned up: {test_image_path}")

if __name__ == "__main__":
    test_visualization()
