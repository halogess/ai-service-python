"""
Test visualization with existing test data (no database required)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.visualization_service import VisualizationService
from services.docling_fusion_service import DoclingFusionService

def test_visualization_standalone():
    """Test visualization with mock data"""
    print("=" * 60)
    print("TESTING VISUALIZATION SERVICE (STANDALONE)")
    print("=" * 60)
    
    # Use the BAB II PDF from test
    pdf_path = r"E:/docker-volumes/validasi-ta/dokumen/222117032/332/pdf/BAB II.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found: {pdf_path}")
        return
    
    print(f"\nPDF: {pdf_path}")
    
    # Create mock alignment data (based on real test output)
    mock_alignments = [
        {
            'element_id': 63853,
            'element_sequence': 1,
            'element_type': 'paragraph',
            'merged_bbox': [287.09, 114.10, 336.14, 131.63],
            'matched_pdf_units': [
                {'text': 'BAB', 'bbox': [287.09, 114.10, 310.0, 131.63]},
                {'text': 'II', 'bbox': [312.0, 114.10, 336.14, 131.63]}
            ]
        },
        {
            'element_id': 63854,
            'element_sequence': 2,
            'element_type': 'paragraph',
            'merged_bbox': [234.86, 141.84, 388.52, 159.37],
            'matched_pdf_units': [
                {'text': 'TEORI', 'bbox': [234.86, 141.84, 280.0, 159.37]},
                {'text': 'PENUNJANG', 'bbox': [285.0, 141.84, 388.52, 159.37]}
            ]
        },
        {
            'element_id': 63856,
            'element_sequence': 3,
            'element_type': 'paragraph',
            'merged_bbox': [113.47, 196.87, 510.47, 313.39],
            'matched_pdf_units': [
                {'text': 'Dalam bab ini akan dibahas...', 'bbox': [113.47, 196.87, 510.47, 220.0]},
                {'text': 'teori-teori yang menunjang...', 'bbox': [113.47, 225.0, 510.47, 250.0]},
                {'text': 'penelitian dan pengembangan...', 'bbox': [113.47, 255.0, 510.47, 280.0]}
            ]
        }
    ]
    
    # Create mock Docling predictions
    mock_docling = [
        {'label': 'title', 'bbox': [280, 110, 345, 135], 'confidence': 0.95},
        {'label': 'section_header', 'bbox': [225, 138, 395, 165], 'confidence': 0.88},
        {'label': 'paragraph', 'bbox': [105, 190, 520, 320], 'confidence': 0.92},
    ]
    
    # Create mock header/footer
    mock_header_footer = [
        {'bbox': [280, 750, 330, 780], 'text': '12', 'zone': 'footer'}
    ]
    
    print("\n[1] Running fusion...")
    fusion_service = DoclingFusionService()
    fused_results = fusion_service.fuse_alignments_with_docling(
        alignments=mock_alignments,
        header_footer_units=mock_header_footer,
        docling_predictions=mock_docling
    )
    
    print(f"   {len(fused_results)} fused results:")
    for i, f in enumerate(fused_results):
        print(f"      [{i+1}] {f.get('label'):<15} elem_id={f.get('element_id')}, overlap={f.get('overlap', 0):.0%}")
    
    print("\n[2] Generating visualizations...")
    vis_service = VisualizationService(output_dir='visualization_output')
    
    saved_paths = vis_service.visualize_page(
        pdf_path=pdf_path,
        page_num=0,
        alignments=mock_alignments,
        fused_results=fused_results,
        doc_id=332
    )
    
    print("\n[3] Saved images:")
    for name, path in saved_paths.items():
        abs_path = os.path.abspath(path)
        print(f"   {name}: {abs_path}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION TEST COMPLETED!")
    print("=" * 60)
    print(f"\nCheck folder: visualization_output/doc_332/")

if __name__ == '__main__':
    test_visualization_standalone()
