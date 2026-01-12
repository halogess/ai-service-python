"""
Test visualization of alignment and fusion results
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set environment variables before imports
os.environ['DB_HOST'] = 'host.docker.internal'
os.environ['DB_PORT'] = '3307'
os.environ['DB_NAME'] = 'db_korektor_buku'
os.environ['DB_USER'] = 'jessica'
os.environ['DB_PASSWORD'] = 'pass123'
os.environ['VOLUME_BASE_PATH'] = 'E:/docker-volumes/validasi-ta'

from services.visualization_service import VisualizationService, visualize_alignment_results
from services.alignment_service import AlignmentService
from services.docling_fusion_service import DoclingFusionService
from services.pdf_extraction_service import PDFExtractor
from database import SessionLocal
from models import Dokumen, Antrian

def test_visualization():
    """Test visualization with real document"""
    print("=" * 60)
    print("TESTING VISUALIZATION SERVICE")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # Get a document from antrian
        task = db.query(Antrian).filter(Antrian.antrian_worker == 'struktur').first()
        if not task or not task.dokumen_id:
            print("[ERROR] No struktur task found in antrian")
            return
        
        doc_id = task.dokumen_id
        doc = db.query(Dokumen).get(doc_id)
        if not doc:
            print(f"[ERROR] Document {doc_id} not found")
            return
        
        print(f"\nDocument: {doc.dokumen_nama} (ID: {doc_id})")
        
        pdf_path = os.path.join(os.environ['VOLUME_BASE_PATH'], doc.dokumen_pdf_path)
        print(f"PDF Path: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"[ERROR] PDF not found: {pdf_path}")
            return
        
        print("\n[1] Extracting PDF...")
        extractor = PDFExtractor(pdf_path)
        extractor.open()
        
        # Get page 1 dimensions
        page = extractor.get_page(0)
        page_width = page.rect.width
        page_height = page.rect.height
        total_pages = extractor.page_count
        
        # Extract data
        extraction_data = extractor.extract_merging_data(0)
        extractor.close()
        
        # Transform to items
        from services.merging_extraction_service import MergingExtractionService
        merge_service = MergingExtractionService()
        extraction_items = merge_service._transform_extraction_data_to_items(extraction_data)
        
        print(f"   Extracted {len(extraction_items)} items from page 1")
        
        print("\n[2] Running alignment...")
        alignment_service = AlignmentService()
        result = alignment_service.align(doc_id, 1, extraction_items, page_width, page_height, total_pages)
        
        if not result.get('success'):
            print("[ERROR] Alignment failed")
            return
        
        alignments = result.get('final_alignments', [])
        header_footer_units = result.get('header_footer_units', [])
        
        print(f"   {len(alignments)} alignments, {len(header_footer_units)} header/footer units")
        
        print("\n[3] Running Docling fusion...")
        # Create mock Docling predictions for testing (since we don't want to run Docling)
        mock_docling = [
            {'label': 'title', 'bbox': [250, 110, 350, 140], 'confidence': 0.95},
            {'label': 'section_header', 'bbox': [230, 135, 400, 165], 'confidence': 0.88},
            {'label': 'paragraph', 'bbox': [110, 190, 520, 320], 'confidence': 0.92},
        ]
        
        fusion_service = DoclingFusionService()
        fused_results = fusion_service.fuse_alignments_with_docling(
            alignments=alignments,
            header_footer_units=header_footer_units,
            docling_predictions=mock_docling
        )
        
        print(f"   {len(fused_results)} fused results")
        for i, f in enumerate(fused_results[:5]):
            print(f"      [{i+1}] {f.get('label')} - elem_id={f.get('element_id')}, overlap={f.get('overlap', 0):.0%}")
        
        print("\n[4] Generating visualizations...")
        vis_service = VisualizationService(output_dir='visualization_output')
        
        saved_paths = vis_service.visualize_page(
            pdf_path=pdf_path,
            page_num=0,
            alignments=alignments,
            fused_results=fused_results,
            doc_id=doc_id
        )
        
        print("\n   Saved images:")
        for name, path in saved_paths.items():
            print(f"      {name}: {path}")
        
        print("\n" + "=" * 60)
        print("VISUALIZATION TEST COMPLETED!")
        print("=" * 60)
        print(f"\nCheck folder: visualization_output/doc_{doc_id}/")
        
    finally:
        db.close()

if __name__ == '__main__':
    test_visualization()
