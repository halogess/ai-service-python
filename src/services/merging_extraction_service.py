
import json
import logging
from sqlalchemy.orm import Session
from models import Dokumen, DokumenElemen, DokumenSection, DokumenPart
from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from services.docling_service import DoclingService
from database import SessionLocal

logger = logging.getLogger(__name__)

class MergingExtractionService:
    def __init__(self):
        self.alignment_service = AlignmentService()
        self.docling_service = DoclingService()

    def process_document(self, doc_id: int):
        """
        Process a document:
        1. Extract PDF content page by page
        2. Validate/Align with OpenXML elements
        3. Run Docling classification
        4. Save results to database (DokumenElemen)
        """
        db = SessionLocal()
        try:
            doc = db.query(Dokumen).get(doc_id)
            if not doc:
                logger.error(f"Document {doc_id} not found")
                return False
            
            pdf_path = doc.dokumen_pdf_path
            if not pdf_path:
                logger.error(f"Document {doc_id} has no PDF path")
                return False

            # 1. Run Docling (Document-level)
            logger.info(f"Running Docling for doc {doc_id}")
            docling_results = self.docling_service.classify_document(doc_id)
            docling_predictions = docling_results.get('predictions_by_page', {}) if docling_results.get('success') else {}

            # 2. Open PDF and iterate pages
            extractor = PDFExtractor(pdf_path)
            extractor.open()
            total_pages = extractor.page_count
            
            logger.info(f"Processing {total_pages} pages for doc {doc_id}")

            # Get page dimensions
            # We need these for alignment service
            # (AlignmentService usually fetches sections to get dimensions, 
            # but we can pass actual PDF dimensions if needed, or let it rely on sections)
            
            for page_num in range(1, total_pages + 1):
                # Extract PDF data
                # PDFExtractor uses 0-based indexing
                pdf_page = extractor.get_page(page_num - 1)
                page_width = pdf_page.rect.width
                page_height = pdf_page.rect.height
                
                extraction_data = extractor.extract_merging_data(page_num - 1)
                
                # Identify items list from extraction data
                # extract_merging_data returns dict with 'char_groups', 'shapes', etc.
                # AlignmentService expects a flattened list or specific structure?
                # AlignmentService._flatten_extraction_items expects a list of items with 'type', 'data', 'bbox'.
                # But extract_merging_data returns a DICT of collected items.
                # We need to CONVERT extraction_data dict to a list of items for AlignmentService.
                # This logic was in pdf_extraction.js (frontend) in legacy.
                # I need to replicate `processMergingResponse` from frontend JS here.
                
                extraction_items = self._transform_extraction_data_to_items(extraction_data)
                
                # Perform Alignment
                # Passes 0 as min_openxml_idx for now (TODO: tracking across pages)
                alignment_result = self.alignment_service.align(
                    doc_id, page_num, extraction_items, 
                    page_width, page_height, total_pages, 
                    min_openxml_idx=0 
                )
                
                if alignment_result['success']:
                    self._save_alignment_results(db, alignment_result['final_alignments'], docling_predictions.get(str(page_num), []))
                
            extractor.close()
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error processing doc {doc_id}: {e}", exc_info=True)
            db.rollback()
            return False
        finally:
            db.close()

    def _transform_extraction_data_to_items(self, data):
        """Transform extraction dict into list of typed items for alignment."""
        items = []
        
        # Char Groups -> 'group'
        for g in data.get('char_groups', []):
            items.append({
                'type': 'group',
                'bbox': g.get('bbox') or g.get('merged_bbox'),
                'data': {'text': g.get('text', '')}
            })
            
        # Basic Tables -> 'table'
        for t in data.get('basic_tables', []):
            items.append({
                'type': 'table',
                'bbox': t.get('bbox'),
                'data': {'rows': t.get('rows', [])}
            })
            
        # Hline Tables -> 'hline_table'
        for t in data.get('hline_tables', []):
            items.append({
                'type': 'hline_table',
                'bbox': t.get('bbox'),
                'data': {
                    'rows': t.get('rows', []),
                    'cells': [] # Legacy structure might have cells at top too
                }
            })
            
        # Shapes -> 'shape'
        for s in data.get('shapes', []):
            items.append({
                'type': 'shape',
                'bbox': s.get('bbox'),
                'data': {'text': s.get('text', '')}
            })
            
        # Images -> 'image'
        for img in data.get('page_images', []):
            items.append({
                'type': 'image',
                'bbox': img.get('bbox'),
                'data': {}
            })
            
        # Sort by vertical reading order (top-down, then left-right)
        # Using simple sort
        def get_y(item):
            b = item.get('bbox')
            return b[1] if b else 0
        def get_x(item):
            b = item.get('bbox')
            return b[0] if b else 0
            
        items.sort(key=lambda x: (get_y(x), get_x(x)))
        return items

    def _save_alignment_results(self, db, alignments, docling_predictions):
        """Update DokumenElemen with alignment metadata."""
        for align in alignments:
            elem_id = align.get('element_id')
            if not elem_id: continue
            
            elem = db.query(DokumenElemen).get(elem_id)
            if elem:
                # Load existing JSON tree
                try:
                    tree = json.loads(elem.delemen_json_tree) if isinstance(elem.delemen_json_tree, str) else (elem.delemen_json_tree or {})
                except:
                    tree = {}
                
                # Add alignment metadata
                # Note: This overwrites existing alignment data
                tree['alignment'] = {
                    'matched_pdf_units': align.get('matched_pdf_units', []),
                    'merged_bbox': align.get('merged_bbox'),
                    'is_table': align.get('is_table', False),
                    'timestamp': 'auto-generated'
                }
                
                # Simple Docling integration: check overlap with merged_bbox
                bbox = align.get('merged_bbox')
                if bbox and docling_predictions:
                    preds = []
                    bx0, by0, bx1, by1 = bbox
                    pad = 5
                    for pred in docling_predictions:
                        p = pred.get('bbox') # [x0, y0, x1, y1]
                        if p:
                            # Intersection over Union or simple containment?
                            # Legacy check: center inside
                            cx = (p[0]+p[2])/2
                            cy = (p[1]+p[3])/2
                            if bx0-pad <= cx <= bx1+pad and by0-pad <= cy <= by1+pad:
                                preds.append(pred)
                    
                    if preds:
                        tree['docling_predictions'] = preds

                # Save back to DB
                elem.delemen_json_tree = json.dumps(tree, ensure_ascii=False)
                # db.add(elem) # Attached to session
