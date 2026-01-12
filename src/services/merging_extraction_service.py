
import os
import json
import logging
from sqlalchemy.orm import Session
from models import Dokumen, DokumenElemen, DokumenSection, DokumenPart
from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from services.docling_service import DoclingService
from services.docling_fusion_service import DoclingFusionService
from services.visualization_service import VisualizationService
from database import SessionLocal

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")
VISUALIZATION_OUTPUT = os.getenv("VISUALIZATION_OUTPUT", "visualization_output")

class MergingExtractionService:
    def __init__(self):
        self.alignment_service = AlignmentService()
        self.docling_service = DoclingService()
        self.fusion_service = DoclingFusionService()
        self.visualization_service = VisualizationService(output_dir=VISUALIZATION_OUTPUT)

    def process_document(self, doc_id: int, generate_visualizations: bool = False, save_to_db: bool = True, output_dir: str = None):
        """
        Process a document:
        1. Extract PDF content page by page
        2. Validate/Align with OpenXML elements
        3. Run Docling classification
        4. Save results to database (DokumenElemen) [Optional]
        5. Optionally generate visualization images
        
        Args:
            doc_id: Document ID to process
            generate_visualizations: If True, generate PNG visualizations of alignment and fusion
            save_to_db: If True, commit changes to database. If False, run pipeline but don't save.
            output_dir: If provided, save visualizations to this directory.
        """
        db = SessionLocal()
        try:
            doc = db.query(Dokumen).get(doc_id)
            if not doc:
                logger.error(f"Document {doc_id} not found")
                return False
            
            pdf_path = os.path.join(STORAGE_BASE, doc.dokumen_pdf_path)
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

            # Track max_openxml_idx across pages to prevent backward matching
            max_openxml_idx = 0
            
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
                
                # Perform Alignment with cross-page tracking
                alignment_result = self.alignment_service.align(
                    doc_id, page_num, extraction_items, 
                    page_width, page_height, total_pages, 
                    min_openxml_idx=max_openxml_idx  # Use previous page's max
                )
                
                if alignment_result['success']:
                    # Update cross-page tracking from alignment result
                    max_openxml_idx = alignment_result.get('max_openxml_idx', max_openxml_idx)
                    logger.debug(f"Page {page_num}: max_openxml_idx updated to {max_openxml_idx}")
                    
                    alignments = alignment_result['final_alignments']
                    header_footer_units = alignment_result.get('header_footer_units', [])
                    section_data = alignment_result.get('page_debug', {}).get('section_data')
                    page_docling_preds = docling_predictions.get(str(page_num), [])
                    
                    # Save alignment results with header_footer_units for proper Docling fusion
                    fused_results = self._save_alignment_results(
                        db, 
                        alignments, 
                        page_docling_preds,
                        header_footer_units=header_footer_units,
                        section_data=section_data
                    )
                    
                    # Generate visualizations if enabled
                    if generate_visualizations:
                        try:
                            vis_paths = self.visualization_service.visualize_page(
                                pdf_path=pdf_path,
                                page_num=page_num - 1,  # 0-based for visualization
                                alignments=alignments,
                                fused_results=fused_results,
                                doc_id=doc_id,
                                output_dir_override=output_dir
                            )
                            logger.info(f"Page {page_num}: Generated visualizations - {list(vis_paths.keys())}")
                            
                            # Also save the fused results to JSON for debugging
                            if vis_paths:
                                # Get output dir from one of the paths
                                json_output_dir = os.path.dirname(list(vis_paths.values())[0])
                                
                                # Save fused results
                                json_path = os.path.join(json_output_dir, f"page_{page_num}_fusion_data.json")
                                with open(json_path, 'w', encoding='utf-8') as f:
                                    json.dump({
                                        'page': page_num,
                                        'doc_id': doc_id,
                                        'fused_results': fused_results,
                                        'raw_docling': page_docling_preds,
                                        'alignments': alignments
                                    }, f, indent=2, ensure_ascii=False)
                                    
                        except Exception as vis_err:
                            logger.warning(f"Page {page_num}: Visualization/JSON save failed - {vis_err}")
                
            extractor.close()
            
            if save_to_db:
                db.commit()
                logger.info(f"Committed changes to database for doc {doc_id}")
            else:
                logger.info(f"Skipping database commit for doc {doc_id} (save_to_db=False)")
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing doc {doc_id}: {e}", exc_info=True)
            db.rollback()
            raise e  # Re-raise to let caller handle/report
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
                'data': {
                    'text': s.get('text', ''),
                    'image_bbox': s.get('image_bbox')
                }
            })
            
        # Images -> 'image'
        for img in data.get('page_images', []):
            items.append({
                'type': 'image',
                'bbox': img.get('bbox'),
                'data': {}
            })

        # Sort by reading order (line-aware) to match legacy frontend
        # Items are on the same line if >=30% Y overlap (based on smaller height)
        from functools import cmp_to_key

        def compare_items(a, b):
            if not a.get('bbox') or not b.get('bbox'):
                return 0

            y_a0, y_a1 = a['bbox'][1], a['bbox'][3]
            y_b0, y_b1 = b['bbox'][1], b['bbox'][3]
            height_a = y_a1 - y_a0
            height_b = y_b1 - y_b0

            overlap_start = max(y_a0, y_b0)
            overlap_end = min(y_a1, y_b1)
            overlap_amount = max(0, overlap_end - overlap_start)
            smaller_height = min(height_a, height_b)

            overlap_ratio = (overlap_amount / smaller_height) if smaller_height > 0 else 0
            is_same_line = overlap_ratio >= 0.30

            if is_same_line:
                return -1 if a['bbox'][0] < b['bbox'][0] else (1 if a['bbox'][0] > b['bbox'][0] else 0)
            return -1 if y_a0 < y_b0 else (1 if y_a0 > y_b0 else 0)

        items.sort(key=cmp_to_key(compare_items))
        return items

    def _save_alignment_results(self, db, alignments, docling_predictions, header_footer_units=None, section_data=None):
        """
        Update DokumenElemen with alignment metadata and fused Docling predictions.
        
        Args:
            db: Database session
            alignments: List of alignment results
            docling_predictions: List of Docling predictions for this page
            header_footer_units: Optional list of header/footer PDF units
            section_data: Optional section data with margin info
        """
        # Use fusion service for proper Docling-Alignment integration
        if section_data:
            self.fusion_service.section_data = section_data
        
        # Perform fusion
        fused_results = self.fusion_service.fuse_alignments_with_docling(
            alignments=alignments,
            header_footer_units=header_footer_units or [],
            docling_predictions=docling_predictions or []
        )
        
        # Build lookup: element_id -> list of fused results
        fused_by_element = {}
        for result in fused_results:
            elem_id = result.get('element_id')
            if elem_id:
                if elem_id not in fused_by_element:
                    fused_by_element[elem_id] = []
                fused_by_element[elem_id].append(result)
        
        # Update each alignment's DokumenElemen
        for align in alignments:
            elem_id = align.get('element_id')
            if not elem_id:
                continue
            
            elem = db.query(DokumenElemen).get(elem_id)
            if not elem:
                continue
            
            # Load existing JSON tree
            try:
                tree = json.loads(elem.delemen_json_tree) if isinstance(elem.delemen_json_tree, str) else (elem.delemen_json_tree or {})
            except:
                tree = {}
            
            # Add alignment metadata
            tree['alignment'] = {
                'matched_pdf_units': align.get('matched_pdf_units', []),
                'merged_bbox': align.get('merged_bbox'),
                'is_table': align.get('is_table', False),
                'timestamp': 'auto-generated'
            }
            
            # Add fused Docling results for this element
            element_fusions = fused_by_element.get(elem_id, [])
            if element_fusions:
                tree['docling_fusion'] = [{
                    'label': f.get('label'),
                    'bbox': f.get('bbox'),
                    'overlap': f.get('overlap'),
                    'merged_count': f.get('merged_count', 1),
                    'is_picture_merge': f.get('is_picture_merge', False),
                    'docling_label': f.get('docling_label')
                } for f in element_fusions]
                
                # Also set primary label (first/best match)
                tree['docling_label'] = element_fusions[0].get('label')
            
            # Save back to DB
            elem.delemen_json_tree = json.dumps(tree, ensure_ascii=False)
        
        # Return fused results for visualization
        return fused_results
