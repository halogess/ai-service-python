
import os
import json
import logging
import difflib
import re
from datetime import datetime
from sqlalchemy.orm import Session
from models import Aturan, Bab, Dokumen, DokumenSection, DokumenPart, DokumenElemen, DokumenElemenVisual, DokumenNote, DokumenFormatText, DokumenFormatParagraf
from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from services.docling_service import DoclingService
from services.docling_fusion_service import DoclingFusionService
from services.visualization_service import VisualizationService
from utils.cross_page_claims import analyze_cross_page_entries
from database import SessionLocal
from services.merging_extraction import (
    MergingExtractionClaimRepairMixin,
    MergingExtractionFusionRepairsMixin,
    MergingExtractionPersistenceMixin,
    MergingExtractionStructuralLabelsMixin,
    MergingExtractionTargetAssignmentMixin,
)

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")
VISUALIZATION_OUTPUT = os.getenv("VISUALIZATION_OUTPUT", "visualization_output")


class MergingExtractionProcessPipelineMixin:


    def process_document(
        self,
        doc_id: int,
        generate_visualizations: bool = False,
        save_to_db: bool = True,
        output_dir: str = None,
        ref_tipe: str = 'dokumen'
    ):
        """
        Process a reference target (dokumen or bab).
        1. Extract PDF content page by page
        2. Validate/Align with OpenXML elements
        3. Run Docling classification
        4. Save results to database (DokumenElemenVisual) [Optional]
        5. Optionally generate visualization images
        
        Args:
            doc_id: Reference ID to process (dokumen_id for dokumen, bab_id for bab/buku)
            generate_visualizations: If True, generate PNG visualizations of alignment and fusion
            save_to_db: If True, commit changes to database. If False, run pipeline but don't save.
            output_dir: If provided, save visualizations to this directory.
            ref_tipe: Reference type ('dokumen', 'bab', or legacy alias 'buku')
        """
        db = SessionLocal()
        try:
            save_debug_json = bool(output_dir)
            ref_id = doc_id
            if ref_id is None:
                logger.error("process_document called without ref_id")
                return False

            canonical_ref_tipe = self._canonical_ref_tipe(ref_tipe)

            if canonical_ref_tipe == 'dokumen':
                doc = db.query(Dokumen).get(ref_id)
                if not doc:
                    logger.error(f"Document {ref_id} not found")
                    return False
                relative_pdf_path = doc.dokumen_pdf_path
            elif canonical_ref_tipe == 'bab':
                bab = db.query(Bab).get(ref_id)
                if not bab:
                    logger.error(f"Bab {ref_id} not found")
                    return False
                relative_pdf_path = bab.bab_pdf_path
            elif canonical_ref_tipe == 'aturan':
                aturan = db.query(Aturan).get(ref_id)
                if not aturan:
                    logger.error(f"Aturan {ref_id} not found")
                    return False
                relative_pdf_path = aturan.aturan_template_pdf_path
            else:
                logger.error(f"Unknown ref_tipe: {ref_tipe}")
                return False

            pdf_path = os.path.join(STORAGE_BASE, relative_pdf_path or "")
            if not relative_pdf_path:
                logger.error(f"{canonical_ref_tipe}:{ref_id} has no PDF path")
                return False

            # 1. Run Docling (Document-level)
            logger.info(f"Running Docling for {canonical_ref_tipe}:{ref_id}")
            if canonical_ref_tipe == 'dokumen':
                docling_results = self.docling_service.classify_document(ref_id)
            else:
                docling_results = self.docling_service.classify_pdf(
                    pdf_path=pdf_path,
                    ref_tipe=canonical_ref_tipe,
                    ref_id=ref_id
                )
            docling_predictions = docling_results.get('predictions_by_page', {}) if docling_results.get('success') else {}

            # 2. Open PDF and iterate pages
            extractor = PDFExtractor(pdf_path)
            extractor.open()
            total_pages = extractor.page_count
            
            logger.info(f"Processing {total_pages} pages for {canonical_ref_tipe}:{ref_id}")

            # Track max_openxml_idx across pages to prevent backward matching
            max_openxml_idx = 0
            pointer_state = {
                'jump_lock_streak': 0,
                'pointer_freeze_streak': 0,
                'last_stable_pointer_max': 0,
            }
            page_vis_payload = {}
            structural_state = self._new_structural_label_state()
            
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

                page_docling_preds = docling_predictions.get(str(page_num), [])
                extraction_items = self._annotate_picture_visual_items(
                    extraction_items,
                    page_docling_preds
                )
                footnote_groups = []
                if canonical_ref_tipe == 'dokumen':
                    footnote_groups, footnote_item_idxs = self._build_footnote_groups(
                        extraction_items, page_docling_preds, ref_id, page_num
                    )
                    if footnote_item_idxs:
                        extraction_items = [
                            item for idx, item in enumerate(extraction_items)
                            if idx not in footnote_item_idxs
                        ]
                
                # Perform Alignment with cross-page tracking
                alignment_result = self.alignment_service.align(
                    ref_id, page_num, extraction_items,
                    page_width, page_height, total_pages, 
                    min_openxml_idx=max_openxml_idx,  # Use previous page's max
                    ref_tipe=canonical_ref_tipe
                )
                
                if alignment_result['success']:
                    page_debug = alignment_result.get('page_debug') or {}
                    pointer_update = self.alignment_service._resolve_next_page_pointer(
                        max_openxml_idx,
                        alignment_result,
                        pointer_state=pointer_state
                    )
                    max_openxml_idx = pointer_update['next_min_openxml_idx']
                    pointer_state = pointer_update['pointer_state']
                    logger.debug(f"Page {page_num}: max_openxml_idx updated to {max_openxml_idx}")
                    
                    alignments = alignment_result['final_alignments']
                    header_footer_units = alignment_result.get('header_footer_units', [])
                    section_data = alignment_result.get('page_debug', {}).get('section_data')
                    if canonical_ref_tipe == 'dokumen':
                        page_docling_preds, footnote_entries = self._assign_docling_footnotes(
                            db, ref_id, page_num, page_docling_preds, footnote_groups
                        )
                    else:
                        footnote_entries = []
                    
                    # Save alignment results with header_footer_units for proper Docling fusion
                    fused_results, fusion_debug = self._save_alignment_results(
                        db, 
                        alignments, 
                        page_docling_preds,
                        footnote_entries=footnote_entries,
                        header_footer_units=header_footer_units,
                        section_data=section_data,
                        doc_id=ref_id,
                        page_num=page_num,
                        structural_state=structural_state
                    )
                    if fusion_debug:
                        page_debug.update(fusion_debug)
                    orphan_chart_visual_count = sum(
                        1 for unit in (alignment_result.get('unaligned_pdf_units') or [])
                        if unit.get('is_chart_visual')
                    )
                    page_debug['orphan_chart_visual_count'] = orphan_chart_visual_count
                    if orphan_chart_visual_count:
                        logger.warning(
                            "Page %s has orphan chart visual units: %s",
                            page_num,
                            orphan_chart_visual_count
                        )

                    if save_to_db or generate_visualizations or save_debug_json:
                        payload = {
                            'alignments': alignments,
                            'fused_results': fused_results,
                            'header_footer_units': header_footer_units,
                            'section_data': section_data,
                            'page_height': page_height,
                        }
                        page_vis_payload[page_num] = payload

                    if generate_visualizations:
                        all_pdf_units = self.alignment_service._flatten_extraction_items(extraction_items)
                        unaligned_units = alignment_result.get('unaligned_pdf_units', [])
                        unfused_units = self._collect_unfused_pdf_units(
                            all_pdf_units,
                            fused_results,
                            unaligned_units
                        )
                        unaligned_for_vis = unaligned_units + unfused_units

                        page_vis_payload[page_num]['unaligned_pdf_units'] = unaligned_for_vis
                        page_vis_payload[page_num]['raw_docling'] = page_docling_preds
                
            extractor.close()

            if page_vis_payload and (save_to_db or generate_visualizations or save_debug_json):
                try:
                    duplicate_analysis = self._analyze_duplicate_openxml_elements(page_vis_payload)
                    duplicate_element_ids = {
                        elem_id
                        for elem_id, analysis in (duplicate_analysis or {}).items()
                        if analysis.get('is_invalid_duplicate')
                    }
                    for page_num in sorted(page_vis_payload):
                        payload = page_vis_payload[page_num]
                        alignments = payload.get('alignments')
                        removed_duplicate_element_ids = set()
                        if alignments and duplicate_element_ids:
                            alignments, removed_duplicate_element_ids = self._merge_duplicate_units_with_neighbors(
                                alignments,
                                duplicate_element_ids
                            )
                            payload['alignments'] = alignments
                        self._sync_fused_bboxes_with_alignments(
                            payload.get('fused_results'),
                            alignments,
                            removed_duplicate_element_ids
                        )

                    claim_resolution_stats = self._resolve_document_visual_claims(page_vis_payload)
                    covered_claim_clear_stats = self._clear_same_page_covered_claims(page_vis_payload)
                    invalid_duplicate_repair_stats = self._repair_invalid_duplicate_claims_to_local_targets(
                        db,
                        canonical_ref_tipe,
                        ref_id,
                        page_vis_payload
                    )
                    claim_reassignment_stats = self._repair_same_page_null_claims(page_vis_payload)
                    target_assignment_stats = self._assign_null_results_to_unclaimed_targets(
                        db,
                        canonical_ref_tipe,
                        ref_id,
                        page_vis_payload
                    )
                    bookmark_backfill_count = self._backfill_document_bookmark_proxies(
                        db,
                        canonical_ref_tipe,
                        ref_id,
                        page_vis_payload
                    )
                    body_text_backfill_count = self._backfill_document_text_proxies(
                        db,
                        canonical_ref_tipe,
                        ref_id,
                        page_vis_payload
                    )
                    redundant_proxy_stats = self._drop_redundant_same_page_proxies(page_vis_payload)
                    post_backfill_repair_stats = self._repair_same_page_null_claims(page_vis_payload)
                    header_footer_repair_stats = self._repair_document_header_footer_claims(
                        db,
                        canonical_ref_tipe,
                        ref_id,
                        page_vis_payload
                    )
                    visual_chain_repair_stats = self._repair_adjacent_page_visual_chains(page_vis_payload)
                    post_visual_chain_proxy_stats = self._drop_redundant_same_page_proxies(page_vis_payload)
                    logger.info(
                        "Resolved document-level claims for %s:%s cleared=%s affected_pages=%s covered_claims_cleared=%s invalid_duplicate_reassigned=%s invalid_duplicate_elements=%s same_page_reassigned=%s merged_fragments=%s synthetic_dropped=%s target_assigned=%s note_assigned=%s bookmark_backfill=%s body_text_backfill=%s redundant_proxy_dropped=%s post_backfill_reassigned=%s post_backfill_merged=%s post_backfill_dropped=%s header_footer_reassigned=%s visual_chain_reassigned=%s visual_chain_dropped=%s post_visual_chain_proxy_dropped=%s",
                        canonical_ref_tipe,
                        ref_id,
                        claim_resolution_stats.get('cleared_claims', 0),
                        claim_resolution_stats.get('affected_pages', 0),
                        covered_claim_clear_stats.get('cleared_rows', 0),
                        invalid_duplicate_repair_stats.get('reassigned_rows', 0),
                        invalid_duplicate_repair_stats.get('repaired_elements', 0),
                        claim_reassignment_stats.get('reassigned_claims', 0),
                        claim_reassignment_stats.get('merged_fragment_rows', 0),
                        claim_reassignment_stats.get('dropped_synthetic_rows', 0),
                        target_assignment_stats.get('assigned_body_targets', 0),
                        target_assignment_stats.get('assigned_note_targets', 0),
                        bookmark_backfill_count,
                        body_text_backfill_count,
                        redundant_proxy_stats.get('dropped_rows', 0),
                        post_backfill_repair_stats.get('reassigned_claims', 0),
                        post_backfill_repair_stats.get('merged_fragment_rows', 0),
                        post_backfill_repair_stats.get('dropped_synthetic_rows', 0),
                        header_footer_repair_stats.get('reassigned_rows', 0),
                        visual_chain_repair_stats.get('reassigned_rows', 0),
                        visual_chain_repair_stats.get('dropped_rows', 0),
                        post_visual_chain_proxy_stats.get('dropped_rows', 0),
                    )

                    if save_to_db:
                        for page_num in sorted(page_vis_payload):
                            payload = page_vis_payload[page_num]
                            payload['fused_results'] = self._replace_visual_records(
                                db,
                                canonical_ref_tipe,
                                ref_id,
                                page_num,
                                payload.get('fused_results'),
                                structural_state=structural_state,
                                section_data=payload.get('section_data'),
                                apply_duplicate_claim_guard=False
                            )

                    if generate_visualizations:
                        for page_num in sorted(page_vis_payload):
                            payload = page_vis_payload[page_num]
                            alignments = payload.get('alignments')
                            duplicate_units = self._collect_duplicate_units_for_page(
                                alignments,
                                duplicate_analysis,
                                page_num
                            )

                            vis_paths = self.visualization_service.visualize_page(
                                pdf_path=pdf_path,
                                page_num=page_num - 1,  # 0-based for visualization
                                alignments=payload.get('alignments'),
                                fused_results=payload.get('fused_results'),
                                header_footer_units=payload.get('header_footer_units'),
                                unaligned_pdf_units=payload.get('unaligned_pdf_units'),
                                duplicate_mapping_units=duplicate_units,
                                doc_id=ref_id,
                                output_dir_override=output_dir
                            )
                            logger.info(f"Page {page_num}: Generated visualizations - {list(vis_paths.keys())}")
                    if save_debug_json:
                        os.makedirs(output_dir, exist_ok=True)
                        for page_num in sorted(page_vis_payload):
                            payload = page_vis_payload[page_num]
                            json_path = os.path.join(output_dir, f"page_{page_num}_fusion_data.json")
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump({
                                    'page': page_num,
                                    'doc_id': ref_id if canonical_ref_tipe == 'dokumen' else None,
                                    'ref_tipe': canonical_ref_tipe,
                                    'ref_id': ref_id,
                                    'fused_results': payload.get('fused_results'),
                                    'raw_docling': payload.get('raw_docling'),
                                    'alignments': payload.get('alignments')
                                }, f, indent=2, ensure_ascii=False)
                except Exception as vis_err:
                    logger.warning(f"Visualization/JSON save failed - {vis_err}")
            
            if save_to_db:
                db.commit()
                logger.info(f"Committed changes to database for {canonical_ref_tipe}:{ref_id}")
            else:
                logger.info(f"Skipping database commit for {canonical_ref_tipe}:{ref_id} (save_to_db=False)")
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing {canonical_ref_tipe}:{ref_id}: {e}", exc_info=True)
            db.rollback()
            raise e  # Re-raise to let caller handle/report
        finally:
            db.close()
