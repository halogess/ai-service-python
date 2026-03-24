
import os
import json
import logging
import difflib
import re
from datetime import datetime
from sqlalchemy.orm import Session
from models import Bab, Dokumen, DokumenSection, DokumenPart, DokumenElemen, DokumenElemenVisual, DokumenNote, DokumenFormatText, DokumenFormatParagraf
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


class MergingExtractionService(
    MergingExtractionTargetAssignmentMixin,
    MergingExtractionStructuralLabelsMixin,
    MergingExtractionFusionRepairsMixin,
    MergingExtractionClaimRepairMixin,
    MergingExtractionPersistenceMixin,
):
    FOOTNOTE_LABELS = {"footnote"}
    FOOTNOTE_MATCH_MIN_RATIO = 0.55
    FOOTNOTE_OVERLAP_THRESHOLD = 0.3
    FOOTNOTE_LOG_PATH = os.path.join("logs", "footnote_matches.txt")
    DUPLICATE_SEQUENCE_GAP_THRESHOLD = 2
    SHORT_DUPLICATE_UNIT_LEN = 12
    BAB_TITLE_REGEX = re.compile(r'^\s*bab\b', re.IGNORECASE)
    # Support heading numbering with optional spaces around dots:
    # e.g. "3.1 Judul", "3. 1 Judul", "3 . 1 Judul"
    SUBCHAPTER_TITLE_REGEX = re.compile(r'^\s*\d+(?:\s*\.\s*\d+)+\.?(?:\s+.+)?$', re.IGNORECASE)
    CODE_TITLE_HEADER_REGEX = re.compile(
        r'\b(?:segmen\s*program|listing|algoritma|algorithm|kode\s*program|script)\b',
        re.IGNORECASE
    )
    CODE_LINE_NUMBER_REGEX = re.compile(r'^\s*\d{1,3}\s*[:.)]\s*')
    CODE_TEXT_HINT_REGEX = re.compile(
        r'\b(?:def|class|return|if|else|elif|for|while|import|from|public|private|protected|'
        r'static|void|int|float|double|string|bool|yield|await|select|insert|update|delete|'
        r'create|join|where)\b',
        re.IGNORECASE
    )
    # Keep numeric list marker strict, and avoid treating "3. 1 ..." as list.
    LIST_NUMERIC_REGEX = re.compile(r'^\s*\d+(?!\s*\.\s*\d)(?:[.)])', re.IGNORECASE)
    LIST_ALPHA_REGEX = re.compile(r'^\s*[a-z](?:[.)])', re.IGNORECASE)
    # OCR sering mengubah bullet menjadi "o " / "O " / "0 " di awal baris.
    LIST_TEXTUAL_BULLET_REGEX = re.compile(r'^\s*[oO0](?=\s+)')
    # Tangkap simbol bullet umum, plus fallback "simbol apa pun" sebagai token awal.
    LIST_BULLET_REGEX = re.compile(
        r'^\s*(?:'
        r'[\u2022\u2023\u25e6\u2043\u2219\u00b7\u2024\u25aa\u25cf\u25cb\u25ef\u25c9\u25a0\u25a1\u25c6\u25c7\u2713\u2714\u2717\u2718\u2610\u2611\u2612\u2794\u27a4\*\-\u2013\u2014\.\+]'
        r'|[^\w\s](?=\s|$)'
        r')'
    )
    FIGURE_PANEL_MARKER_REGEX = re.compile(r'^\s*\([a-z]\)\s*$', re.IGNORECASE)

    def __init__(self):
        self.alignment_service = AlignmentService()
        self.docling_service = DoclingService()
        self.fusion_service = DoclingFusionService()
        self.visualization_service = VisualizationService(output_dir=VISUALIZATION_OUTPUT)

    @staticmethod
    def _is_env_enabled_default_true(env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _read_positive_int_env(env_name: str, default_value: int) -> int:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else default_value
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _read_float_env(env_name: str, default_value: float, min_value: float = None, max_value: float = None) -> float:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return default_value
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    @staticmethod
    def _canonical_ref_tipe(ref_tipe: str) -> str:
        if ref_tipe == 'buku':
            return 'bab'
        return ref_tipe

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

    def _annotate_picture_visual_items(self, extraction_items, docling_predictions):
        if not extraction_items or not docling_predictions:
            return extraction_items

        overlap_threshold = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_DOCLING_OVERLAP",
            self.fusion_service.OVERLAP_THRESHOLD,
            min_value=0.0,
            max_value=1.0
        )
        min_width = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_MIN_WIDTH",
            160.0,
            min_value=1.0
        )
        min_height = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_MIN_HEIGHT",
            100.0,
            min_value=1.0
        )
        shape_min_width = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_SHAPE_MIN_WIDTH",
            1.0,
            min_value=0.0
        )
        shape_min_height = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_SHAPE_MIN_HEIGHT",
            1.0,
            min_value=0.0
        )

        picture_preds = [
            pred for pred in (docling_predictions or [])
            if str(pred.get('label') or '').strip().lower() == 'picture' and pred.get('bbox')
        ]
        if not picture_preds:
            return extraction_items

        for item in extraction_items:
            item_type = str(item.get('type') or '').strip().lower()
            if item_type not in {'hline_table', 'shape'}:
                continue

            bbox = item.get('bbox')
            if not bbox or len(bbox) < 4:
                continue

            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            item_min_width = shape_min_width if item_type == 'shape' else min_width
            item_min_height = shape_min_height if item_type == 'shape' else min_height
            if width < item_min_width or height < item_min_height:
                continue

            best_overlap = 0.0
            best_pred_bbox = None
            for pred in picture_preds:
                pred_bbox = pred.get('bbox')
                overlap = self.fusion_service.calculate_overlap(bbox, pred_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_bbox = pred_bbox

            if best_overlap < overlap_threshold:
                continue

            item['docling_label'] = 'picture'
            item['docling_picture_overlap'] = round(best_overlap, 4)
            item['docling_picture_bbox'] = list(best_pred_bbox) if best_pred_bbox else None
            item['is_docling_picture_area'] = True
            item['suppress_text_alignment'] = True
            item['is_chart_visual'] = True

        return extraction_items

    def _pdf_unit_key(self, unit):
        unit_id = unit.get('unit_id')
        if unit_id is not None:
            return ('unit_id', unit_id)
        item_idx = unit.get('item_idx')
        if item_idx is not None:
            return ('item_idx', item_idx)
        bbox = unit.get('bbox')
        if bbox and len(bbox) >= 4:
            return ('bbox', tuple(bbox))
        return None

    def _unit_overlaps_fused(self, unit_bbox, fused_results):
        if not unit_bbox or len(unit_bbox) < 4:
            return False
        for result in fused_results or []:
            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            if self.fusion_service.calculate_overlap(unit_bbox, bbox) > 0:
                return True
        return False

    def _collect_unfused_pdf_units(self, all_pdf_units, fused_results, unaligned_pdf_units):
        unaligned_keys = set()
        for unit in unaligned_pdf_units or []:
            key = self._pdf_unit_key(unit)
            if key is not None:
                unaligned_keys.add(key)

        unfused = []
        seen = set(unaligned_keys)
        for unit in all_pdf_units or []:
            if not unit or not unit.get('bbox'):
                continue
            key = self._pdf_unit_key(unit)
            if key is None or key in seen:
                continue
            if self._unit_overlaps_fused(unit.get('bbox'), fused_results):
                seen.add(key)
                continue
            unfused.append(unit)
            seen.add(key)
        return unfused

    def _collect_duplicate_openxml_element_ids(self, page_vis_payload):
        duplicate_analysis = self._analyze_duplicate_openxml_elements(page_vis_payload)
        return {
            elem_id
            for elem_id, analysis in (duplicate_analysis or {}).items()
            if analysis.get('is_invalid_duplicate')
        }

    def _analyze_duplicate_openxml_elements(self, page_vis_payload):
        element_entries = {}
        page_heights = {}

        for page_num, payload in (page_vis_payload or {}).items():
            parsed_page_num = self._try_parse_int_id(page_num)
            if parsed_page_num is None:
                continue
            page_height = payload.get('page_height')
            if page_height is not None:
                try:
                    page_heights[parsed_page_num] = float(page_height)
                except (TypeError, ValueError):
                    pass
            for alignment in payload.get('alignments') or []:
                elem_id = self._try_parse_int_id((alignment or {}).get('element_id'))
                bbox = (alignment or {}).get('merged_bbox') or (alignment or {}).get('bbox')
                if elem_id is None or not bbox or len(bbox) < 4:
                    continue
                element_entries.setdefault(elem_id, []).append({
                    'page': parsed_page_num,
                    'bbox': bbox,
                })

        duplicate_analysis = {}
        for elem_id, entries in element_entries.items():
            analysis = analyze_cross_page_entries(entries, page_heights=page_heights)
            if analysis.get('is_multi_page'):
                duplicate_analysis[elem_id] = analysis
        return duplicate_analysis

    def _get_alignment_sequence_value(self, alignment):
        seq = alignment.get('element_sequence') if alignment else None
        if seq is None:
            return None
        try:
            return int(seq)
        except (TypeError, ValueError):
            return None

    def _get_alignment_center_y(self, alignment):
        bbox = (alignment or {}).get('merged_bbox') or (alignment or {}).get('bbox')
        if not bbox or len(bbox) < 4:
            return None
        return (bbox[1] + bbox[3]) / 2

    def _normalize_text_value(self, value):
        if value is None:
            return ''
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        return self.alignment_service._normalize_text(str(value))

    def _try_parse_int_id(self, value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.isdigit():
                return int(trimmed)
        return None

    def _normalize_part_position(self, position):
        if not position:
            return 'default'
        normalized = str(position).strip().lower()
        if normalized in ('first', 'even', 'default'):
            return normalized
        return 'default'

    def _resolve_section_start_page(self, db, canonical_ref_tipe, ref_id, section_id):
        if not db or ref_id is None or section_id is None:
            return None

        ref_tipes = ('bab', 'buku') if canonical_ref_tipe == 'bab' else (canonical_ref_tipe,)

        row = (
            db.query(DokumenElemenVisual.dev_page)
            .join(DokumenElemen, DokumenElemenVisual.dokumen_elemen_id == DokumenElemen.delemen_id)
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .filter(
                DokumenElemenVisual.dev_ref_tipe.in_(ref_tipes),
                DokumenElemenVisual.dev_ref_id == ref_id,
                DokumenElemenVisual.dev_page.isnot(None),
                DokumenPart.dsec_id == section_id,
                DokumenPart.dpart_type == 'body'
            )
            .order_by(DokumenElemenVisual.dev_page.asc())
            .first()
        )
        if not row or row[0] is None:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def _current_page_has_section_body_element(self, db, fused_results, section_id):
        if not db or not fused_results or section_id is None:
            return False

        candidate_ids = set()
        for result in fused_results:
            label = self._get_visual_label(result)
            if label in ('page_header', 'page_footer'):
                continue
            elem_id = self._try_parse_int_id(result.get('element_id'))
            if elem_id is not None:
                candidate_ids.add(elem_id)

        if not candidate_ids:
            return False

        row = (
            db.query(DokumenElemen.delemen_id)
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .filter(
                DokumenElemen.delemen_id.in_(candidate_ids),
                DokumenPart.dsec_id == section_id,
                DokumenPart.dpart_type == 'body'
            )
            .first()
        )
        return row is not None

    def _select_header_footer_part(self, part_infos, preference_order):
        if not part_infos:
            return None

        order_index = {pos: idx for idx, pos in enumerate(preference_order)}

        for position in preference_order:
            matches = [
                info for info in part_infos
                if info.get('position') == position and info.get('elements')
            ]
            if matches:
                return sorted(matches, key=lambda item: item.get('part_id') or 0)[0]

        candidates = [info for info in part_infos if info.get('elements')]
        if not candidates:
            return None

        def key_fn(item):
            position = item.get('position') or 'default'
            return (order_index.get(position, len(order_index)), item.get('part_id') or 0)

        return sorted(candidates, key=key_fn)[0]

    def _extract_element_text_norm(self, element):
        if not element:
            return ''
        tree = self._load_json_tree(element.delemen_json_tree)
        text = self.alignment_service._extract_text_from_json_tree(tree)
        return self._normalize_text_value(text)

    def _build_header_footer_mapping_context(
        self,
        db,
        canonical_ref_tipe,
        ref_id,
        page_num,
        fused_results,
        section_data
    ):
        section_id = self._try_parse_int_id((section_data or {}).get('dsec_id')) if isinstance(section_data, dict) else None
        if section_id is None:
            return None

        section = db.query(DokumenSection).filter(DokumenSection.dsec_id == section_id).first()
        if section is None:
            return None

        section_start_page = self._resolve_section_start_page(db, canonical_ref_tipe, ref_id, section_id)
        if section_start_page is None and self._current_page_has_section_body_element(db, fused_results, section_id):
            section_start_page = page_num

        is_first_page = section_start_page is not None and int(page_num) == int(section_start_page)
        is_even_page = int(page_num) % 2 == 0

        part_rows = db.query(DokumenPart).filter(
            DokumenPart.dsec_id == section_id,
            DokumenPart.dpart_type.in_(('header', 'footer'))
        ).all()
        if not part_rows:
            return None

        part_ids = [part.dpart_id for part in part_rows if part.dpart_id is not None]
        elements_by_part = {}
        if part_ids:
            part_elements = db.query(DokumenElemen).filter(DokumenElemen.dpart_id.in_(part_ids)).all()
            for element in part_elements:
                elements_by_part.setdefault(element.dpart_id, []).append(element)

        context = {}
        for part_type in ('header', 'footer'):
            type_parts = [part for part in part_rows if (part.dpart_type or '').lower() == part_type]
            part_infos = []
            for part in type_parts:
                position = self._normalize_part_position(part.dpart_position)
                raw_elements = elements_by_part.get(part.dpart_id, [])
                ordered_elements = sorted(
                    raw_elements,
                    key=lambda elem: (
                        0 if str(elem.delemen_type or '').lower() == 'paragraph' else 1,
                        elem.delemen_sequence if elem.delemen_sequence is not None else 2**31 - 1,
                        elem.delemen_id
                    )
                )
                part_infos.append({
                    'part_id': part.dpart_id,
                    'position': position,
                    'elements': [
                        {
                            'id': int(elem.delemen_id),
                            'text_norm': self._extract_element_text_norm(elem)
                        }
                        for elem in ordered_elements
                    ]
                })

            if is_first_page and bool(section.dsec_has_title_page):
                preference_order = ['first', 'default', 'even']
            elif bool(section.dsec_different_odd_even) and is_even_page:
                preference_order = ['even', 'default', 'first']
            else:
                preference_order = ['default', 'first', 'even']

            context[part_type] = {
                'selected_part': self._select_header_footer_part(part_infos, preference_order),
                'parts': part_infos
            }

        return context

    def _resolve_header_footer_element_id(self, result, visual_label, header_footer_context):
        if visual_label not in ('page_header', 'page_footer') or not header_footer_context:
            return None

        part_type = 'header' if visual_label == 'page_header' else 'footer'
        selected_part = (header_footer_context.get(part_type) or {}).get('selected_part')
        if not selected_part:
            return None

        candidates = selected_part.get('elements') or []
        if not candidates:
            return None

        text_norm = self._normalize_text_value(result.get('text'))
        if text_norm:
            matches = [
                candidate for candidate in candidates
                if candidate.get('text_norm') and candidate.get('text_norm') == text_norm
            ]
            if len(matches) == 1:
                return matches[0].get('id')
            if len(matches) > 1:
                return None

        return candidates[0].get('id')

    def _simplify_duplicate_unit_text(self, text):
        if not text:
            return ''
        return re.sub(r'[\W_]+', '', text, flags=re.UNICODE)

    def _get_bbox_center_y(self, bbox):
        if not bbox or len(bbox) < 4:
            return None
        return (bbox[1] + bbox[3]) / 2

    def _get_caption_structural_label(self, bbox, fused_results):
        if not bbox:
            return 'caption'
        best_label = None
        best_gap = None
        for result in fused_results or []:
            label = str(result.get('label') or result.get('docling_label') or '').lower()
            if label not in ('picture', 'table'):
                continue
            cand_bbox = result.get('bbox')
            if not cand_bbox or len(cand_bbox) < 4:
                continue
            if bbox[1] >= cand_bbox[3]:
                gap = bbox[1] - cand_bbox[3]
            elif cand_bbox[1] >= bbox[3]:
                gap = cand_bbox[1] - bbox[3]
            else:
                gap = 0
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_label = label
        if best_label == 'table':
            return 'caption_tabel'
        if best_label == 'picture':
            return 'caption_gambar'
        return 'caption'

    def _coerce_text(self, value):
        if value is None:
            return ''
        if isinstance(value, list):
            return ' '.join(str(v) for v in value)
        return str(value)

    def _text_starts_with_bab(self, text):
        if not text:
            return False
        return bool(self.BAB_TITLE_REGEX.match(text))

    def _is_subchapter_title(self, text):
        if not text:
            return False
        return bool(self.SUBCHAPTER_TITLE_REGEX.match(text))

    def _get_text_list_marker(self, text):
        if not text:
            return None
        if self._is_subchapter_title(text):
            return None
        if self.LIST_NUMERIC_REGEX.match(text):
            return 'numeric'
        if self.LIST_ALPHA_REGEX.match(text):
            return 'alpha'
        if self.LIST_TEXTUAL_BULLET_REGEX.match(text):
            return 'bullet_textual'
        if self.LIST_BULLET_REGEX.match(text):
            return 'bullet_symbol'
        return None

    def _get_visual_label(self, result):
        return str(result.get('label') or result.get('docling_label') or '').lower()

    def _is_picture_result(self, result):
        if not result:
            return False
        label = str(result.get('label') or '').lower()
        docling_label = str(result.get('docling_label') or '').lower()
        return label == 'picture' or docling_label == 'picture'

    def _is_caption_like_visual_result(self, result):
        if not result:
            return False
        label = self._get_visual_label(result)
        if label == 'caption':
            return True
        if result.get('is_chart_caption_text'):
            return True
        text = self._coerce_text(result.get('text')).strip()
        if not text:
            return False
        return self.fusion_service._is_caption_candidate(text)

    def _select_chart_visual_alignments(self, alignments):
        selected = [
            alignment for alignment in (alignments or [])
            if (
                alignment.get('is_image_part') or
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment') or
                self._alignment_has_visual_units(alignment)
            )
        ]
        return selected or list(alignments or [])

    def _is_valid_same_page_chart_caption_pair(self, picture_result, caption_result):
        if not self._is_picture_result(picture_result):
            return False
        if not self._is_caption_like_visual_result(caption_result):
            return False
        caption_text = self._coerce_text((caption_result or {}).get('text'))
        generic_picture_caption = (
            self._try_parse_int_id((picture_result or {}).get('matched_pdf_unit_count')) is not None and
            self._try_parse_int_id((picture_result or {}).get('matched_pdf_unit_count')) > 0 and
            bool((picture_result or {}).get('is_picture_area')) and
            self._is_figure_caption_text(caption_text)
        )
        if not (
            picture_result.get('is_openxml_chart') or
            picture_result.get('repair_reason') == 'chart_visual_attach' or
            picture_result.get('is_chart_visual_attachment') or
            generic_picture_caption
        ):
            return False

        picture_bbox = picture_result.get('bbox')
        caption_bbox = caption_result.get('bbox')
        if not picture_bbox or not caption_bbox:
            return False
        if len(picture_bbox) < 4 or len(caption_bbox) < 4:
            return False

        caption_gap = float(caption_bbox[1]) - float(picture_bbox[3])
        if caption_gap < -4 or caption_gap > 80:
            return False
        x_overlap = self._bbox_x_overlap_ratio(picture_bbox, caption_bbox)
        if x_overlap < 0.15:
            return False
        return True

    def _select_valid_same_page_chart_caption_results(self, results):
        if not results:
            return []
        picture_results = [result for result in results if self._is_picture_result(result)]
        caption_results = [result for result in results if self._is_caption_like_visual_result(result)]
        if not picture_results or not caption_results:
            return []

        best_pair = None
        best_gap = None
        for picture_result in picture_results:
            picture_bbox = picture_result.get('bbox')
            if not picture_bbox or len(picture_bbox) < 4:
                continue
            for caption_result in caption_results:
                if caption_result is picture_result:
                    continue
                if not self._is_valid_same_page_chart_caption_pair(picture_result, caption_result):
                    continue
                caption_bbox = caption_result.get('bbox')
                gap = max(0.0, float(caption_bbox[1]) - float(picture_bbox[3]))
                if best_pair is None or gap < best_gap:
                    best_pair = (picture_result, caption_result)
                    best_gap = gap
        if not best_pair:
            return []
        return [best_pair[0], best_pair[1]]

    def _select_valid_same_page_chart_caption_claims(self, claims):
        if not claims:
            return []
        pair_results = self._select_valid_same_page_chart_caption_results(
            [claim.get('result') for claim in claims if claim.get('result')]
        )
        if not pair_results:
            return []
        allowed_result_ids = {id(result) for result in pair_results}
        return [
            claim for claim in claims
            if id(claim.get('result')) in allowed_result_ids
        ]

    def _is_figure_panel_marker_text(self, text):
        if not text:
            return False
        return bool(self.FIGURE_PANEL_MARKER_REGEX.match(text.strip()))

    def _has_adjacent_picture_result(self, fused_results, idx):
        if not fused_results or idx < 0 or idx >= len(fused_results):
            return False

        prev_idx = idx - 1
        while prev_idx >= 0:
            prev_label = self._get_visual_label(fused_results[prev_idx])
            if prev_label not in ('page_header', 'page_footer'):
                break
            prev_idx -= 1

        if prev_idx >= 0 and self._is_picture_result(fused_results[prev_idx]):
            return True

        next_idx = idx + 1
        while next_idx < len(fused_results):
            next_label = self._get_visual_label(fused_results[next_idx])
            if next_label not in ('page_header', 'page_footer'):
                break
            next_idx += 1

        return next_idx < len(fused_results) and self._is_picture_result(fused_results[next_idx])

    def _looks_like_code_line_text(self, text):
        text = self._coerce_text(text).strip()
        if not text:
            return False
        if self.CODE_LINE_NUMBER_REGEX.match(text):
            return True
        if self.CODE_TEXT_HINT_REGEX.search(text):
            return True
        symbol_count = sum(1 for ch in text if ch in '{}[]();=<>:+-*/%#\\')
        return symbol_count >= 3

    def _count_following_code_like_lines(self, fused_results, start_idx):
        count = 0
        for i in range(start_idx + 1, len(fused_results)):
            candidate = fused_results[i]
            visual_label = self._get_visual_label(candidate)
            if visual_label in ('page_header', 'page_footer'):
                continue
            if visual_label == 'code':
                count += 1
                continue
            if visual_label == 'text' and self._looks_like_code_line_text(candidate.get('text')):
                count += 1
                continue
            break
        return count

    def _load_json_tree(self, raw_tree):
        if raw_tree is None:
            return None
        if isinstance(raw_tree, str):
            try:
                return json.loads(raw_tree)
            except Exception:
                return None
        return raw_tree

    def _get_element_json_tree(self, element, cache):
        if not element:
            return None
        elem_id = element.delemen_id
        if elem_id in cache:
            return cache[elem_id]
        tree = self._load_json_tree(element.delemen_json_tree)
        cache[elem_id] = tree
        return tree

    def _normalize_alignment_value(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ('val', 'value', 'align', 'alignment'):
                if key in value:
                    return self._normalize_alignment_value(value.get(key))
            return None
        if isinstance(value, list):
            for item in value:
                normalized = self._normalize_alignment_value(item)
                if normalized:
                    return normalized
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized.isdigit():
            code = int(normalized)
            return {
                0: 'left',
                1: 'center',
                2: 'right',
                3: 'both',
                4: 'distribute'
            }.get(code, normalized)
        if normalized in ('start', 'left'):
            return 'left'
        if normalized in ('end', 'right'):
            return 'right'
        if normalized in ('justify', 'both'):
            return 'both'
        if normalized == 'distribute':
            return 'distribute'
        if normalized in ('centercontinuous', 'center_continuous', 'center-continuous'):
            return 'center'
        return normalized
