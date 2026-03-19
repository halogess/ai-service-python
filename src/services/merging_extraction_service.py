
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
from database import SessionLocal

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")
VISUALIZATION_OUTPUT = os.getenv("VISUALIZATION_OUTPUT", "visualization_output")

class MergingExtractionService:
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
    def _is_env_enabled_default_true(env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

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

                    if save_to_db or generate_visualizations:
                        payload = {
                            'alignments': alignments,
                            'fused_results': fused_results,
                            'header_footer_units': header_footer_units,
                            'section_data': section_data
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

            if page_vis_payload and (save_to_db or generate_visualizations):
                try:
                    duplicate_element_ids = self._collect_duplicate_openxml_element_ids(page_vis_payload)
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
                    logger.info(
                        "Resolved document-level claims for %s:%s cleared=%s affected_pages=%s",
                        canonical_ref_tipe,
                        ref_id,
                        claim_resolution_stats.get('cleared_claims', 0),
                        claim_resolution_stats.get('affected_pages', 0)
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
                                duplicate_element_ids
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

                            json_output_dir = output_dir
                            if not json_output_dir and vis_paths:
                                json_output_dir = os.path.dirname(list(vis_paths.values())[0])
                            if json_output_dir:
                                json_path = os.path.join(json_output_dir, f"page_{page_num}_fusion_data.json")
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
        element_pages = {}
        for page_num, payload in (page_vis_payload or {}).items():
            for alignment in payload.get('alignments') or []:
                elem_id = alignment.get('element_id')
                if elem_id is None:
                    continue
                element_pages.setdefault(elem_id, set()).add(page_num)
        return {elem_id for elem_id, pages in element_pages.items() if len(pages) > 1}

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

    def _extract_text_run_ids(self, json_tree):
        ids = []
        def walk(node):
            if isinstance(node, dict):
                if 'dftx_id' in node:
                    ids.append(node.get('dftx_id'))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
        walk(json_tree)
        return [i for i in ids if i is not None]

    def _extract_paragraph_alignment(self, json_tree):
        if not json_tree:
            return None
        key_candidates = {
            'alignment',
            'align',
            'textalign',
            'text_align',
            'paragraphalignment',
            'paragraph_align',
            'justification',
            'jc',
            'dfp_jc'
        }
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if str(key).lower() in key_candidates:
                        normalized = self._normalize_alignment_value(value)
                        if normalized:
                            return normalized
                    found = walk(value)
                    if found:
                        return found
            elif isinstance(node, list):
                for item in node:
                    found = walk(item)
                    if found:
                        return found
            return None

        return walk(json_tree)

    def _get_paragraph_alignment_from_dfp(self, db, dfp_id, dfp_cache):
        if not db or not dfp_id:
            return None
        if dfp_id in dfp_cache:
            return dfp_cache[dfp_id]
        alignment = None
        try:
            paragraph_format = db.query(DokumenFormatParagraf).filter(
                DokumenFormatParagraf.dfp_id == dfp_id
            ).first()
            if paragraph_format and paragraph_format.dfp_jc is not None:
                alignment = self._normalize_alignment_value(paragraph_format.dfp_jc)
        except Exception:
            alignment = None
        dfp_cache[dfp_id] = alignment
        return alignment

    def _get_element_alignment(self, element, json_tree, db=None, dfp_cache=None):
        alignment = self._extract_paragraph_alignment(json_tree)
        if not alignment and db and dfp_cache is not None and isinstance(json_tree, dict):
            dfp_id = json_tree.get('dfp_id')
            if dfp_id:
                alignment = self._get_paragraph_alignment_from_dfp(db, dfp_id, dfp_cache)
        return alignment

    def _get_element_bold_state(self, element, json_tree, db, bold_cache):
        if not db or not element or not json_tree:
            return None
        dftx_ids = self._extract_text_run_ids(json_tree)
        if not dftx_ids:
            return None
        missing = [dftx_id for dftx_id in dftx_ids if dftx_id not in bold_cache]
        if missing:
            try:
                rows = db.query(
                    DokumenFormatText.dftx_id,
                    DokumenFormatText.dftx_bold
                ).filter(
                    DokumenFormatText.dftx_id.in_(tuple(missing))
                ).all()
                for dftx_id, dftx_bold in rows:
                    bold_cache[dftx_id] = bool(dftx_bold)
            except Exception:
                for dftx_id in missing:
                    bold_cache[dftx_id] = None
        states = [bold_cache.get(dftx_id) for dftx_id in dftx_ids if dftx_id in bold_cache]
        states = [s for s in states if s is not None]
        if not states:
            return None
        return any(states)

    def _is_paragraph_center_aligned(self, element, json_cache, align_cache, db=None, dfp_cache=None):
        if not element:
            return False
        elem_id = element.delemen_id
        if elem_id in align_cache:
            return align_cache[elem_id]
        tree = self._get_element_json_tree(element, json_cache)
        alignment = self._get_element_alignment(element, tree, db=db, dfp_cache=dfp_cache)
        is_center = alignment in ('center', 'centre')
        align_cache[elem_id] = is_center
        return is_center

    def _new_structural_label_state(self):
        return {
            'in_bab_block': False,
            'list_marker_levels': {},
            'current_list_level': None,
            'list_context_active': False,
            'non_list_streak': 0
        }

    def _apply_structural_labels(self, db, fused_results, structural_state=None, skip_if_labeled=False):
        if not fused_results:
            return
        if skip_if_labeled:
            all_labeled = all(
                result.get('dev_label_struktural') not in (None, '')
                for result in fused_results
            )
            if all_labeled:
                return
        element_ids = {
            result.get('element_id')
            for result in fused_results
            if result.get('element_id') is not None
        }
        element_map = {}
        if db and element_ids:
            elements = db.query(DokumenElemen).filter(
                DokumenElemen.delemen_id.in_(element_ids)
            ).all()
            element_map = {elem.delemen_id: elem for elem in elements}

        json_cache = {}
        align_cache = {}
        dfp_align_cache = {}
        bold_cache = {}
        if structural_state is None:
            structural_state = self._new_structural_label_state()
        in_bab_block = bool(structural_state.get('in_bab_block', False))
        list_marker_levels = dict(structural_state.get('list_marker_levels') or {})
        current_list_level = structural_state.get('current_list_level')
        list_context_active = bool(structural_state.get('list_context_active', False))
        non_list_streak = int(structural_state.get('non_list_streak', 0) or 0)

        for idx, result in enumerate(fused_results):
            visual_label = self._get_visual_label(result)
            docling_label = str(result.get('docling_label') or '').lower()
            if visual_label in ('page_header', 'page_footer'):
                result['dev_label_struktural'] = visual_label
                continue
            text = self._coerce_text(result.get('text')).strip()
            elem_id = result.get('element_id')
            element = element_map.get(elem_id)

            elem_type = result.get('element_type')
            if not elem_type and element is not None:
                elem_type = element.delemen_type
            elem_type_norm = str(elem_type).lower() if elem_type else None

            is_section_header = visual_label == 'section_header'
            is_subchapter_text = self._is_subchapter_title(text)
            center_aligned = False
            if is_section_header and element is not None:
                center_aligned = self._is_paragraph_center_aligned(
                    element,
                    json_cache,
                    align_cache,
                    db=db,
                    dfp_cache=dfp_align_cache
                )

            structural_label = None
            if is_section_header and center_aligned:
                if in_bab_block or self._text_starts_with_bab(text):
                    structural_label = 'judul_bab'
                    in_bab_block = True
                else:
                    in_bab_block = False
            else:
                in_bab_block = False

            if not structural_label and is_subchapter_text and visual_label in ('section_header', 'list_item'):
                structural_label = 'judul_subbab'

            is_bab_heading_text = self._text_starts_with_bab(text)
            if (
                not structural_label
                and is_section_header
                and not is_subchapter_text
                and not is_bab_heading_text
            ):
                code_like_lines = self._count_following_code_like_lines(fused_results, idx)
                if code_like_lines >= 2:
                    structural_label = 'judul_kode'
                elif code_like_lines >= 1 and self.CODE_TITLE_HEADER_REGEX.search(text):
                    structural_label = 'judul_kode'

            if not structural_label:
                if (
                    visual_label not in ('picture', 'table', 'formula', 'code')
                    and self._is_figure_panel_marker_text(text)
                    and self._has_adjacent_picture_result(
                        fused_results,
                        idx
                    )
                ):
                    structural_label = 'caption_gambar'

            if not structural_label and (visual_label == 'footnote' or docling_label == 'footnote'):
                structural_label = 'footnote'

            if not structural_label:
                if visual_label == 'caption':
                    structural_label = self._get_caption_structural_label(
                        result.get('bbox'),
                        fused_results
                    )
                else:
                    structural_label = {
                        'picture': 'gambar',
                        'table': 'tabel',
                        'formula': 'rumus',
                        'code': 'kode',
                        'page_header': 'page_header',
                        'page_footer': 'page_footer'
                    }.get(visual_label)

            if not structural_label:
                is_list_candidate = False
                is_list_item_type = bool(elem_type_norm and elem_type_norm.startswith('list-item-'))
                if is_list_item_type:
                    is_list_candidate = True
                elif visual_label in ('section_header', 'list_item'):
                    is_list_candidate = True

                if is_list_candidate:
                    marker = self._get_text_list_marker(text)
                    if marker:
                        if not list_context_active or non_list_streak > 1:
                            list_marker_levels = {}
                            current_list_level = None
                        list_context_active = True
                        non_list_streak = 0
                        if marker in list_marker_levels:
                            list_level = list_marker_levels[marker]
                        else:
                            list_level = (current_list_level or 0) + 1
                            list_marker_levels[marker] = list_level
                        current_list_level = list_level
                        structural_label = f'list_level_{list_level}'
                    else:
                        non_list_streak += 1
                        if non_list_streak > 1:
                            list_context_active = False
                            list_marker_levels = {}
                            current_list_level = None
                        if visual_label == 'list_item' or is_list_item_type:
                            structural_label = 'paragraf'
                else:
                    if list_context_active:
                        non_list_streak += 1
                        if non_list_streak > 1:
                            list_context_active = False

            if not structural_label:
                if elem_type_norm == 'paragraph' and visual_label == 'text':
                    structural_label = 'paragraf'

            if not structural_label and visual_label == 'section_header':
                structural_label = 'section_header'

            if structural_label in ('judul_bab', 'judul_subbab', 'judul_kode') or is_subchapter_text:
                list_marker_levels = {}
                current_list_level = None
                list_context_active = False
                non_list_streak = 0

            result['dev_label_struktural'] = structural_label

        structural_state['in_bab_block'] = in_bab_block
        structural_state['list_marker_levels'] = dict(list_marker_levels)
        structural_state['current_list_level'] = current_list_level
        structural_state['list_context_active'] = list_context_active
        structural_state['non_list_streak'] = non_list_streak

        # Expand caption labels to subsequent lines when formatting matches
        if db:
            for idx, result in enumerate(fused_results):
                visual_label = str(
                    result.get('label') or result.get('docling_label') or ''
                ).lower()
                if visual_label != 'caption':
                    continue
                caption_label = result.get('dev_label_struktural') or 'caption'
                elem_id = result.get('element_id')
                element = element_map.get(elem_id)
                tree = self._get_element_json_tree(element, json_cache)
                prev_align = self._get_element_alignment(
                    element,
                    tree,
                    db=db,
                    dfp_cache=dfp_align_cache
                )
                prev_bold = self._get_element_bold_state(
                    element,
                    tree,
                    db,
                    bold_cache
                )
                if prev_align is None or prev_bold is None:
                    continue
                j = idx + 1
                while j < len(fused_results):
                    next_result = fused_results[j]
                    next_visual = str(
                        next_result.get('label') or next_result.get('docling_label') or ''
                    ).lower()
                    if next_visual in ('page_header', 'page_footer'):
                        j += 1
                        continue
                    if next_visual not in ('section_header', 'text'):
                        break
                    next_elem_id = next_result.get('element_id')
                    next_element = element_map.get(next_elem_id)
                    next_tree = self._get_element_json_tree(next_element, json_cache)
                    next_align = self._get_element_alignment(
                        next_element,
                        next_tree,
                        db=db,
                        dfp_cache=dfp_align_cache
                    )
                    next_bold = self._get_element_bold_state(
                        next_element,
                        next_tree,
                        db,
                        bold_cache
                    )
                    if next_align is None or next_bold is None:
                        break
                    if next_align != prev_align or next_bold != prev_bold:
                        break
                    next_result['dev_label_struktural'] = caption_label
                    prev_align = next_align
                    prev_bold = next_bold
                    j += 1

    def _merge_duplicate_units_with_neighbors(self, alignments, duplicate_element_ids):
        if not alignments or not duplicate_element_ids:
            return alignments, set()

        def is_visual_alignment(alignment):
            if not alignment:
                return False
            if (
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment') or
                alignment.get('is_image_part')
            ):
                return True
            units = alignment.get('matched_pdf_units', []) or []
            return any(
                unit.get('is_chart_visual') or unit.get('item_type') in ('image', 'shape', 'hline_table')
                for unit in units
            )

        ordered = [
            alignment for alignment in alignments
            if not alignment.get('is_table') and alignment.get('merged_bbox')
        ]
        ordered.sort(key=lambda a: (self._get_alignment_center_y(a) or 0, a.get('merged_bbox')[0]))

        touched = set()
        removed_element_ids = set()
        for idx, alignment in enumerate(ordered):
            if alignment.get('element_id') not in duplicate_element_ids:
                continue
            if not self._is_duplicate_sequence_far(
                alignments,
                alignment,
                self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ):
                continue

            units = list(alignment.get('matched_pdf_units', []))
            if not units:
                continue

            above = ordered[idx - 1] if idx > 0 else None
            below = ordered[idx + 1] if idx + 1 < len(ordered) else None

            remaining_units = []
            for unit in units:
                if unit.get('item_type') != 'group':
                    remaining_units.append(unit)
                    continue
                unit_bbox = unit.get('bbox')
                if not unit_bbox:
                    remaining_units.append(unit)
                    continue
                unit_text = self._normalize_text_value(unit.get('text'))
                if not unit_text:
                    remaining_units.append(unit)
                    continue

                target = None
                above_text = self._normalize_text_value(above.get('element_text')) if above else ''
                below_text = self._normalize_text_value(below.get('element_text')) if below else ''
                above_contains = bool(above_text) and unit_text in above_text
                below_contains = bool(below_text) and unit_text in below_text
                if unit_text and len(unit_text) <= self.SHORT_DUPLICATE_UNIT_LEN:
                    simplified_unit = self._simplify_duplicate_unit_text(unit_text)
                    if simplified_unit:
                        if not above_contains and above_text:
                            simplified_above = self._simplify_duplicate_unit_text(above_text)
                            if simplified_above and simplified_unit in simplified_above:
                                above_contains = True
                        if not below_contains and below_text:
                            simplified_below = self._simplify_duplicate_unit_text(below_text)
                            if simplified_below and simplified_unit in simplified_below:
                                below_contains = True

                if above_contains and not below_contains:
                    target = above
                elif below_contains and not above_contains:
                    target = below
                elif above_contains and below_contains:
                    unit_y = self._get_bbox_center_y(unit_bbox)
                    above_y = self._get_alignment_center_y(above)
                    below_y = self._get_alignment_center_y(below)
                    above_delta = abs(unit_y - above_y) if unit_y is not None and above_y is not None else None
                    below_delta = abs(unit_y - below_y) if unit_y is not None and below_y is not None else None
                    if above_delta is None and below_delta is None:
                        target = below
                    elif above_delta is None:
                        target = below
                    elif below_delta is None:
                        target = above
                    else:
                        target = above if above_delta <= below_delta else below

                if is_visual_alignment(target):
                    target = None

                if not target:
                    remaining_units.append(unit)
                    continue

                unit_key = self._pdf_unit_key(unit)
                target_units = target.setdefault('matched_pdf_units', [])
                target_keys = {
                    self._pdf_unit_key(u)
                    for u in target_units
                    if self._pdf_unit_key(u) is not None
                }
                if unit_key is None or unit_key in target_keys:
                    remaining_units.append(unit)
                    continue

                unit['merged_from_duplicate'] = True
                target_units.append(unit)
                target_units.sort(key=lambda u: u.get('item_idx', -1))
                touched.add(id(target))

            alignment['matched_pdf_units'] = remaining_units
            touched.add(id(alignment))
            if not remaining_units:
                removed_element_ids.add(alignment.get('element_id'))

        if touched:
            for alignment in alignments:
                if id(alignment) in touched:
                    self.alignment_service._recompute_alignment_bboxes(alignment)

        if not removed_element_ids:
            return alignments, set()
        return (
            [alignment for alignment in alignments if alignment.get('element_id') not in removed_element_ids],
            removed_element_ids
        )

    def _sync_fused_bboxes_with_alignments(self, fused_results, alignments, removed_element_ids=None):
        if not fused_results or not alignments:
            return
        if removed_element_ids:
            fused_results[:] = [
                result for result in fused_results
                if not (
                    result.get('source') == 'alignment'
                    and result.get('element_id') in removed_element_ids
                )
            ]

        alignment_by_id = {}
        for alignment in alignments:
            elem_id = alignment.get('element_id')
            if elem_id is None:
                continue
            alignment_by_id.setdefault(elem_id, []).append(alignment)

        updated_results = []
        seen_picture_bboxes = set()

        for result in fused_results:
            if (
                result.get('source') != 'alignment' and
                not (
                    result.get('source') == 'merged' and
                    result.get('element_id') is not None
                )
            ):
                updated_results.append(result)
                continue
            elem_id = result.get('element_id')
            if elem_id is None:
                updated_results.append(result)
                continue

            is_picture = (
                result.get('label') == 'picture'
                or result.get('docling_label') == 'picture'
                or result.get('has_pdf_image')
                or result.get('is_image_part')
            )
            alignments_for_elem = alignment_by_id.get(elem_id, [])
            has_chart_alignment = any(
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment')
                for alignment in alignments_for_elem
            )

            if is_picture and alignments_for_elem and not has_chart_alignment:
                image_units = [
                    unit
                    for alignment in alignments_for_elem
                    for unit in (alignment.get('matched_pdf_units', []) or [])
                    if unit.get('item_type') in ('image', 'shape') or unit.get('text') == '[IMG]'
                ]
                if image_units:
                    for unit in image_units:
                        bbox = unit.get('bbox')
                        if not bbox or len(bbox) < 4:
                            continue
                        key = (elem_id, tuple(bbox))
                        if key in seen_picture_bboxes:
                            continue
                        seen_picture_bboxes.add(key)
                        new_result = dict(result)
                        new_result['bbox'] = list(bbox)
                        updated_results.append(new_result)
                    continue

            candidate_alignments = alignments_for_elem
            if not is_picture and alignments_for_elem:
                if result.get('is_text_part'):
                    candidate_alignments = [
                        alignment for alignment in alignments_for_elem
                        if alignment.get('is_text_part')
                    ]
                elif result.get('is_image_part') is not True:
                    candidate_alignments = [
                        alignment for alignment in alignments_for_elem
                        if not alignment.get('is_image_part')
                    ]
                if not candidate_alignments:
                    candidate_alignments = alignments_for_elem

            align_bboxes = [
                alignment.get('merged_bbox')
                for alignment in candidate_alignments
                if alignment.get('merged_bbox')
            ]
            if not align_bboxes:
                updated_results.append(result)
                continue
            align_bbox = self.alignment_service._merge_bboxes(align_bboxes)
            if not align_bbox:
                updated_results.append(result)
                continue
            if is_picture and has_chart_alignment:
                result['bbox'] = list(align_bbox)
                updated_results.append(result)
                continue
            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                result['bbox'] = list(align_bbox)
                updated_results.append(result)
                continue
            result['bbox'] = [
                min(bbox[0], align_bbox[0]),
                min(bbox[1], align_bbox[1]),
                max(bbox[2], align_bbox[2]),
                max(bbox[3], align_bbox[3])
            ]
            updated_results.append(result)

        fused_results[:] = updated_results

    def _alignment_has_visual_units(self, alignment):
        if not alignment:
            return False
        return any(
            unit.get('is_chart_visual') or unit.get('item_type') in ('image', 'shape', 'hline_table')
            for unit in (alignment.get('matched_pdf_units') or [])
        )

    def _sort_fused_results_in_reading_order(self, fused_results):
        if not fused_results:
            return
        from functools import cmp_to_key

        def sort_key(item):
            return item.get('bbox') or [0, 0, 0, 0]

        def compare(a, b):
            a_bbox = sort_key(a)
            b_bbox = sort_key(b)
            y_diff = a_bbox[1] - b_bbox[1]
            if abs(y_diff) > 10:
                return -1 if y_diff < 0 else 1
            x_diff = a_bbox[0] - b_bbox[0]
            return -1 if x_diff < 0 else (1 if x_diff > 0 else 0)

        fused_results.sort(key=cmp_to_key(compare))

    def _find_best_visual_alignment_for_bbox(self, alignments, target_bbox):
        if not alignments or not target_bbox:
            return None

        candidates = []
        for alignment in alignments:
            bbox = alignment.get('merged_bbox')
            if not bbox or not self._alignment_has_visual_units(alignment):
                continue
            if not (
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment')
            ):
                continue
            overlap = self.fusion_service.calculate_overlap(bbox, target_bbox)
            if overlap <= 0:
                continue
            candidates.append((alignment, overlap))

        if not candidates:
            return None

        def candidate_score(entry):
            alignment, overlap = entry
            bbox = alignment.get('merged_bbox')
            return (
                1 if alignment.get('is_openxml_visual_slot') else 0,
                overlap,
                self._bbox_area(bbox),
                alignment.get('element_sequence') or 0,
            )

        return max(candidates, key=candidate_score)[0]

    def _build_picture_result_from_alignment(self, alignment, docling_bbox=None, repair_reason=None):
        if not alignment or not alignment.get('merged_bbox'):
            return None
        matched_units = alignment.get('matched_pdf_units') or []
        has_pdf_image = any(unit.get('item_type') == 'image' for unit in matched_units)
        has_shape_units = any(
            unit.get('item_type') in ('shape', 'hline_table') or unit.get('is_chart_visual')
            for unit in matched_units
        )
        has_table_units = any(unit.get('item_type') in ('table', 'hline_table') for unit in matched_units)
        openxml_indices = alignment.get('openxml_indices') or []
        overlap = 0.0
        if docling_bbox:
            overlap = self.fusion_service.calculate_overlap(alignment.get('merged_bbox'), docling_bbox)
        return {
            'bbox': list(alignment.get('merged_bbox')),
            'label': 'picture',
            'text': alignment.get('element_text', ''),
            'overlap': overlap,
            'source': 'alignment',
            'element_id': alignment.get('element_id'),
            'element_type': alignment.get('element_type'),
            'element_sequence': alignment.get('element_sequence'),
            'openxml_idx': min(openxml_indices) if openxml_indices else alignment.get('openxml_idx'),
            'docling_label': 'picture' if docling_bbox else None,
            'is_text_part': alignment.get('is_text_part'),
            'is_image_part': alignment.get('is_image_part'),
            'unit_id': alignment.get('unit_id'),
            'merged_count': 1,
            'is_picture_area': True,
            'has_shape_units': has_shape_units,
            'has_pdf_image': has_pdf_image,
            'has_table_units': has_table_units,
            'is_text_only_item': False,
            'is_openxml_chart': alignment.get('is_openxml_chart', False),
            'is_openxml_visual_slot': alignment.get('is_openxml_visual_slot', False),
            'visual_slot_promoted': alignment.get('visual_slot_promoted', False),
            'repair_reason': repair_reason or alignment.get('repair_reason'),
        }

    def _picture_body_text_overlap_ratio(self, picture_result, fused_results):
        if not picture_result or not fused_results:
            return 0.0
        picture_bbox = picture_result.get('bbox')
        if not picture_bbox:
            return 0.0

        max_overlap = 0.0
        picture_elem_id = picture_result.get('element_id')
        for other in fused_results:
            if other is picture_result:
                continue
            if other.get('element_id') == picture_elem_id and picture_elem_id is not None:
                continue
            other_bbox = other.get('bbox')
            if not other_bbox:
                continue
            label = self._get_visual_label(other)
            if label in ('picture', 'caption', 'table', 'page_header', 'page_footer', 'formula', 'code', 'footnote'):
                continue
            if self.fusion_service._is_caption_candidate(self._coerce_text(other.get('text'))):
                continue
            overlap = self.fusion_service.calculate_overlap(picture_bbox, other_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    def _repair_picture_fusion_results(self, alignments, fused_results, docling_predictions=None):
        if not fused_results:
            return fused_results, {
                'missing_picture_repair_count': 0,
                'picture_overlap_prune_count': 0,
            }

        debug = {
            'missing_picture_repair_count': 0,
            'picture_overlap_prune_count': 0,
        }
        raw_picture_preds = [
            pred for pred in (docling_predictions or [])
            if pred.get('label') == 'picture' and pred.get('bbox')
        ]
        picture_results = [result for result in fused_results if self._is_picture_result(result)]

        if raw_picture_preds and not picture_results:
            for pred in raw_picture_preds:
                alignment = self._find_best_visual_alignment_for_bbox(alignments, pred.get('bbox'))
                if not alignment:
                    continue
                replacement = self._build_picture_result_from_alignment(
                    alignment,
                    docling_bbox=pred.get('bbox'),
                    repair_reason='missing_picture_repair'
                )
                if not replacement:
                    continue
                existing_result = None
                for result in fused_results:
                    if result.get('element_id') == alignment.get('element_id'):
                        existing_result = result
                        break
                if existing_result is not None:
                    existing_result.update(replacement)
                else:
                    fused_results.append(replacement)
                debug['missing_picture_repair_count'] += 1
            picture_results = [result for result in fused_results if self._is_picture_result(result)]

        alignment_by_element = {}
        for alignment in alignments or []:
            elem_id = alignment.get('element_id')
            if elem_id is None or not alignment.get('merged_bbox'):
                continue
            alignment_by_element.setdefault(elem_id, []).append(alignment)

        picture_overlap_threshold = self._read_float_env(
            'ALIGNMENT_PICTURE_TEXT_OVERLAP_REPAIR_THRESHOLD',
            0.2,
            min_value=0.0,
            max_value=1.0
        )

        for result in picture_results:
            overlap_ratio = self._picture_body_text_overlap_ratio(result, fused_results)
            if overlap_ratio <= picture_overlap_threshold:
                continue
            elem_id = result.get('element_id')
            candidate_alignments = alignment_by_element.get(elem_id) or []
            if not candidate_alignments:
                continue
            best_alignment = max(
                candidate_alignments,
                key=lambda alignment: (
                    1 if alignment.get('is_openxml_visual_slot') else 0,
                    self.fusion_service.calculate_overlap(
                        alignment.get('merged_bbox'),
                        result.get('bbox')
                    ) if alignment.get('merged_bbox') and result.get('bbox') else 0.0,
                    self._bbox_area(alignment.get('merged_bbox')),
                )
            )
            align_bbox = best_alignment.get('merged_bbox')
            if not align_bbox:
                continue
            result['bbox'] = list(align_bbox)
            result['repair_reason'] = 'picture_overlap_prune'
            result['picture_text_overlap_ratio'] = overlap_ratio
            debug['picture_overlap_prune_count'] += 1

        self._sort_fused_results_in_reading_order(fused_results)
        return fused_results, debug

    @staticmethod
    def _bbox_area(bbox):
        if not bbox or len(bbox) < 4:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return width * height

    def _visual_result_claim_score(self, result):
        overlap = float((result or {}).get('overlap') or 0.0)
        area = self._bbox_area((result or {}).get('bbox'))
        text_len = len(self._coerce_text((result or {}).get('text')))
        return overlap, area, text_len

    def _visual_existing_claim_score(self, row):
        bbox = [
            getattr(row, 'dev_bbox_x0', None),
            getattr(row, 'dev_bbox_y0', None),
            getattr(row, 'dev_bbox_x1', None),
            getattr(row, 'dev_bbox_y1', None),
        ]
        area = self._bbox_area(bbox)
        text_len = len(self._coerce_text(getattr(row, 'dev_text', None)))
        # Historical rows do not store overlap, so default to 0.0.
        return 0.0, area, text_len

    def _is_table_like_visual_result(self, result):
        if not result:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label == 'table':
            return True
        if result.get('has_table_units'):
            return True
        element_type = str(result.get('element_type') or '').strip().lower()
        return 'table' in element_type

    def _clear_visual_result_claim(self, result, reason, winner_claim=None):
        if not result or result.get('element_id') is None:
            return False
        result['element_id'] = None
        result['duplicate_claim_conflict'] = True
        result['duplicate_claim_reason'] = reason
        if winner_claim:
            result['duplicate_claim_winner_page'] = winner_claim.get('page')
            winner_result = winner_claim.get('result') or {}
            result['duplicate_claim_winner_element_id'] = winner_result.get('element_id')
        return True

    def _resolve_document_visual_claims(self, page_vis_payload):
        if not page_vis_payload:
            return {
                'cleared_claims': 0,
                'affected_pages': 0,
                'same_page_cleared': 0,
                'far_gap_cleared': 0
            }

        claims_by_element = {}
        for page_num, payload in (page_vis_payload or {}).items():
            parsed_page_num = self._try_parse_int_id(page_num)
            if parsed_page_num is None:
                continue
            for result in payload.get('fused_results') or []:
                elem_id = self._try_parse_int_id((result or {}).get('element_id'))
                if elem_id is None:
                    continue
                visual_label = self._get_visual_label(result)
                if visual_label in ('page_header', 'page_footer'):
                    continue
                if self._is_table_like_visual_result(result):
                    continue
                claims_by_element.setdefault(elem_id, []).append({
                    'page': parsed_page_num,
                    'result': result,
                    'score': self._visual_result_claim_score(result)
                })

        cleared_claims = 0
        same_page_cleared = 0
        far_gap_cleared = 0
        affected_pages = set()

        for elem_id, claims in claims_by_element.items():
            claims_by_page = {}
            for claim in claims:
                claims_by_page.setdefault(claim['page'], []).append(claim)

            for page, page_claims in sorted(claims_by_page.items()):
                winner_claim = max(page_claims, key=lambda claim: claim['score'])
                for claim in page_claims:
                    if claim is winner_claim:
                        continue
                    if self._clear_visual_result_claim(claim.get('result'), 'same_page_duplicate', winner_claim):
                        cleared_claims += 1
                        same_page_cleared += 1
                        affected_pages.add(page)

        return {
            'cleared_claims': cleared_claims,
            'affected_pages': len(affected_pages),
            'same_page_cleared': same_page_cleared,
            'far_gap_cleared': far_gap_cleared
        }

    def _collect_existing_claims_by_element(self, db, canonical_ref_tipe, ref_id, page_num, element_ids):
        if not db or ref_id is None or page_num is None or not element_ids:
            return {}

        query = db.query(DokumenElemenVisual).filter(
            DokumenElemenVisual.dev_ref_id == ref_id,
            DokumenElemenVisual.dokumen_elemen_id.in_(list(element_ids)),
            DokumenElemenVisual.dev_page.isnot(None),
            DokumenElemenVisual.dev_page != page_num
        )
        if canonical_ref_tipe == 'bab':
            query = query.filter(DokumenElemenVisual.dev_ref_tipe.in_(('bab', 'buku')))
        else:
            query = query.filter(DokumenElemenVisual.dev_ref_tipe == canonical_ref_tipe)

        existing_rows = list(query.all() or [])
        claims_by_element = {}
        for row in existing_rows:
            elem_id = self._try_parse_int_id(getattr(row, 'dokumen_elemen_id', None))
            page = self._try_parse_int_id(getattr(row, 'dev_page', None))
            if elem_id is None or page is None:
                continue
            claims_by_element.setdefault(elem_id, []).append({
                'dev_id': self._try_parse_int_id(getattr(row, 'dev_id', None)),
                'page': page,
                'score': self._visual_existing_claim_score(row),
                'label': getattr(row, 'dev_label', None),
                'is_table_like': str(getattr(row, 'dev_label', '') or '').strip().lower() == 'table'
            })
        return claims_by_element

    def _prune_far_gap_duplicate_claims(self, fused_results, page_num, existing_claims_by_element):
        if not fused_results:
            return fused_results, 0, set()

        current_page = self._try_parse_int_id(page_num)
        if current_page is None:
            return fused_results, 0, set()

        claim_result_indices = {}
        for idx, result in enumerate(fused_results):
            elem_id = self._try_parse_int_id((result or {}).get('element_id'))
            if elem_id is None:
                continue
            visual_label = self._get_visual_label(result)
            if visual_label in ('page_header', 'page_footer'):
                continue
            claim_result_indices.setdefault(elem_id, []).append(idx)

        if not claim_result_indices:
            return fused_results, 0, set()

        cleared_current_claims = 0
        clear_existing_ids = set()

        for elem_id, indices in claim_result_indices.items():
            current_results = [fused_results[idx] for idx in indices]
            if any(self._is_table_like_visual_result(result) for result in current_results):
                continue

            existing_claims = existing_claims_by_element.get(elem_id) or []
            far_claims = [
                claim for claim in existing_claims
                if abs((claim.get('page') or current_page) - current_page) > self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ]
            if not far_claims:
                continue
            if any(bool(claim.get('is_table_like')) for claim in far_claims):
                continue

            best_current_idx = max(indices, key=lambda i: self._visual_result_claim_score(fused_results[i]))
            best_current_score = self._visual_result_claim_score(fused_results[best_current_idx])
            best_existing_score = max((claim.get('score') or (0.0, 0.0, 0)) for claim in far_claims)

            if best_current_score > best_existing_score:
                for idx in indices:
                    if idx == best_current_idx or fused_results[idx].get('element_id') is None:
                        continue
                    fused_results[idx]['element_id'] = None
                    fused_results[idx]['duplicate_claim_conflict'] = True
                    cleared_current_claims += 1
                for claim in far_claims:
                    dev_id = claim.get('dev_id')
                    if dev_id is not None:
                        clear_existing_ids.add(dev_id)
            else:
                for idx in indices:
                    if fused_results[idx].get('element_id') is None:
                        continue
                    fused_results[idx]['element_id'] = None
                    fused_results[idx]['duplicate_claim_conflict'] = True
                    cleared_current_claims += 1

        return fused_results, cleared_current_claims, clear_existing_ids

    def _replace_visual_records(self, db, ref_tipe, ref_id, page_num, fused_results, structural_state=None, section_data=None, apply_duplicate_claim_guard=True):
        if not db or ref_id is None or page_num is None:
            return list(fused_results or [])
        canonical_ref_tipe = self._canonical_ref_tipe(ref_tipe)
        fused_results = list(fused_results or [])

        if apply_duplicate_claim_guard:
            claimed_element_ids = set()
            for result in fused_results:
                elem_id = self._try_parse_int_id(result.get('element_id'))
                if elem_id is None:
                    continue
                visual_label = self._get_visual_label(result)
                if visual_label in ('page_header', 'page_footer'):
                    continue
                claimed_element_ids.add(elem_id)

            existing_claims = self._collect_existing_claims_by_element(
                db,
                canonical_ref_tipe,
                ref_id,
                page_num,
                claimed_element_ids
            )
            fused_results, cleared_claim_rows, clear_existing_ids = self._prune_far_gap_duplicate_claims(
                fused_results,
                page_num,
                existing_claims
            )
            if clear_existing_ids:
                db.query(DokumenElemenVisual).filter(
                    DokumenElemenVisual.dev_id.in_(list(clear_existing_ids))
                ).update(
                    {DokumenElemenVisual.dokumen_elemen_id: None},
                    synchronize_session=False
                )

            logger.debug(
                "Page %s: far-gap duplicate claim guard cleared_current=%s cleared_existing=%s",
                page_num,
                cleared_claim_rows,
                len(clear_existing_ids)
            )

        if fused_results:
            self._apply_structural_labels(
                db,
                fused_results,
                structural_state=structural_state,
                skip_if_labeled=True
            )
        delete_query = db.query(DokumenElemenVisual).filter(
            DokumenElemenVisual.dev_ref_id == ref_id,
            DokumenElemenVisual.dev_page == page_num
        )
        if canonical_ref_tipe == 'bab':
            delete_query = delete_query.filter(DokumenElemenVisual.dev_ref_tipe.in_(('bab', 'buku')))
        else:
            delete_query = delete_query.filter(DokumenElemenVisual.dev_ref_tipe == canonical_ref_tipe)
        delete_query.delete(synchronize_session=False)

        has_header_footer_rows = any(
            self._get_visual_label(result) in ('page_header', 'page_footer')
            for result in (fused_results or [])
        )
        header_footer_context = None
        if has_header_footer_rows:
            header_footer_context = self._build_header_footer_mapping_context(
                db,
                canonical_ref_tipe,
                ref_id,
                page_num,
                fused_results,
                section_data
            )

        header_footer_total = 0
        header_footer_mapped = 0
        header_footer_null = 0

        for result in fused_results or []:
            text_content = result.get('text', '')
            if isinstance(text_content, list):
                text_content = " ".join(text_content)
            elif text_content is None:
                text_content = ""

            bbox = result.get('bbox')
            x0 = y0 = x1 = y1 = 0
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = bbox

            visual_label = self._get_visual_label(result)
            final_element_id = result.get('element_id')
            if visual_label in ('page_header', 'page_footer'):
                header_footer_total += 1
                parsed_element_id = self._try_parse_int_id(final_element_id)
                if parsed_element_id is None:
                    parsed_element_id = self._resolve_header_footer_element_id(
                        result,
                        visual_label,
                        header_footer_context
                    )
                final_element_id = parsed_element_id
                result['element_id'] = final_element_id

                if final_element_id is None:
                    header_footer_null += 1
                else:
                    header_footer_mapped += 1

            dev = DokumenElemenVisual(
                dev_ref_tipe=canonical_ref_tipe,
                dev_ref_id=ref_id,
                dev_page=page_num,
                dokumen_elemen_id=final_element_id,
                dev_bbox_x0=float(x0),
                dev_bbox_y0=float(y0),
                dev_bbox_x1=float(x1),
                dev_bbox_y1=float(y1),
                dev_label=result.get('label') or result.get('docling_label'),
                dev_label_struktural=result.get('dev_label_struktural'),
                dev_text=text_content
            )
            db.add(dev)

        if header_footer_total > 0:
            logger.info(
                "Page %s: header/footer rows total=%s mapped=%s null=%s",
                page_num,
                header_footer_total,
                header_footer_mapped,
                header_footer_null
            )

        # SessionLocal is configured with autoflush=False, so flush explicitly to
        # make this page's claims visible to duplicate-claim guard on next pages.
        db.flush()
        return fused_results

    def _is_duplicate_sequence_far(self, alignments, alignment, threshold):
        seq = self._get_alignment_sequence_value(alignment)
        if seq is None:
            return False

        target_y = self._get_alignment_center_y(alignment)
        if target_y is None:
            return False

        prev_seq = None
        next_seq = None
        best_above_delta = None
        best_below_delta = None
        for candidate in alignments or []:
            if candidate is alignment:
                continue
            cand_seq = self._get_alignment_sequence_value(candidate)
            if cand_seq is None:
                continue
            cand_y = self._get_alignment_center_y(candidate)
            if cand_y is None:
                continue
            delta = cand_y - target_y
            if delta < 0:
                delta = abs(delta)
                if best_above_delta is None or delta < best_above_delta:
                    best_above_delta = delta
                    prev_seq = cand_seq
            elif delta > 0:
                if best_below_delta is None or delta < best_below_delta:
                    best_below_delta = delta
                    next_seq = cand_seq

        if prev_seq is None and next_seq is None:
            return False
        if prev_seq is None:
            return (next_seq - seq) > threshold
        if next_seq is None:
            return (seq - prev_seq) > threshold
        return (seq - prev_seq) > threshold or (next_seq - seq) > threshold

    def _collect_duplicate_units_for_page(self, alignments, duplicate_element_ids):
        if not alignments or not duplicate_element_ids:
            return []
        duplicates = []
        for alignment in alignments:
            if alignment.get('element_id') not in duplicate_element_ids:
                continue
            if not self._is_duplicate_sequence_far(
                alignments,
                alignment,
                self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ):
                continue
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    duplicates.extend(
                        unit for unit in cell.get('matched_pdf_units', [])
                        if unit.get('item_type') == 'group'
                    )
            else:
                duplicates.extend(
                    unit for unit in alignment.get('matched_pdf_units', [])
                    if unit.get('item_type') == 'group'
                )
        return duplicates

    def _assign_docling_footnotes(self, db, doc_id, page_num, docling_predictions, footnote_groups):
        if not docling_predictions:
            self._append_footnote_log(doc_id, page_num, "no_docling_predictions")
            return docling_predictions, []

        if not footnote_groups:
            self._append_footnote_log(doc_id, page_num, "no_docling_footnotes")
            return docling_predictions, []

        notes = db.query(DokumenNote).filter(
            DokumenNote.dokumen_id == doc_id,
            DokumenNote.dnote_kind == "footnote"
        ).all()

        if not notes:
            self._append_footnote_log(doc_id, page_num, "no_dokumen_note")
            return docling_predictions, []

        note_candidates = []
        for note in notes:
            dnote_type = (note.dnote_type or '').lower()
            if dnote_type in ("separator", "continuationseparator"):
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "skip_note",
                    note_id=note.dnote_id,
                    delemen_id=note.delemen_id,
                    note_type=note.dnote_type,
                    reason="separator"
                )
                continue
            raw_tree = note.dnote_json_tree
            if isinstance(raw_tree, str):
                try:
                    tree = json.loads(raw_tree)
                except Exception:
                    self._append_footnote_log(
                        doc_id,
                        page_num,
                        "skip_note",
                        note_id=note.dnote_id,
                        delemen_id=note.delemen_id,
                        note_type=note.dnote_type,
                        reason="invalid_json"
                    )
                    continue
            else:
                tree = raw_tree or {}
            if not isinstance(tree, dict):
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "skip_note",
                    note_id=note.dnote_id,
                    delemen_id=note.delemen_id,
                    note_type=note.dnote_type,
                    reason="tree_not_dict"
                )
                continue
            text = self.alignment_service._extract_text_from_json_tree(tree)
            text_norm = self.alignment_service._normalize_text(text)
            if not text_norm:
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "skip_note",
                    note_id=note.dnote_id,
                    delemen_id=note.delemen_id,
                    note_type=note.dnote_type,
                    reason="empty_text"
                )
                continue
            note_candidates.append({
                "note": note,
                "tree": tree,
                "text": text,
                "text_norm": text_norm
            })
            self._append_footnote_log(
                doc_id,
                page_num,
                "note_candidate",
                note_id=note.dnote_id,
                delemen_id=note.delemen_id,
                note_type=note.dnote_type,
                text=text
            )

        if not note_candidates:
            self._append_footnote_log(doc_id, page_num, "no_note_candidates")
            return docling_predictions, []

        best_scores = {}
        candidates = []
        best_scores = {}
        group_norms = {}
        for group_idx, group in enumerate(footnote_groups):
            raw_text = group.get('text') or ''
            doc_text = group.get('docling_pred', {}).get('text') if group.get('docling_pred') else ''
            if isinstance(doc_text, list):
                doc_text = ' '.join(str(t) for t in doc_text)
            text_norm = self.alignment_service._normalize_text(str(raw_text))
            if len(text_norm) < 3:
                text_norm = self.alignment_service._normalize_text(str(doc_text))
            if len(text_norm) < 3:
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "skip_group",
                    docling_idx=group.get('docling_idx'),
                    reason="text_too_short"
                )
                continue
            group_norms[group_idx] = text_norm

        if not group_norms:
            self._append_footnote_log(doc_id, page_num, "no_group_candidates")
            return docling_predictions, []

        for group_idx, text_norm in group_norms.items():
            for note_idx, note_entry in enumerate(note_candidates):
                score = self._compute_text_similarity(text_norm, note_entry["text_norm"])
                best_scores[group_idx] = max(best_scores.get(group_idx, 0.0), score)
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "candidate_score",
                    docling_idx=footnote_groups[group_idx].get('docling_idx'),
                    note_id=note_entry["note"].dnote_id,
                    delemen_id=note_entry["note"].delemen_id,
                    note_type=note_entry["note"].dnote_type,
                    score=round(score, 3),
                    pass_threshold=1 if score >= self.FOOTNOTE_MATCH_MIN_RATIO else 0
                )
                if score >= self.FOOTNOTE_MATCH_MIN_RATIO:
                    candidates.append((score, group_idx, note_idx))

        if not candidates:
            for group_idx in group_norms:
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "no_candidate_above_threshold",
                    docling_idx=footnote_groups[group_idx].get('docling_idx'),
                    best_score=round(best_scores.get(group_idx, 0.0), 3)
                )
            footnote_entries = self._build_footnote_entries(footnote_groups, {})
            filtered_preds = self._filter_docling_predictions(docling_predictions, footnote_groups)
            return filtered_preds, footnote_entries

        candidates.sort(key=lambda x: x[0], reverse=True)
        used_group = set()
        used_note = set()
        matched_groups = {}

        for score, group_idx, note_idx in candidates:
            if group_idx in used_group or note_idx in used_note:
                continue
            group = footnote_groups[group_idx]
            note_entry = note_candidates[note_idx]
            self._append_footnote_log(
                doc_id,
                page_num,
                "match",
                docling_idx=group.get('docling_idx'),
                note_id=note_entry["note"].dnote_id,
                delemen_id=note_entry["note"].delemen_id,
                note_type=note_entry["note"].dnote_type,
                score=round(score, 3),
                docling_text=group.get("docling_pred", {}).get("text"),
                note_text=note_entry["text"],
                group_text=group.get("text")
            )
            matched_groups[group_idx] = note_entry["note"]
            used_group.add(group_idx)
            used_note.add(note_idx)

        for group_idx in group_norms:
            if group_idx not in matched_groups:
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "no_match",
                    docling_idx=footnote_groups[group_idx].get('docling_idx'),
                    best_score=round(best_scores.get(group_idx, 0.0), 3)
                )

        logger.debug(
            "Docling footnotes matched: %s on page %s",
            len(matched_groups),
            page_num
        )

        footnote_entries = self._build_footnote_entries(footnote_groups, matched_groups)
        filtered_preds = self._filter_docling_predictions(docling_predictions, footnote_groups)
        return filtered_preds, footnote_entries

    def _build_footnote_groups(self, extraction_items, docling_predictions, doc_id, page_num):
        footnote_preds = []
        for idx, pred in enumerate(docling_predictions or []):
            label = str(pred.get('label', '')).lower()
            if label in self.FOOTNOTE_LABELS and pred.get('bbox'):
                footnote_preds.append((idx, pred))
                self._append_footnote_log(
                    doc_id,
                    page_num,
                    "docling_footnote",
                    docling_idx=idx,
                    label=label,
                    bbox=pred.get("bbox"),
                    text=pred.get("text")
                )

        if not footnote_preds:
            return [], set()

        pdf_units = self.alignment_service._flatten_extraction_items(extraction_items)
        groups = []
        excluded_item_idxs = set()

        for docling_idx, pred in footnote_preds:
            doc_bbox = pred.get('bbox')
            matched_units = []
            for unit in pdf_units:
                if not unit.get('bbox') or not unit.get('text'):
                    continue
                if unit.get('item_type') in ('table', 'hline_table', 'shape', 'image'):
                    continue
                overlap = self.fusion_service.calculate_overlap(unit['bbox'], doc_bbox)
                if overlap >= self.FOOTNOTE_OVERLAP_THRESHOLD:
                    matched_units.append(unit)
                    excluded_item_idxs.add(unit['item_idx'])

            matched_units.sort(key=lambda x: x['item_idx'])
            merged_bbox = self.alignment_service._merge_bboxes(
                [u.get('bbox') for u in matched_units]
            ) if matched_units else doc_bbox
            merged_text = ' '.join(u.get('text', '') for u in matched_units).strip()
            if not merged_text:
                merged_text = pred.get('text', '')

            groups.append({
                'docling_idx': docling_idx,
                'docling_pred': pred,
                'bbox': merged_bbox,
                'text': merged_text,
                'matched_units': matched_units
            })

            self._append_footnote_log(
                doc_id,
                page_num,
                "footnote_group",
                docling_idx=docling_idx,
                group_units=len(matched_units),
                group_text=merged_text
            )

        return groups, excluded_item_idxs

    def _build_footnote_entries(self, footnote_groups, matched_groups):
        entries = []
        for group_idx, group in enumerate(footnote_groups or []):
            note = matched_groups.get(group_idx)
            pred = group.get('docling_pred') or {}
            label = str(pred.get('label', 'footnote')).lower() or 'footnote'
            entries.append({
                "bbox": group.get("bbox") or pred.get("bbox"),
                "label": "footnote",
                "text": group.get("text") or pred.get("text"),
                "overlap": pred.get("score", 0),
                "source": "note",
                "element_id": note.dnote_id if note else None,
                "note_id": note.dnote_id if note else None,
                "note_kind": note.dnote_kind if note else "footnote",
                "note_type": note.dnote_type if note else None,
                "docling_label": label,
                "merged_count": 1
            })
        return entries

    def _filter_docling_predictions(self, docling_predictions, footnote_groups):
        if not docling_predictions or not footnote_groups:
            return docling_predictions
        remove_idxs = {g.get('docling_idx') for g in footnote_groups if g.get('docling_idx') is not None}
        return [pred for idx, pred in enumerate(docling_predictions) if idx not in remove_idxs]

    def _compute_text_similarity(self, a, b):
        if not a or not b:
            return 0.0
        if a in b or b in a:
            return min(len(a), len(b)) / max(len(a), len(b))
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _append_footnote_log(self, doc_id, page_num, event, **fields):
        os.makedirs(os.path.dirname(self.FOOTNOTE_LOG_PATH), exist_ok=True)

        def sanitize(text):
            if isinstance(text, list):
                text = ' '.join(str(t) for t in text)
            return str(text or '').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')

        timestamp = datetime.now().isoformat(timespec='seconds')
        parts = [timestamp, f"doc_id={doc_id}", f"page={page_num}", f"event={event}"]
        for key, value in fields.items():
            parts.append(f"{key}={sanitize(value)}")
        line = "\t".join(parts) + "\n"
        try:
            with open(self.FOOTNOTE_LOG_PATH, "a", encoding="utf-8") as log_file:
                log_file.write(line)
        except OSError:
            # Footnote trace logging is best-effort and should not fail the pipeline.
            return

    def _save_alignment_results(self, db, alignments, docling_predictions, footnote_entries=None, header_footer_units=None, section_data=None, doc_id=None, page_num=None, structural_state=None):
        """
        Build fused results for visualization and downstream persistence.
        
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

        if footnote_entries:
            fused_results.extend(footnote_entries)
        fused_results, repair_debug = self._repair_picture_fusion_results(
            alignments,
            fused_results,
            docling_predictions=docling_predictions or []
        )
        self._sort_fused_results_in_reading_order(fused_results)

        self._apply_structural_labels(db, fused_results, structural_state=structural_state)

        # Return fused results for visualization
        return fused_results, repair_debug
