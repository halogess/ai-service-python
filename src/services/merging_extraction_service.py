
import os
import json
import logging
import difflib
import re
from sqlalchemy import text
from datetime import datetime
from sqlalchemy.orm import Session
from models import Dokumen, DokumenSection, DokumenPart, DokumenElemen, DokumenElemenVisual, DokumenNote
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
        4. Save results to database (DokumenElemenVisual) [Optional]
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
                footnote_groups, footnote_item_idxs = self._build_footnote_groups(
                    extraction_items, page_docling_preds, doc_id, page_num
                )
                if footnote_item_idxs:
                    extraction_items = [
                        item for idx, item in enumerate(extraction_items)
                        if idx not in footnote_item_idxs
                    ]
                
                # Perform Alignment with cross-page tracking
                alignment_result = self.alignment_service.align(
                    doc_id, page_num, extraction_items, 
                    page_width, page_height, total_pages, 
                    min_openxml_idx=max_openxml_idx  # Use previous page's max
                )
                
                if alignment_result['success']:
                    # Update cross-page tracking from alignment result (never allow backtracking)
                    new_max_openxml_idx = alignment_result.get('max_openxml_idx')
                    if new_max_openxml_idx is not None:
                        if new_max_openxml_idx < max_openxml_idx:
                            logger.warning(
                                f"Page {page_num}: max_openxml_idx backtracked "
                                f"({new_max_openxml_idx} < {max_openxml_idx}); keeping previous value."
                            )
                        else:
                            max_openxml_idx = new_max_openxml_idx
                    logger.debug(f"Page {page_num}: max_openxml_idx updated to {max_openxml_idx}")
                    
                    alignments = alignment_result['final_alignments']
                    header_footer_units = alignment_result.get('header_footer_units', [])
                    section_data = alignment_result.get('page_debug', {}).get('section_data')
                    page_docling_preds, footnote_entries = self._assign_docling_footnotes(
                        db, doc_id, page_num, page_docling_preds, footnote_groups
                    )
                    
                    # Save alignment results with header_footer_units for proper Docling fusion
                    fused_results = self._save_alignment_results(
                        db, 
                        alignments, 
                        page_docling_preds,
                        footnote_entries=footnote_entries,
                        header_footer_units=header_footer_units,
                        section_data=section_data,
                        doc_id=doc_id,
                        page_num=page_num,
                        structural_state=structural_state
                    )

                    if save_to_db and not generate_visualizations:
                        self._replace_visual_records(
                            db,
                            doc_id,
                            page_num,
                            fused_results,
                            structural_state=structural_state
                        )
                    
                    # Generate visualizations if enabled
                    if generate_visualizations:
                        all_pdf_units = self.alignment_service._flatten_extraction_items(extraction_items)
                        unaligned_units = alignment_result.get('unaligned_pdf_units', [])
                        unfused_units = self._collect_unfused_pdf_units(
                            all_pdf_units,
                            fused_results,
                            unaligned_units
                        )
                        unaligned_for_vis = unaligned_units + unfused_units

                        page_vis_payload[page_num] = {
                            'alignments': alignments,
                            'fused_results': fused_results,
                            'header_footer_units': header_footer_units,
                            'unaligned_pdf_units': unaligned_for_vis,
                            'raw_docling': page_docling_preds
                        }
                
            extractor.close()

            if generate_visualizations and page_vis_payload:
                try:
                    duplicate_element_ids = self._collect_duplicate_openxml_element_ids(page_vis_payload)
                    for page_num, payload in page_vis_payload.items():
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

                        if save_to_db:
                            self._replace_visual_records(
                                db,
                                doc_id,
                                page_num,
                                payload.get('fused_results'),
                                structural_state=structural_state
                            )

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
                            doc_id=doc_id,
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
                                    'doc_id': doc_id,
                                    'fused_results': payload.get('fused_results'),
                                    'raw_docling': payload.get('raw_docling'),
                                    'alignments': payload.get('alignments')
                                }, f, indent=2, ensure_ascii=False)
                except Exception as vis_err:
                    logger.warning(f"Visualization/JSON save failed - {vis_err}")
            
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
            row = db.execute(
                text("SELECT dfp_jc FROM dokumen_format_paragraf WHERE dfp_id = :dfp_id"),
                {"dfp_id": dfp_id}
            ).fetchone()
            if row and row[0] is not None:
                alignment = self._normalize_alignment_value(row[0])
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
                rows = db.execute(
                    text(
                        "SELECT dftx_id, dftx_bold "
                        "FROM dokumen_format_text "
                        "WHERE dftx_id IN :ids"
                    ),
                    {"ids": tuple(missing)}
                ).fetchall()
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
            if result.get('source') != 'alignment':
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

            if is_picture and alignments_for_elem:
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

    def _replace_visual_records(self, db, doc_id, page_num, fused_results, structural_state=None):
        if not db or doc_id is None or page_num is None:
            return
        if fused_results:
            self._apply_structural_labels(
                db,
                fused_results,
                structural_state=structural_state,
                skip_if_labeled=True
            )
        db.query(DokumenElemenVisual).filter(
            DokumenElemenVisual.dokumen_id == doc_id,
            DokumenElemenVisual.dev_page == page_num
        ).delete(synchronize_session=False)

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

            dev = DokumenElemenVisual(
                dokumen_id=doc_id,
                dev_page=page_num,
                dokumen_elemen_id=result.get('element_id'),
                dev_bbox_x0=float(x0),
                dev_bbox_y0=float(y0),
                dev_bbox_x1=float(x1),
                dev_bbox_y1=float(y1),
                dev_label=result.get('label') or result.get('docling_label'),
                dev_label_struktural=result.get('dev_label_struktural'),
                dev_text=text_content
            )
            db.add(dev)

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
        with open(self.FOOTNOTE_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(line)

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

        self._apply_structural_labels(db, fused_results, structural_state=structural_state)

        # Return fused results for visualization
        return fused_results
