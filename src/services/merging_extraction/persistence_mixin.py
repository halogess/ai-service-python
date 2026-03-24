import difflib
import json
import logging
import os
import re
from datetime import datetime

from sqlalchemy.orm import Session

from models import (
    Bab,
    Dokumen,
    DokumenElemen,
    DokumenElemenVisual,
    DokumenFormatParagraf,
    DokumenFormatText,
    DokumenNote,
    DokumenPart,
    DokumenSection,
)
from utils.cross_page_claims import analyze_cross_page_entries

logger = logging.getLogger(__name__)


class MergingExtractionPersistenceMixin:
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

            allowed_pair_results = self._select_valid_same_page_chart_caption_results(current_results)
            allowed_pair_result_ids = {id(result) for result in allowed_pair_results}
            scoring_indices = [
                idx for idx in indices
                if id(fused_results[idx]) not in allowed_pair_result_ids
            ]
            if not scoring_indices:
                picture_indices = [
                    idx for idx in indices
                    if self._is_picture_result(fused_results[idx])
                ]
                scoring_indices = picture_indices or list(indices)

            best_current_idx = max(scoring_indices, key=lambda i: self._visual_result_claim_score(fused_results[i]))
            best_current_score = self._visual_result_claim_score(fused_results[best_current_idx])
            best_existing_score = max((claim.get('score') or (0.0, 0.0, 0)) for claim in far_claims)

            if best_current_score > best_existing_score:
                for idx in indices:
                    if (
                        idx == best_current_idx or
                        id(fused_results[idx]) in allowed_pair_result_ids or
                        fused_results[idx].get('element_id') is None
                    ):
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
        fused_results = [
            result for result in (fused_results or [])
            if not (result or {}).get('_drop_from_output')
        ]

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

    def _collect_duplicate_units_for_page(self, alignments, duplicate_analysis, page_num):
        if not alignments or not duplicate_analysis:
            return []
        current_page = self._try_parse_int_id(page_num)
        if current_page is None:
            return []
        duplicates = []
        for alignment in alignments:
            elem_id = self._try_parse_int_id(alignment.get('element_id'))
            if elem_id is None:
                continue
            analysis = (duplicate_analysis or {}).get(elem_id) or {}
            if not analysis.get('is_invalid_duplicate'):
                continue
            if current_page not in set(analysis.get('invalid_pages') or []):
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
