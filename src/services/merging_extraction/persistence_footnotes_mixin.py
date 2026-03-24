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


class MergingExtractionPersistenceFootnotesMixin:


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
