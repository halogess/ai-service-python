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


class MergingExtractionStructuralLabelsMixin:
    def _is_caption_continuation_candidate(self, result):
        visual_label = self._get_visual_label(result)
        if visual_label not in ('text', 'section_header'):
            return False
        text = self._coerce_text((result or {}).get('text')).strip()
        if not text:
            return False
        if self._text_starts_with_bab(text) or self._is_subchapter_title(text):
            return False
        if self._get_text_list_marker(text):
            return False
        if visual_label == 'section_header':
            text_norm = ' '.join(text.split())
            if len(text_norm) > 48 or len(text_norm.split()) > 7:
                return False
        return True

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
            if not structural_label and not is_subchapter_text and not is_bab_heading_text:
                code_like_lines = self._count_following_code_like_lines(
                    fused_results,
                    idx,
                    allow_title_bridges=visual_label in ('caption', 'text', 'section_header'),
                )
                if is_section_header and code_like_lines >= 2:
                    structural_label = 'judul_kode'
                elif (
                    visual_label in ('caption', 'text', 'section_header')
                    and code_like_lines >= 1
                    and self._is_code_title_like_text(text)
                ):
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
                is_heading_like_type = bool(
                    elem_type_norm and (
                        re.fullmatch(r'h\d+', elem_type_norm) or
                        'heading' in elem_type_norm or
                        'header' in elem_type_norm or
                        'title' in elem_type_norm
                    )
                )
                if is_list_item_type:
                    is_list_candidate = True
                elif visual_label in ('section_header', 'list_item'):
                    is_list_candidate = True
                elif visual_label == 'text' and is_heading_like_type and self._get_text_list_marker(text):
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
                    if not self._is_caption_continuation_candidate(next_result):
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
