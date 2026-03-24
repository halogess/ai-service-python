import json
import logging
import os
import re

from models import DokumenElemen, DokumenSection, DokumenPart, DokumenFormatText, DokumenFormatParagraf

logger = logging.getLogger(__name__)


class AlignmentOpenXmlSectionsMixin:


    @staticmethod
    def _resolve_ref_tipe_for_read(ref_tipe: str):
        if ref_tipe in ('bab', 'buku'):
            return ('bab', 'buku')
        return (ref_tipe,)

    @staticmethod
    def _is_env_enabled_default_true(env_name):
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    def _get_openxml_elements(self, db_session, ref_id: int, ref_tipe: str = 'dokumen'):
        ref_tipes = self._resolve_ref_tipe_for_read(ref_tipe)
        return db_session.query(DokumenElemen).join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id).filter(
            DokumenPart.dsec_id.in_(
                db_session.query(DokumenSection.dsec_id).filter(
                    DokumenSection.dsec_ref_tipe.in_(ref_tipes),
                    DokumenSection.dsec_ref_id == ref_id
                )
            ),
            DokumenPart.dpart_type == 'body'
        ).order_by(DokumenElemen.delemen_sequence).all()

    def _get_doc_sections(self, db_session, ref_id: int, ref_tipe: str = 'dokumen'):
        ref_tipes = self._resolve_ref_tipe_for_read(ref_tipe)
        return db_session.query(DokumenSection).filter(
            DokumenSection.dsec_ref_tipe.in_(ref_tipes),
            DokumenSection.dsec_ref_id == ref_id
        ).order_by(DokumenSection.dsec_index).all()

    def _get_section_for_page(self, sections, page_width, page_height):
        if not sections or not page_width or not page_height:
            return None
        twips_per_point = 20
        for sec in sections:
            sec_width = (sec.dsec_page_width_twips or 0) / twips_per_point
            sec_height = (sec.dsec_page_height_twips or 0) / twips_per_point
            if abs(sec_width - page_width) < 10 and abs(sec_height - page_height) < 10:
                return sec
        return None

    def _estimate_page_sequence_range(self, elements, page_num, total_pages):
        total_elements = len(elements)
        if total_pages < 1:
            total_pages = 1
        elements_per_page = max(1, total_elements // total_pages)
        buffer = max(10, elements_per_page // 2)

        if elements:
            all_sequences = sorted([e.delemen_sequence for e in elements])
            if all_sequences:
                start_idx = max(0, min((page_num - 1) * elements_per_page - buffer, len(all_sequences) - 1))
                end_idx = max(0, min(page_num * elements_per_page + buffer, len(all_sequences) - 1))
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                return (all_sequences[start_idx], all_sequences[end_idx])
        return None

    def _collect_toc_stub_sequences(self, elements):
        if not elements:
            return set()

        infos = []
        for element in elements:
            if isinstance(element, dict):
                seq = element.get('delemen_sequence', element.get('sequence'))
                elem_type = element.get('delemen_type', element.get('type'))
                raw_tree = element.get('delemen_json_tree')
                text = element.get('openxml_text')
            else:
                seq = getattr(element, 'delemen_sequence', None)
                elem_type = getattr(element, 'delemen_type', None)
                raw_tree = getattr(element, 'delemen_json_tree', None)
                text = None

            try:
                seq = int(seq) if seq is not None else None
            except (TypeError, ValueError):
                seq = None
            if seq is None:
                continue

            if text is None:
                text = self._extract_text_from_json_tree(self._parse_json_tree(raw_tree))
            text_norm = self._normalize_text(text or '')
            infos.append({
                'seq': seq,
                'type': str(elem_type or '').strip().lower(),
                'text_norm': text_norm,
            })

        infos.sort(key=lambda item: item['seq'])
        accepted = set()
        run = []

        def flush_run():
            nonlocal run
            if len(run) >= 4 and run[0]['num'] == 1 and run[0]['seq'] <= 12:
                accepted.update(item['seq'] for item in run)
            run = []

        for info in infos:
            if not info['type'].startswith('list-item'):
                flush_run()
                continue
            match = self.TOC_BAB_STUB_RE.match(info['text_norm'])
            if not match:
                flush_run()
                continue

            item = {
                'seq': info['seq'],
                'num': int(match.group(1)),
            }
            if not run:
                run = [item]
                continue

            prev = run[-1]
            if item['seq'] == prev['seq'] + 1 and item['num'] == prev['num'] + 1:
                run.append(item)
                continue

            flush_run()
            run = [item]

        flush_run()
        return accepted

    def _has_shape_content(self, json_tree):
        if not json_tree:
            return False
        if isinstance(json_tree, dict):
            if json_tree.get('type') == 'shape':
                return True
            for v in json_tree.values():
                if self._has_shape_content(v):
                    return True
        elif isinstance(json_tree, list):
            for i in json_tree:
                if self._has_shape_content(i):
                    return True
        return False

    def _is_openxml_chart_element(self, json_tree):
        def walk(node):
            if isinstance(node, dict):
                node_type = str(node.get('type') or '').strip().lower()
                if node_type == 'chart':
                    return True
                for value in node.values():
                    if walk(value):
                        return True
                return False
            if isinstance(node, list):
                for child in node:
                    if walk(child):
                        return True
            return False

        return walk(json_tree)

    def _is_openxml_visual_slot_element(self, text, elem_type, style_ids, is_openxml_chart=False):
        if is_openxml_chart:
            return False
        element_type = str(elem_type or '').strip().lower()
        if not element_type or 'paragraph' not in element_type:
            return False
        raw_text = str(text or '').strip()
        if (
            self._is_env_enabled_default_true("ALIGNMENT_ENABLE_IMAGE_PLACEHOLDER_VISUAL_SLOT")
            and self._is_image_placeholder_only_text(raw_text)
        ):
            return True
        if self._normalize_text(raw_text).strip():
            return False
        style_tokens = {
            self._normalize_hint_token(style_id).replace(' ', '')
            for style_id in (style_ids or [])
            if style_id is not None
        }
        return 'gambarlampiran' in style_tokens

    def _is_image_placeholder_only_text(self, text):
        if not text:
            return False
        return bool(self.IMAGE_PLACEHOLDER_ONLY_RE.match(str(text).strip()))

    def _is_chart_caption_text(self, text):
        if not text:
            return False
        return bool(self.CHART_CAPTION_TEXT_RE.match(str(text).strip()))

    def _is_table_element(self, etype):
        return etype in ['table', 'grid_table']

    @staticmethod
    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_json_tree(self, raw_tree):
        if isinstance(raw_tree, str):
            try:
                return json.loads(raw_tree)
            except Exception:
                return {}
        if raw_tree is None:
            return {}
        return raw_tree
