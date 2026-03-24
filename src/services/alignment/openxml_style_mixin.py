import json
import logging
import os
import re

from models import DokumenElemen, DokumenSection, DokumenPart, DokumenFormatText, DokumenFormatParagraf

logger = logging.getLogger(__name__)


class AlignmentOpenXmlStyleMixin:


    def _collect_format_ids(self, json_tree):
        dftx_ids = set()
        dfp_ids = set()

        def walk(node):
            if isinstance(node, dict):
                if 'dftx_id' in node:
                    parsed = self._safe_int(node.get('dftx_id'))
                    if parsed is not None:
                        dftx_ids.add(parsed)
                if 'dfp_id' in node:
                    parsed = self._safe_int(node.get('dfp_id'))
                    if parsed is not None:
                        dfp_ids.add(parsed)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(json_tree)
        return dftx_ids, dfp_ids

    @staticmethod
    def _normalize_hint_token(value):
        if value is None:
            return ''
        text = str(value).strip().lower()
        if not text:
            return ''
        return text.replace('"', '').replace("'", '')

    def _extract_style_hints_from_json_tree(self, json_tree):
        font_families = set()
        style_ids = set()

        def add_font(value):
            normalized = self._normalize_hint_token(value)
            if not normalized:
                return
            parts = re.split(r'[;,/]+', normalized)
            for part in parts:
                part = part.strip()
                if part:
                    font_families.add(part)

        def add_style(value):
            normalized = self._normalize_hint_token(value)
            if normalized:
                style_ids.add(normalized)

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    key_norm = str(key).lower()
                    if any(marker in key_norm for marker in self.FONT_KEY_MARKERS):
                        if isinstance(value, str):
                            add_font(value)
                    if 'style' in key_norm and isinstance(value, str):
                        add_style(value)
                    walk(value)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(json_tree)
        return font_families, style_ids

    def _extract_style_hints_from_format_ids(
        self,
        dftx_ids,
        dfp_ids,
        text_format_cache=None,
        paragraph_format_cache=None
    ):
        font_families = set()
        style_ids = set()
        text_cache = text_format_cache or {}
        paragraph_cache = paragraph_format_cache or {}

        for dftx_id in dftx_ids or []:
            text_fmt = text_cache.get(dftx_id) or {}
            font_ascii = text_fmt.get('dftx_font_ascii')
            normalized = self._normalize_hint_token(font_ascii)
            if normalized:
                font_families.add(normalized)

        for dfp_id in dfp_ids or []:
            para_fmt = paragraph_cache.get(dfp_id) or {}
            style_id = para_fmt.get('dfp_p_style_id')
            normalized = self._normalize_hint_token(style_id)
            if normalized:
                style_ids.add(normalized)

        return font_families, style_ids

    @staticmethod
    def _row_to_mapping(row):
        if row is None:
            return {}
        if isinstance(row, dict):
            return row
        if hasattr(row, '_mapping'):
            return dict(row._mapping)
        return {}

    def _prefetch_format_cache(self, db_session, elements, page_seq_range=None):
        cache = {'text': {}, 'paragraph': {}}
        if not db_session or not elements:
            return cache

        seq_min = seq_max = None
        if page_seq_range and len(page_seq_range) == 2:
            seq_min, seq_max = page_seq_range

        dftx_ids = set()
        dfp_ids = set()

        for elem in elements:
            elem_seq = getattr(elem, 'delemen_sequence', None)
            if seq_min is not None and seq_max is not None:
                if elem_seq is None or elem_seq < seq_min or elem_seq > seq_max:
                    continue
            tree = self._parse_json_tree(getattr(elem, 'delemen_json_tree', None))
            text_ids, para_ids = self._collect_format_ids(tree)
            dftx_ids.update(text_ids)
            dfp_ids.update(para_ids)

        if dftx_ids:
            try:
                rows = db_session.query(
                    DokumenFormatText.dftx_id,
                    DokumenFormatText.dftx_font_ascii,
                    DokumenFormatText.dftx_bold,
                    DokumenFormatText.dftx_italic,
                    DokumenFormatText.dftx_underline
                ).filter(
                    DokumenFormatText.dftx_id.in_(tuple(dftx_ids))
                ).all()
                for row in rows:
                    row_id = self._safe_int(row.dftx_id)
                    if row_id is None:
                        continue
                    cache['text'][row_id] = {
                        'dftx_font_ascii': row.dftx_font_ascii,
                        'dftx_bold': row.dftx_bold,
                        'dftx_italic': row.dftx_italic,
                        'dftx_underline': row.dftx_underline,
                    }
            except Exception as exc:
                logger.warning("Failed prefetch dokumen_format_text: %s", exc)

        if dfp_ids:
            try:
                rows = db_session.query(
                    DokumenFormatParagraf.dfp_id,
                    DokumenFormatParagraf.dfp_p_style_id
                ).filter(
                    DokumenFormatParagraf.dfp_id.in_(tuple(dfp_ids))
                ).all()
                for row in rows:
                    row_id = self._safe_int(row.dfp_id)
                    if row_id is None:
                        continue
                    cache['paragraph'][row_id] = {
                        'dfp_p_style_id': row.dfp_p_style_id,
                    }
            except Exception as exc:
                logger.warning("Failed prefetch dokumen_format_paragraf: %s", exc)

        return cache

    def _extract_openxml_style_hints(
        self,
        json_tree,
        text_format_cache=None,
        paragraph_format_cache=None
    ):
        json_fonts, json_styles = self._extract_style_hints_from_json_tree(json_tree)
        dftx_ids, dfp_ids = self._collect_format_ids(json_tree)
        format_fonts, format_styles = self._extract_style_hints_from_format_ids(
            dftx_ids,
            dfp_ids,
            text_format_cache=text_format_cache,
            paragraph_format_cache=paragraph_format_cache
        )

        font_families = sorted(json_fonts | format_fonts)
        style_ids = sorted(json_styles | format_styles)

        is_code_font = any(
            any(marker in font_name for marker in self.CODE_FONT_MARKERS)
            for font_name in font_families
        )
        is_code_style = any(
            any(marker in style_id for marker in self.CODE_STYLE_MARKERS)
            for style_id in style_ids
        )

        return {
            'font_families': font_families,
            'style_ids': style_ids,
            'is_code_font': is_code_font,
            'is_code_style': is_code_style,
            'is_code_like_openxml': bool(is_code_font or is_code_style),
        }

    def _extract_table_cells(self, json_tree):
        if json_tree is None:
            return []
        cells = []
        content = json_tree.get('content', {}) if isinstance(json_tree, dict) else {}
        if isinstance(content, dict):
            rows = content.get('rows', [])
        else:
            rows = json_tree.get('rows', []) if isinstance(json_tree, dict) else []

        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for col_idx, cell in enumerate(row.get('cells', [])):
                cell_text = self._extract_cell_text(cell)
                if cell_text.strip():
                    cells.append({'row': row_idx, 'col': col_idx, 'text': cell_text})
        return cells

    def _extract_cell_text(self, cell):
        if isinstance(cell, str):
            return cell
        if isinstance(cell, dict):
            if cell.get('type') == 'text' and 'value' in cell:
                return str(cell['value'])
            return self._extract_text_from_json_tree(cell)
        if isinstance(cell, list):
            texts = []
            for item in cell:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'value' in item:
                        texts.append(str(item['value']))
                    elif item.get('type') == 'math' and 'text' in item:
                        texts.append(str(item['text']))
                    else:
                        texts.append(self._extract_text_from_json_tree(item))
                elif isinstance(item, str):
                    texts.append(item)
            return ' '.join(texts)
        return ""

    def _extract_text_from_json_tree(self, json_tree):
        """Recursively extract text from dokumen_elemen_json_tree.

        Images are converted to context-based placeholders [IMG:abc123] where the hash
        is derived from surrounding text. This enables matching even when image counts
        differ between PDF and Word (e.g., charts in Word but not in PDF).
        Ported from legacy dokumen_elemen_routes.py.
        """
        if json_tree is None:
            return ""

        # First pass: collect all items in order (text and images)
        items = []

        def collect_items(node):
            if isinstance(node, dict):
                # Check if this is an image type element
                if node.get('type') == 'image':
                    items.append({'type': 'image'})
                    return

                # Check for text content
                if node.get('type') == 'text' and 'value' in node:
                    items.append({'type': 'text', 'value': str(node['value'])})
                    return
                if 'value' in node and node.get('type') != 'image':
                    items.append({'type': 'text', 'value': str(node['value'])})
                if 'text' in node:
                    items.append({'type': 'text', 'value': str(node['text'])})
                if 't' in node:
                    items.append({'type': 'text', 'value': str(node['t'])})
                if 'content' in node:
                    if isinstance(node['content'], str):
                        items.append({'type': 'text', 'value': node['content']})
                    else:
                        collect_items(node['content'])
                # Recurse through all values
                for key, value in node.items():
                    if key not in ['text', 't', 'content', 'value', 'type', 'rId']:
                        collect_items(value)
            elif isinstance(node, list):
                for item in node:
                    collect_items(item)

        collect_items(json_tree)

        # Second pass: generate count-based placeholders for images
        # Use simple numbering: [IMG:1], [IMG:2], etc. based on order in element
        result_parts = []
        image_counter = 0

        for item in items:
            if item['type'] == 'text':
                result_parts.append(item['value'])
            elif item['type'] == 'image':
                image_counter += 1
                result_parts.append(f'[IMG:{image_counter}]')

        return ' '.join(result_parts).strip()

    def _extract_text_and_images_separately(self, json_tree):
        if json_tree is None:
            return {'text_only': '', 'images': [], 'has_images': False, 'combined': '', 'ordered_items': []}

        items = []

        def collect_items(node):
            if isinstance(node, dict):
                if node.get('type') == 'image':
                    items.append({'type': 'image'})
                    return
                if node.get('type') == 'text' and 'value' in node:
                    items.append({'type': 'text', 'value': str(node['value'])})
                    return
                if 'value' in node and node.get('type') != 'image':
                    items.append({'type': 'text', 'value': str(node['value'])})
                if 'text' in node:
                    items.append({'type': 'text', 'value': str(node['text'])})
                if 't' in node:
                    items.append({'type': 'text', 'value': str(node['t'])})
                if 'content' in node:
                    if isinstance(node['content'], str):
                        items.append({'type': 'text', 'value': node['content']})
                    else:
                        collect_items(node['content'])
                for key, value in node.items():
                    if key not in ['text', 't', 'content', 'value', 'type', 'rId']:
                        collect_items(value)
            elif isinstance(node, list):
                for item in node:
                    collect_items(item)

        collect_items(json_tree)

        text_parts = []
        combined_parts = []
        images = []
        ordered_items = []
        image_counter = 0

        for item in items:
            if item['type'] == 'text':
                text_parts.append(item['value'])
                combined_parts.append(item['value'])
                ordered_items.append({'type': 'text', 'value': item['value']})
            elif item['type'] == 'image':
                image_counter += 1
                placeholder = f'[IMG:{image_counter}]'
                images.append({'placeholder': placeholder, 'index': image_counter})
                combined_parts.append(placeholder)
                ordered_items.append({'type': 'image', 'local_index': image_counter})

        return {
            'text_only': ' '.join(text_parts).strip(),
            'images': images,
            'has_images': len(images) > 0,
            'combined': ' '.join(combined_parts).strip(),
            'ordered_items': ordered_items
        }
