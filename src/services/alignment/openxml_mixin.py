import json
import logging
import re

from models import DokumenElemen, DokumenSection, DokumenPart, DokumenFormatText, DokumenFormatParagraf

logger = logging.getLogger(__name__)


class AlignmentOpenXmlMixin:
    CODE_FONT_MARKERS = (
        'courier',
        'lucida',
        'consola',
        'monospace',
        'menlo',
        'monaco',
        'fira code',
        'source code',
        'jetbrains mono',
        'inconsolata',
        'cascadia',
        'terminal',
    )
    CODE_STYLE_MARKERS = (
        'code',
        'algoritma',
        'algorithm',
        'segmenprogram',
        'segmen_program',
        'programcontent',
        'listing',
        'source',
        'monospace',
    )
    FONT_KEY_MARKERS = (
        'font',
        'rfonts',
        'ascii',
        'hansi',
        'eastasia',
        'typeface',
    )

    @staticmethod
    def _resolve_ref_tipe_for_read(ref_tipe: str):
        if ref_tipe in ('bab', 'buku'):
            return ('bab', 'buku')
        return (ref_tipe,)

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

    def _build_openxml_units(self, elements, page_seq_range=None, db_session=None, format_cache=None):
        units = []
        table_debug = []
        global_image_counter = 0
        active_format_cache = format_cache
        if active_format_cache is None:
            active_format_cache = self._prefetch_format_cache(
                db_session,
                elements,
                page_seq_range=page_seq_range
            )
        text_format_cache = active_format_cache.get('text', {})
        paragraph_format_cache = active_format_cache.get('paragraph', {})

        for elem in elements:
            json_tree = self._parse_json_tree(elem.delemen_json_tree)

            elem_has_shape = self._has_shape_content(json_tree)
            style_hints = self._extract_openxml_style_hints(
                json_tree,
                text_format_cache=text_format_cache,
                paragraph_format_cache=paragraph_format_cache
            )
            is_openxml_chart = self._is_openxml_chart_element(json_tree)

            if self._is_table_element(elem.delemen_type):
                cells = self._extract_table_cells(json_tree)
                table_info = {
                    'elem_id': elem.delemen_id,
                    'cells_count': len(cells),
                    'has_shape': elem_has_shape,
                    'units_created': 0,
                    'action': ''
                }

                if cells:
                    table_info['action'] = f'created {len(cells)} cell units'
                    table_info['units_created'] = len(cells)
                    for cell in cells:
                        text = cell['text']
                        unit_id = f"{elem.delemen_id}_r{cell['row']}_c{cell['col']}"
                        units.append({
                            'unit_id': unit_id,
                            'elem_id': elem.delemen_id,
                            'elem_seq': elem.delemen_sequence,
                            'elem_type': elem.delemen_type,
                            'text': text,
                            'text_normalized': self._normalize_text(text).rstrip('.:'),
                            'is_cell': True,
                            'row': cell['row'],
                            'col': cell['col'],
                            'has_shape': elem_has_shape,
                            'font_families': style_hints.get('font_families', []),
                            'style_ids': style_hints.get('style_ids', []),
                            'is_code_font': style_hints.get('is_code_font', False),
                            'is_code_style': style_hints.get('is_code_style', False),
                            'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                            'is_openxml_chart': is_openxml_chart,
                        })
                elif elem_has_shape:
                    table_info['action'] = 'created shape placeholder'
                    table_info['units_created'] = 1
                    units.append({
                        'unit_id': str(elem.delemen_id),
                        'elem_id': elem.delemen_id,
                        'elem_seq': elem.delemen_sequence,
                        'elem_type': elem.delemen_type,
                        'text': '',
                        'text_normalized': '',
                        'is_cell': False,
                        'row': None,
                        'col': None,
                        'has_shape': True,
                        'font_families': style_hints.get('font_families', []),
                        'style_ids': style_hints.get('style_ids', []),
                        'is_code_font': style_hints.get('is_code_font', False),
                        'is_code_style': style_hints.get('is_code_style', False),
                        'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                        'is_openxml_chart': is_openxml_chart,
                    })
                table_debug.append(table_info)
            else:
                content = self._extract_text_and_images_separately(json_tree)
                if content['has_images']:
                    text_unit_created = False
                    for item in content['ordered_items']:
                        if item['type'] == 'image':
                            global_image_counter += 1
                            ph = '[IMG]'
                            units.append({
                                'unit_id': f"{elem.delemen_id}_img{global_image_counter}",
                                'elem_id': elem.delemen_id,
                                'elem_seq': elem.delemen_sequence,
                                'elem_type': elem.delemen_type,
                                'text': ph,
                                'text_normalized': ph.lower(),
                                'is_cell': False,
                                'image_index': global_image_counter,
                                'is_text_part': False,
                                'is_image_part': True,
                                'has_shape': True,
                                'font_families': style_hints.get('font_families', []),
                                'style_ids': style_hints.get('style_ids', []),
                                'is_code_font': style_hints.get('is_code_font', False),
                                'is_code_style': style_hints.get('is_code_style', False),
                                'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                                'is_openxml_chart': is_openxml_chart,
                            })
                        elif item['type'] == 'text' and not text_unit_created:
                            if content['text_only']:
                                units.append({
                                    'unit_id': f"{elem.delemen_id}_text",
                                    'elem_id': elem.delemen_id,
                                    'elem_seq': elem.delemen_sequence,
                                    'elem_type': elem.delemen_type,
                                    'text': content['text_only'],
                                    'text_normalized': self._normalize_text(content['text_only']).rstrip('.:'),
                                    'is_cell': False,
                                    'is_text_part': True,
                                    'has_shape': elem_has_shape,
                                    'font_families': style_hints.get('font_families', []),
                                    'style_ids': style_hints.get('style_ids', []),
                                    'is_code_font': style_hints.get('is_code_font', False),
                                    'is_code_style': style_hints.get('is_code_style', False),
                                    'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                                    'is_openxml_chart': is_openxml_chart,
                                })
                                text_unit_created = True
                else:
                    text = content['combined'] if content['combined'] else self._extract_text_from_json_tree(json_tree)
                    units.append({
                        'unit_id': str(elem.delemen_id),
                        'elem_id': elem.delemen_id,
                        'elem_seq': elem.delemen_sequence,
                        'elem_type': elem.delemen_type,
                        'text': text,
                        'text_normalized': self._normalize_text(text).rstrip('.:'),
                        'is_cell': False,
                        'has_shape': elem_has_shape,
                        'font_families': style_hints.get('font_families', []),
                        'style_ids': style_hints.get('style_ids', []),
                        'is_code_font': style_hints.get('is_code_font', False),
                        'is_code_style': style_hints.get('is_code_style', False),
                        'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                        'is_openxml_chart': is_openxml_chart,
                    })
        return units, table_debug

    def _format_unaligned_openxml(self, all_units, indices):
        return [
            {
                'openxml_unit_id': all_units[i]['unit_id'],
                'elem_id': all_units[i]['elem_id'],
                'elem_type': all_units[i]['elem_type'],
                'text': all_units[i]['text'],
                'text_normalized': all_units[i]['text_normalized'],
                'is_cell': all_units[i]['is_cell'],
                'row': all_units[i].get('row'),
                'col': all_units[i].get('col'),
                'has_shape': all_units[i].get('has_shape', False),
                'font_families': all_units[i].get('font_families', []),
                'style_ids': all_units[i].get('style_ids', []),
                'is_code_font': all_units[i].get('is_code_font', False),
                'is_code_style': all_units[i].get('is_code_style', False),
                'is_code_like_openxml': all_units[i].get('is_code_like_openxml', False),
                'is_openxml_chart': all_units[i].get('is_openxml_chart', False),
            }
            for i in indices
        ]
