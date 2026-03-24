import json
import logging
import os
import re

from models import DokumenElemen, DokumenSection, DokumenPart, DokumenFormatText, DokumenFormatParagraf

logger = logging.getLogger(__name__)


class AlignmentOpenXmlUnitsMixin:


    def _build_openxml_units(self, elements, page_seq_range=None, db_session=None, format_cache=None):
        units = []
        table_debug = []
        global_image_counter = 0
        toc_stub_sequences = self._collect_toc_stub_sequences(elements)
        block_state = {
            'current_block': {},
            'block_order': 0,
            'last_key': None,
            'last_kind': None,
        }
        active_format_cache = format_cache
        if active_format_cache is None:
            active_format_cache = self._prefetch_format_cache(
                db_session,
                elements,
                page_seq_range=page_seq_range
            )
        text_format_cache = active_format_cache.get('text', {})
        paragraph_format_cache = active_format_cache.get('paragraph', {})

        def apply_block_metadata(unit_payload, unit_text, **kwargs):
            metadata = self._derive_block_metadata(
                unit_text,
                current_block=block_state.get('current_block'),
                **kwargs,
            )
            current_key = metadata.get('block_key')
            current_kind = metadata.get('block_kind')
            current_role = metadata.get('content_role')
            start_new_block = False
            if current_role in {'heading', 'continuation_heading'}:
                start_new_block = True
            elif current_key and current_key != block_state.get('last_key'):
                start_new_block = True
            elif current_kind in {'table', 'figure', 'caption'} and current_kind != block_state.get('last_kind'):
                start_new_block = True
            elif block_state.get('block_order') <= 0:
                start_new_block = True

            if start_new_block:
                block_state['block_order'] += 1
            block_order = block_state['block_order']

            block_state['current_block'] = metadata.get('current_block') or {}
            if current_key:
                block_state['last_key'] = current_key
            if current_kind:
                block_state['last_kind'] = current_kind

            unit_payload['block_kind'] = current_kind
            unit_payload['block_key'] = current_key
            unit_payload['content_role'] = current_role
            unit_payload['block_order'] = block_order
            return unit_payload

        for elem in elements:
            if elem.delemen_sequence in toc_stub_sequences:
                continue
            json_tree = self._parse_json_tree(elem.delemen_json_tree)

            elem_has_shape = self._has_shape_content(json_tree)
            style_hints = self._extract_openxml_style_hints(
                json_tree,
                text_format_cache=text_format_cache,
                paragraph_format_cache=paragraph_format_cache
            )
            is_openxml_chart = self._is_openxml_chart_element(json_tree)
            is_openxml_visual_slot = self._is_openxml_visual_slot_element(
                '',
                elem.delemen_type,
                style_hints.get('style_ids', []),
                is_openxml_chart=is_openxml_chart
            )

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
                            'is_openxml_visual_slot': False,
                        })
                        units[-1] = apply_block_metadata(
                            units[-1],
                            text,
                            elem_type=elem.delemen_type,
                            style_ids=style_hints.get('style_ids', []),
                            is_table=True,
                            is_code_like=style_hints.get('is_code_like_openxml', False),
                        )
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
                        'is_openxml_visual_slot': False,
                    })
                    units[-1] = apply_block_metadata(
                        units[-1],
                        '',
                        elem_type=elem.delemen_type,
                        style_ids=style_hints.get('style_ids', []),
                        is_table=True,
                        is_code_like=style_hints.get('is_code_like_openxml', False),
                    )
                table_debug.append(table_info)
            else:
                content = self._extract_text_and_images_separately(json_tree)
                if content['has_images']:
                    image_only_visual_slot = (
                        self._is_env_enabled_default_true("ALIGNMENT_ENABLE_IMAGE_PLACEHOLDER_VISUAL_SLOT")
                        and self._is_image_placeholder_only_text(
                            content.get('combined') or content.get('text_only') or ''
                        )
                    )
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
                                'is_openxml_visual_slot': image_only_visual_slot,
                            })
                            units[-1] = apply_block_metadata(
                                units[-1],
                                ph,
                                elem_type=elem.delemen_type,
                                style_ids=style_hints.get('style_ids', []),
                                is_chart=is_openxml_chart,
                                is_visual_slot=image_only_visual_slot,
                                is_image_part=True,
                                is_code_like=style_hints.get('is_code_like_openxml', False),
                            )
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
                                    'is_openxml_visual_slot': False,
                                })
                                units[-1] = apply_block_metadata(
                                    units[-1],
                                    content['text_only'],
                                    elem_type=elem.delemen_type,
                                    style_ids=style_hints.get('style_ids', []),
                                    is_code_like=style_hints.get('is_code_like_openxml', False),
                                )
                                text_unit_created = True
                else:
                    text = content['combined'] if content['combined'] else self._extract_text_from_json_tree(json_tree)
                    is_openxml_visual_slot = self._is_openxml_visual_slot_element(
                        text,
                        elem.delemen_type,
                        style_hints.get('style_ids', []),
                        is_openxml_chart=is_openxml_chart
                    )
                    if is_openxml_chart and self._is_chart_caption_text(text):
                        units.append({
                            'unit_id': f"{elem.delemen_id}_chart",
                            'elem_id': elem.delemen_id,
                            'elem_seq': elem.delemen_sequence,
                            'elem_type': elem.delemen_type,
                            'text': '',
                            'text_normalized': '',
                            'is_cell': False,
                            'has_shape': True,
                            'is_image_part': True,
                            'font_families': style_hints.get('font_families', []),
                            'style_ids': style_hints.get('style_ids', []),
                            'is_code_font': style_hints.get('is_code_font', False),
                            'is_code_style': style_hints.get('is_code_style', False),
                            'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                            'is_openxml_chart': True,
                            'is_openxml_visual_slot': False,
                            'is_chart_caption_text': False,
                        })
                        units[-1] = apply_block_metadata(
                            units[-1],
                            '',
                            elem_type=elem.delemen_type,
                            style_ids=style_hints.get('style_ids', []),
                            is_chart=True,
                            is_image_part=True,
                            is_code_like=style_hints.get('is_code_like_openxml', False),
                        )
                        units.append({
                            'unit_id': f"{elem.delemen_id}_caption",
                            'elem_id': elem.delemen_id,
                            'elem_seq': elem.delemen_sequence,
                            'elem_type': elem.delemen_type,
                            'text': text,
                            'text_normalized': self._normalize_text(text).rstrip('.:'),
                            'is_cell': False,
                            'is_text_part': True,
                            'has_shape': False,
                            'font_families': style_hints.get('font_families', []),
                            'style_ids': style_hints.get('style_ids', []),
                            'is_code_font': style_hints.get('is_code_font', False),
                            'is_code_style': style_hints.get('is_code_style', False),
                            'is_code_like_openxml': style_hints.get('is_code_like_openxml', False),
                            'is_openxml_chart': False,
                            'is_openxml_visual_slot': False,
                            'is_chart_caption_text': True,
                        })
                        units[-1] = apply_block_metadata(
                            units[-1],
                            text,
                            elem_type=elem.delemen_type,
                            style_ids=style_hints.get('style_ids', []),
                            is_caption_text=True,
                            is_code_like=style_hints.get('is_code_like_openxml', False),
                        )
                    else:
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
                            'is_openxml_visual_slot': is_openxml_visual_slot,
                            'is_chart_caption_text': False,
                        })
                        units[-1] = apply_block_metadata(
                            units[-1],
                            text,
                            elem_type=elem.delemen_type,
                            style_ids=style_hints.get('style_ids', []),
                            is_chart=is_openxml_chart,
                            is_visual_slot=is_openxml_visual_slot,
                            is_code_like=style_hints.get('is_code_like_openxml', False),
                        )
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
                'is_openxml_visual_slot': all_units[i].get('is_openxml_visual_slot', False),
                'block_kind': all_units[i].get('block_kind'),
                'block_key': all_units[i].get('block_key'),
                'content_role': all_units[i].get('content_role'),
                'block_order': all_units[i].get('block_order'),
            }
            for i in indices
        ]
