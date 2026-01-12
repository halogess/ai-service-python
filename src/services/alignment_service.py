
import difflib
import re
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from models import DokumenElemen, DokumenSection, DokumenPart
from database import SessionLocal

class AlignmentService:
    def __init__(self):
        pass

    def align_document(self, extraction_results: List[Dict], doc_id: int) -> List[Dict]:
        """
        Align all pages of a document.
        
        Args:
            extraction_results: List of extraction results per page
            doc_id: Document ID
            
        Returns:
            List of alignment results per page
        """
        results = []
        min_openxml_idx = 0
        
        for page_data in extraction_results:
            page_num = page_data.get('page', 1)
            items = page_data.get('items', [])
            page_width = page_data.get('page_width', 595)
            page_height = page_data.get('page_height', 842)
            total_pages = len(extraction_results)
            
            result = self.align(
                doc_id, page_num, items,
                page_width, page_height, total_pages,
                min_openxml_idx
            )
            
            # Update min_openxml_idx for next page
            if result.get('success'):
                min_openxml_idx = result.get('max_openxml_idx', min_openxml_idx)
            
            results.append({
                'success': result.get('success', False),
                'page': page_num,
                'alignments': result.get('alignments', []),
                'unaligned_pdf_units': result.get('unaligned_pdf_units', []),
                'header_footer_units': result.get('header_footer_units', []),
                'max_openxml_idx': result.get('max_openxml_idx', 0),
                'stats': {
                    'aligned_count': len(result.get('alignments', [])),
                    'unaligned_count': len(result.get('unaligned_pdf_units', []))
                }
            })
        
        return results

    def align(self, doc_id: int, page_num: int, extraction_items: List[Dict], 
              page_width: float, page_height: float, total_pages: int,
              min_openxml_idx: int = 0) -> Dict[str, Any]:
        """
        Main entry point for alignment.
        Orchestrates extraction flattening, OpenXML retrieval, alignment, and full post-processing.
        Matches `api_merging_alignment` + `perform_two_pass_alignment` from legacy.
        """
        db = SessionLocal()
        try:
            # 1. Get Section Data for margin logic
            sections = self._get_doc_sections(db, doc_id)
            current_section = self._get_section_for_page(sections, page_width, page_height)
            if not current_section and sections:
                current_section = sections[0]

            # Build section data (matches legacy /dokumen-elemen-api/sections payload)
            section_data = None
            if current_section:
                twips_per_point = 20
                section_data = {
                    'dsec_id': current_section.dsec_id,
                    'dsec_index': current_section.dsec_index,
                    'page_width_twips': current_section.dsec_page_width_twips,
                    'page_height_twips': current_section.dsec_page_height_twips,
                    'page_width_pt': current_section.dsec_page_width_twips / twips_per_point if current_section.dsec_page_width_twips else None,
                    'page_height_pt': current_section.dsec_page_height_twips / twips_per_point if current_section.dsec_page_height_twips else None,
                    'orientation': current_section.dsec_orientation,
                    'margin_top_twips': current_section.dsec_margin_top_twips,
                    'margin_bottom_twips': current_section.dsec_margin_bottom_twips,
                    'margin_left_twips': current_section.dsec_margin_left_twips,
                    'margin_right_twips': current_section.dsec_margin_right_twips,
                    'margin_top_pt': current_section.dsec_margin_top_twips / twips_per_point if current_section.dsec_margin_top_twips else None,
                    'margin_bottom_pt': current_section.dsec_margin_bottom_twips / twips_per_point if current_section.dsec_margin_bottom_twips else None,
                    'margin_left_pt': current_section.dsec_margin_left_twips / twips_per_point if current_section.dsec_margin_left_twips else None,
                    'margin_right_pt': current_section.dsec_margin_right_twips / twips_per_point if current_section.dsec_margin_right_twips else None,
                    'header_margin_twips': current_section.dsec_header_margin_twips,
                    'footer_margin_twips': current_section.dsec_footer_margin_twips,
                    'header_margin_pt': current_section.dsec_header_margin_twips / twips_per_point if current_section.dsec_header_margin_twips else None,
                    'footer_margin_pt': current_section.dsec_footer_margin_twips / twips_per_point if current_section.dsec_footer_margin_twips else None,
                    'gutter_twips': current_section.dsec_gutter_twips,
                    'gutter_position': current_section.dsec_gutter_position
                }

            # 2. Flatten Extraction Items (PDF Units)
            all_pdf_units = self._flatten_extraction_items(extraction_items)

            # 3. Filter Header/Footer units
            pdf_units, header_footer_units = self._filter_header_footer_items(all_pdf_units, current_section, page_height)

            # 4. Get OpenXML elements (body parts only)
            elements = self._get_openxml_elements(db, doc_id)
            
            # 5. Build OpenXML Units (with image numbering logic)
            # Estimate sequence range for page to number images correctly
            page_sequence_range = self._estimate_page_sequence_range(elements, page_num, total_pages)
            openxml_units, table_debug = self._build_openxml_units(elements, page_sequence_range)

            # 6. Perform Two-Pass Alignment (Feature Complete)
            alignment_result = self._perform_two_pass_alignment(pdf_units, openxml_units, min_openxml_idx)
            
            # Add table debug info to page debug
            alignment_result['debug_info']['table_processing'] = table_debug
            alignment_result['debug_info']['section_data'] = section_data

            return {
                'success': True,
                'phase1_alignments': alignment_result['phase1_alignments'],
                'alignments': alignment_result['final_alignments'], # Backward compat alias
                'final_alignments': alignment_result['final_alignments'],
                'shape_alignments': alignment_result['shape_alignments'],
                'unaligned_pdf_units': [pdf_units[i] for i in alignment_result['unaligned_final'] if i < len(pdf_units)],
                'unaligned_pdf_units_phase1': alignment_result['unaligned_after_phase1'],
                'unaligned_openxml_units': self._format_unaligned_openxml(openxml_units, alignment_result['unaligned_openxml']),
                'header_footer_units': header_footer_units,
                'max_openxml_idx': alignment_result.get('max_openxml_idx', 0),
                'page_debug': alignment_result['debug_info']
            }
        finally:
            db.close()

    def _get_openxml_elements(self, db_session, doc_id: int):
        return db_session.query(DokumenElemen).join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id).filter(
            DokumenPart.dsec_id.in_(
                db_session.query(DokumenSection.dsec_id).filter(DokumenSection.dokumen_id == doc_id)
            ),
            DokumenPart.dpart_type == 'body'
        ).order_by(DokumenElemen.delemen_sequence).all()

    def _get_doc_sections(self, db_session, doc_id: int):
        return db_session.query(DokumenSection).filter_by(dokumen_id=doc_id).order_by(DokumenSection.dsec_index).all()

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
        if total_pages < 1: total_pages = 1
        elements_per_page = max(1, total_elements // total_pages)
        buffer = max(10, elements_per_page // 2)
        
        if elements:
            all_sequences = sorted([e.delemen_sequence for e in elements])
            if all_sequences:
                start_idx = max(0, min((page_num - 1) * elements_per_page - buffer, len(all_sequences) - 1))
                end_idx = max(0, min(page_num * elements_per_page + buffer, len(all_sequences) - 1))
                if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
                return (all_sequences[start_idx], all_sequences[end_idx])
        return None

    def _flatten_extraction_items(self, extraction_items):
        collected = []
        for item_idx, item in enumerate(extraction_items):
            itype = item.get('type', '')
            idata = item.get('data', {})
            ibbox = item.get('bbox')
            
            if itype == 'group':
                text = idata.get('text', '')
                if text.strip():
                    collected.append({'item_idx': item_idx, 'item_type': itype, 'text': text, 'bbox': ibbox, 'is_cell': False})
            elif itype == 'paragraph':
                 text = idata.get('text', '')
                 if text.strip():
                    collected.append({'item_idx': item_idx, 'item_type': itype, 'text': text, 'bbox': ibbox, 'is_cell': False})
            elif itype == 'table':
                for r_idx, row in enumerate(idata.get('rows', [])):
                    for c_idx, cell in enumerate(row.get('cells', [])):
                        ctext = self._extract_cell_content_text(cell)
                        if ctext.strip():
                             collected.append({'item_idx': item_idx, 'item_type': itype, 'text': ctext, 'bbox': cell.get('bbox'), 'is_cell': True, 'row': r_idx, 'col': c_idx, 'table_bbox': ibbox})
            elif itype == 'hline_table':
                # Handle hline_table: prefer cells if available
                cells = idata.get('cells', [])
                rows = idata.get('rows', [])
                if rows:
                    all_text = []
                    for r_idx, row in enumerate(rows):
                        for c_idx, cell in enumerate(row.get('cells', [])):
                            ctext = self._extract_cell_content_text(cell)
                            if ctext.strip(): all_text.append(ctext.strip())
                    if all_text:
                         collected.append({'item_idx': item_idx, 'item_type': itype, 'text': ' '.join(all_text), 'bbox': ibbox, 'is_cell': False, 'is_hline_table_unit': True})
            elif itype == 'shape':
                text = idata.get('text', '')
                image_bbox = idata.get('image_bbox')
                if text.strip() or image_bbox:
                     # If it's an image-shape with no text, use [IMG] placeholder
                     if not text.strip() and image_bbox:
                         text = '[IMG]'
                     collected.append({'item_idx': item_idx, 'item_type': itype, 'text': text, 'bbox': ibbox, 'is_cell': False})
            elif itype == 'image':
                collected.append({'item_idx': item_idx, 'item_type': itype, 'text': None, 'bbox': ibbox, 'is_cell': False, 'is_image': True})

        # Merge consecutive shapes
        collected = self._merge_consecutive_shape_items(collected)
        
        pdf_units = []
        unit_counter = 0
        img_counter = 0
        for item in collected:
            if item.get('is_image'):
                img_counter += 1
                ph = '[IMG]'
                pdf_units.append({
                    'unit_id': f'pdf_{unit_counter}', 'item_idx': item['item_idx'], 'item_type': item['item_type'],
                    'text': ph, 'text_normalized': ph.lower(), 'bbox': item['bbox'], 'is_cell': False, 'is_page_number': False
                })
            else:
                txt = item['text']
                pdf_units.append({
                    'unit_id': f'pdf_{unit_counter}', 'item_idx': item['item_idx'], 'item_type': item['item_type'],
                    'text': txt, 'text_normalized': self._normalize_text(txt), 'bbox': item['bbox'],
                    'is_cell': item.get('is_cell', False), 'row': item.get('row'), 'col': item.get('col'),
                    'is_hline_table_unit': item.get('is_hline_table_unit', False),
                    'is_page_number': self._is_likely_page_number(txt, item['bbox']) if not item.get('is_cell') else False
                })
            unit_counter += 1
        return pdf_units

    def _extract_cell_content_text(self, cell):
        texts = []
        content = cell.get('content', [])
        image_counter = 0
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get('type') == 'text' and c.get('text'): texts.append(c.get('text'))
                    elif c.get('type') == 'image': 
                        image_counter += 1
                        texts.append(f'[IMG:{image_counter}]')
        return ' '.join(texts)

    def _merge_consecutive_shape_items(self, items):
        merged = []
        idx = 0
        while idx < len(items):
            item = items[idx]
            if item.get('item_type') != 'shape':
                merged.append(item)
                idx += 1
                continue
            cluster = [item]
            idx += 1
            while idx < len(items) and items[idx].get('item_type') == 'shape':
                cluster.append(items[idx])
                idx += 1
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                mtext = ' '.join(i.get('text', '') for i in cluster).strip()
                mbbox = self._merge_bboxes([i.get('bbox') for i in cluster])
                merged.append({
                    'item_idx': cluster[0]['item_idx'], 'item_type': 'shape', 'text': mtext, 'bbox': mbbox,
                    'is_cell': False, 'row': None, 'col': None
                })
        return merged

    def _merge_bboxes(self, bboxes):
        valid = [b for b in bboxes if b and len(b) >= 4]
        if not valid: return None
        return [min(b[0] for b in valid), min(b[1] for b in valid), max(b[2] for b in valid), max(b[3] for b in valid)]

    def _is_item_in_header_footer_zone(self, bbox, section_data, page_height_pt=842):
        if not bbox or len(bbox) < 4 or not section_data: return False, False
        twips = 20
        margin_top = (getattr(section_data, 'dsec_margin_top_twips', 0) or 0) / twips
        margin_bottom = (getattr(section_data, 'dsec_margin_bottom_twips', 0) or 0) / twips
        sec_h = getattr(section_data, 'dsec_page_height_twips', 0)
        ph = (sec_h / twips) if sec_h else page_height_pt
        yc = (bbox[1] + bbox[3]) / 2
        return yc < margin_top, yc > (ph - margin_bottom)

    def _filter_header_footer_items(self, pdf_units, section, page_height):
        if not section: return pdf_units, []
        filtered, hf = [], []
        for u in pdf_units:
            ish, isf = self._is_item_in_header_footer_zone(u.get('bbox'), section, page_height)
            if ish or isf:
                u['is_header_footer'] = True
                u['zone'] = 'header' if ish else 'footer'
                hf.append(u)
            else:
                u['is_header_footer'] = False
                filtered.append(u)
        return filtered, hf

    def _has_shape_content(self, json_tree):
        if not json_tree: return False
        if isinstance(json_tree, dict):
            if json_tree.get('type') == 'shape': return True
            for v in json_tree.values():
                if self._has_shape_content(v): return True
        elif isinstance(json_tree, list):
            for i in json_tree:
                if self._has_shape_content(i): return True
        return False

    def _is_table_element(self, etype):
        return etype in ['table', 'grid_table']

    def _extract_table_cells(self, json_tree):
        if json_tree is None: return []
        cells = []
        content = json_tree.get('content', {}) if isinstance(json_tree, dict) else {}
        if isinstance(content, dict): rows = content.get('rows', [])
        else: rows = json_tree.get('rows', []) if isinstance(json_tree, dict) else []
        
        for row_idx, row in enumerate(rows):
            if not isinstance(row, dict): continue
            for col_idx, cell in enumerate(row.get('cells', [])):
                cell_text = self._extract_cell_text(cell)
                if cell_text.strip():
                    cells.append({'row': row_idx, 'col': col_idx, 'text': cell_text})
        return cells

    def _extract_cell_text(self, cell):
        if isinstance(cell, str): return cell
        if isinstance(cell, dict):
            if cell.get('type') == 'text' and 'value' in cell: return str(cell['value'])
            return self._extract_text_from_json_tree(cell)
        if isinstance(cell, list):
            texts = []
            for item in cell:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'value' in item: texts.append(str(item['value']))
                    elif item.get('type') == 'math' and 'text' in item: texts.append(str(item['text']))
                    else: texts.append(self._extract_text_from_json_tree(item))
                elif isinstance(item, str): texts.append(item)
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
        
        for i, item in enumerate(items):
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
                    else: collect_items(node['content'])
                for key, value in node.items():
                    if key not in ['text', 't', 'content', 'value', 'type', 'rId']:
                        collect_items(value)
            elif isinstance(node, list):
                for item in node: collect_items(item)
        
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

    def _build_openxml_units(self, elements, page_seq_range=None):
        units = []
        table_debug = []
        global_image_counter = 0
        
        for elem in elements:
            # CRITICAL: Parse JSON tree from string (stored as TEXT in database)
            json_tree = elem.delemen_json_tree
            if isinstance(json_tree, str):
                try:
                    import json
                    json_tree = json.loads(json_tree)
                except:
                    json_tree = {}
            elif json_tree is None:
                json_tree = {}
            
            elem_has_shape = self._has_shape_content(json_tree)
            
            if self._is_table_element(elem.delemen_type):
                cells = self._extract_table_cells(json_tree)
                table_info = {
                    'elem_id': elem.delemen_id, 'cells_count': len(cells), 'has_shape': elem_has_shape,
                    'units_created': 0, 'action': ''
                }
                
                if cells:
                    table_info['action'] = f'created {len(cells)} cell units'
                    table_info['units_created'] = len(cells)
                    for cell in cells:
                        text = cell['text']
                        unit_id = f"{elem.delemen_id}_r{cell['row']}_c{cell['col']}"
                        units.append({
                            'unit_id': unit_id, 'elem_id': elem.delemen_id, 'elem_seq': elem.delemen_sequence,
                            'elem_type': elem.delemen_type, 'text': text,
                            'text_normalized': self._normalize_text(text).rstrip('.:'),
                            'is_cell': True, 'row': cell['row'], 'col': cell['col'], 'has_shape': elem_has_shape
                        })
                elif elem_has_shape:
                    table_info['action'] = 'created shape placeholder'
                    table_info['units_created'] = 1
                    units.append({
                        'unit_id': str(elem.delemen_id), 'elem_id': elem.delemen_id, 'elem_seq': elem.delemen_sequence,
                        'elem_type': elem.delemen_type, 'text': '', 'text_normalized': '',
                        'is_cell': False, 'row': None, 'col': None, 'has_shape': True
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
                                'elem_id': elem.delemen_id, 'elem_seq': elem.delemen_sequence, 'elem_type': elem.delemen_type,
                                'text': ph, 'text_normalized': ph.lower(), 'is_cell': False, 'image_index': global_image_counter,
                                'is_text_part': False, 'is_image_part': True, 'has_shape': True
                            })
                        elif item['type'] == 'text' and not text_unit_created:
                            if content['text_only']:
                                units.append({
                                    'unit_id': f"{elem.delemen_id}_text",
                                    'elem_id': elem.delemen_id, 'elem_seq': elem.delemen_sequence, 'elem_type': elem.delemen_type,
                                    'text': content['text_only'], 'text_normalized': self._normalize_text(content['text_only']).rstrip('.:'),
                                    'is_cell': False, 'is_text_part': True, 'has_shape': elem_has_shape
                                })
                                text_unit_created = True
                else:
                    text = content['combined'] if content['combined'] else self._extract_text_from_json_tree(json_tree)
                    units.append({
                        'unit_id': str(elem.delemen_id), 'elem_id': elem.delemen_id, 'elem_seq': elem.delemen_sequence,
                        'elem_type': elem.delemen_type, 'text': text, 'text_normalized': self._normalize_text(text).rstrip('.:'),
                        'is_cell': False, 'has_shape': elem_has_shape
                    })
        return units, table_debug

    def _perform_two_pass_alignment(self, pdf_units, openxml_units, min_openxml_idx):
        p1_align, p1_un_pdf, _, p1_debug = self._perform_char_alignment(pdf_units, openxml_units, min_openxml_idx)
        
        final_align = list(p1_align)
        final_align.sort(key=lambda x: x.get('element_sequence') or 0)
        final_un_pdf = list(p1_un_pdf)
        
        final_align, final_un_pdf = self._absorb_unaligned_into_alignments(final_align, final_un_pdf, pdf_units)
        
        p1_un_ox = p1_debug.get('unaligned_openxml_indices', [])
        final_align, final_un_pdf, _ = self._match_remaining_with_unaligned_openxml(
            final_align, final_un_pdf, p1_un_ox, pdf_units, openxml_units
        )

        final_align = self._cleanup_punctuation_alignments(final_align)
        final_align, shape_conflict_debug = self._resolve_shape_alignment_conflicts(final_align, pdf_units)
        final_align, final_un_pdf, shape_attach_debug = self._attach_shape_clusters_to_next_alignment(final_align, final_un_pdf, pdf_units)

        # Legacy debug fields
        p1_debug['pass2_shape_debug'] = []
        p1_debug['pass2_shape_matched'] = 0
        p1_debug['pass2_consumed_pdf'] = []
        p1_debug['shape_openxml_count'] = 0
        p1_debug['non_shape_openxml_count'] = len(openxml_units)
        p1_debug['shape_conflict_debug'] = shape_conflict_debug
        p1_debug['shape_conflict_count'] = len(shape_conflict_debug)
        p1_debug['shape_attach_debug'] = shape_attach_debug
        p1_debug['shape_attach_count'] = len(shape_attach_debug)

        max_idx = min_openxml_idx
        if p1_debug.get('max_openxml_idx'): max_idx = p1_debug['max_openxml_idx']

        # Filter unaligned OpenXML to only those within this page's sequence range
        page_unaligned_openxml = p1_un_ox
        if p1_align and p1_un_ox:
            aligned_sequences = [a.get('element_sequence') or 0 for a in p1_align]
            min_seq = min(aligned_sequences)
            max_seq = max(aligned_sequences)
            page_unaligned_openxml = [
                idx for idx in p1_un_ox
                if min_seq <= (openxml_units[idx].get('elem_seq') or 0) <= max_seq
            ]

        return {
            'phase1_alignments': p1_align,
            'final_alignments': final_align,
            'shape_alignments': [],
            'unaligned_after_phase1': p1_un_pdf,
            'unaligned_final': final_un_pdf,
            'unaligned_openxml': page_unaligned_openxml,
            'debug_info': p1_debug,
            'max_openxml_idx': max_idx
        }

    def _perform_char_alignment(self, pdf_units, openxml_units, min_openxml_idx=0):
        if not pdf_units or not openxml_units:
            return [], list(range(len(pdf_units))), list(range(len(openxml_units))), {
                'max_openxml_idx': min_openxml_idx,
                'unaligned_openxml_indices': list(range(len(openxml_units)))
            }

        pdf_concat = ''
        pdf_char_map = []
        pdf_unit_ranges = []
        page_number_indices = set()

        for i, u in enumerate(pdf_units):
            if u.get('is_page_number', False):
                page_number_indices.add(i)
            text = u['text_normalized']
            start = len(pdf_concat)
            for _ in text:
                pdf_char_map.append(i)
            pdf_concat += text
            if text:
                pdf_unit_ranges.append({
                    'unit_idx': i,
                    'unit_id': u['unit_id'],
                    'start': start,
                    'end': len(pdf_concat),
                    'text': u['text'][:50],
                    'text_normalized': text[:50],
                    'item_type': u['item_type']
                })

        openxml_concat = ''
        openxml_char_map = []
        openxml_unit_ranges = []

        for i, u in enumerate(openxml_units):
            text = u['text_normalized']
            start = len(openxml_concat)
            for _ in text:
                openxml_char_map.append(i)
            openxml_concat += text
            if text:
                openxml_unit_ranges.append({
                    'unit_idx': i,
                    'unit_id': u['unit_id'],
                    'start': start,
                    'end': len(openxml_concat),
                    'text': u['text'][:50],
                    'text_normalized': text[:50],
                    'elem_type': u['elem_type']
                })

        sm = difflib.SequenceMatcher(None, pdf_concat, openxml_concat, autojunk=False)
        matching_blocks = sm.get_matching_blocks()
        sorted_blocks = sorted(matching_blocks, key=lambda x: x.b)

        # Log gap analysis to file (legacy behavior)
        with open('gap_analysis.log', 'w', encoding='utf-8') as gap_log:
            gap_log.write("=" * 80 + "\n")
            gap_log.write("GAP ANALYSIS - What OpenXML content is NOT being matched\n")
            gap_log.write("=" * 80 + "\n\n")

            prev_end_ox = 0
            for i, block in enumerate(sorted_blocks):
                if block.size == 0:
                    continue
                gap = block.b - prev_end_ox
                if gap > 50:
                    gap_log.write(f"\n[GAP {i}] OX positions {prev_end_ox} to {block.b} (size: {gap} chars)\n")
                    gap_content = openxml_concat[prev_end_ox:block.b]
                    gap_log.write(f"  Content: \"{gap_content[:200]}...\"\n")
                    gap_units = []
                    for unit_range in openxml_unit_ranges:
                        if unit_range['start'] < block.b and unit_range['end'] > prev_end_ox:
                            gap_units.append(unit_range)
                    gap_log.write(f"  Units in gap: {len(gap_units)}\n")
                    for u in gap_units[:5]:
                        gap_log.write(f"    U{u['unit_idx']}: {u['elem_type']} \"{u['text'][:40]}...\"\n")
                gap_log.write(f"\nBlock {i}: OX[{block.b}], PDF[{block.a}], size={block.size}\n")
                gap_log.write(f"  Matched text: \"{pdf_concat[block.a:block.a + min(block.size, 50)]}...\"\n")
                prev_end_ox = block.b + block.size

        consumed_openxml_positions = set()
        pdf_unit_assignment = {}
        openxml_to_pdf = {}
        match_debug = {}
        matching_log = []
        traversal_log = []

        for block_idx, block in enumerate(sorted_blocks):
            if block.size == 0:
                continue

            block_log = {
                'block_num': block_idx,
                'pdf_start': block.a,
                'openxml_start': block.b,
                'size': block.size,
                'matched_text': pdf_concat[block.a:block.a + min(block.size, 30)],
                'matches': []
            }

            for offset in range(block.size):
                pdf_char_idx = block.a + offset
                openxml_char_idx = block.b + offset

                char = pdf_concat[pdf_char_idx] if pdf_char_idx < len(pdf_concat) else '?'
                pdf_idx = pdf_char_map[pdf_char_idx] if pdf_char_idx < len(pdf_char_map) else -1
                openxml_idx = openxml_char_map[openxml_char_idx] if openxml_char_idx < len(openxml_char_map) else -1

                log_entry = {
                    'step': len(traversal_log),
                    'block': block_idx,
                    'offset': offset,
                    'char': char,
                    'pdf_char_idx': pdf_char_idx,
                    'openxml_char_idx': openxml_char_idx,
                    'pdf_unit': pdf_idx,
                    'openxml_unit': openxml_idx,
                    'pdf_unit_id': pdf_units[pdf_idx]['unit_id'] if 0 <= pdf_idx < len(pdf_units) else None,
                    'openxml_unit_id': openxml_units[openxml_idx]['unit_id'] if 0 <= openxml_idx < len(openxml_units) else None,
                    'action': None,
                    'reason': None
                }

                if openxml_char_idx in consumed_openxml_positions:
                    log_entry['action'] = 'SKIP'
                    log_entry['reason'] = 'openxml_pos_consumed'
                    traversal_log.append(log_entry)
                    continue

                if pdf_char_idx < len(pdf_char_map) and openxml_char_idx < len(openxml_char_map):
                    is_shape_pdf = False
                    if 0 <= pdf_idx < len(pdf_units):
                        is_shape_pdf = pdf_units[pdf_idx].get('item_type') == 'shape'

                    if pdf_idx in pdf_unit_assignment and not is_shape_pdf:
                        if pdf_unit_assignment[pdf_idx] != openxml_idx:
                            log_entry['action'] = 'SKIP'
                            log_entry['reason'] = f'pdf_assigned_to_different: {pdf_unit_assignment[pdf_idx]}'
                            traversal_log.append(log_entry)
                            continue
                        log_entry['reason'] = 'continue_existing_assignment'
                    else:
                        if openxml_idx < min_openxml_idx:
                            log_entry['action'] = 'SKIP'
                            log_entry['reason'] = f'cross_page_backward: openxml_idx={openxml_idx} < min_from_prev_page={min_openxml_idx}'
                            traversal_log.append(log_entry)
                            continue

                        if not is_shape_pdf:
                            backward_violation = False
                            violation_reason = None

                            for other_pdf_idx, other_openxml_idx in pdf_unit_assignment.items():
                                if pdf_idx > other_pdf_idx and openxml_idx < other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] > pdf[{other_pdf_idx}] but openxml[{openxml_idx}] < openxml[{other_openxml_idx}]'
                                    break
                                if pdf_idx < other_pdf_idx and openxml_idx > other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] < pdf[{other_pdf_idx}] but openxml[{openxml_idx}] > openxml[{other_openxml_idx}]'
                                    break

                            if backward_violation:
                                log_entry['action'] = 'SKIP'
                                log_entry['reason'] = f'backward_match_prevented: {violation_reason}'
                                traversal_log.append(log_entry)
                                continue

                            pdf_unit_assignment[pdf_idx] = openxml_idx
                            log_entry['reason'] = 'new_assignment'
                        else:
                            log_entry['reason'] = 'shape_multi_match'

                    consumed_openxml_positions.add(openxml_char_idx)

                    if openxml_idx not in openxml_to_pdf:
                        openxml_to_pdf[openxml_idx] = {}
                    if pdf_idx not in openxml_to_pdf[openxml_idx]:
                        openxml_to_pdf[openxml_idx][pdf_idx] = 0
                    openxml_to_pdf[openxml_idx][pdf_idx] += 1

                    log_entry['action'] = 'MATCH'
                    log_entry['matched_count'] = openxml_to_pdf[openxml_idx][pdf_idx]
                    traversal_log.append(log_entry)

                    debug_key = (openxml_idx, pdf_idx)
                    if debug_key not in match_debug:
                        match_debug[debug_key] = {'matched_chars': []}
                    match_debug[debug_key]['matched_chars'].append(pdf_concat[pdf_char_idx])

                    if len(block_log['matches']) < 5:
                        block_log['matches'].append({
                            'char': pdf_concat[pdf_char_idx],
                            'pdf_unit': pdf_idx,
                            'openxml_unit': openxml_idx
                        })

            if block_log['matches']:
                matching_log.append(block_log)

        unit_matching_summary = []
        for i, u in enumerate(pdf_units):
            matched_to = []
            for openxml_idx, pdf_counts in openxml_to_pdf.items():
                if i in pdf_counts:
                    matched_to.append({
                        'openxml_unit_idx': openxml_idx,
                        'openxml_unit_id': openxml_units[openxml_idx]['unit_id'],
                        'matched_chars': pdf_counts[i]
                    })

            unit_matching_summary.append({
                'pdf_unit_idx': i,
                'unit_id': u['unit_id'],
                'item_type': u['item_type'],
                'text': u['text'][:30],
                'consumed': i in pdf_unit_assignment,
                'is_page_number': u.get('is_page_number', False),
                'matched_to': matched_to
            })

        suspicious_page_numbers = self._detect_suspicious_page_numbers(pdf_units, pdf_unit_assignment, openxml_to_pdf)

        for entry in unit_matching_summary:
            entry['is_suspicious_page_number'] = entry['pdf_unit_idx'] in suspicious_page_numbers

        filtered_openxml_to_pdf = {}
        for openxml_idx, pdf_counts in openxml_to_pdf.items():
            filtered_counts = {
                pdf_idx: count for pdf_idx, count in pdf_counts.items()
                if pdf_idx not in suspicious_page_numbers
            }
            if filtered_counts:
                filtered_openxml_to_pdf[openxml_idx] = filtered_counts

        alignments = self._build_alignments_from_matching(
            filtered_openxml_to_pdf, pdf_units, openxml_units, match_debug
        )

        unaligned_pdf_indices = [
            i for i in range(len(pdf_units))
            if i not in pdf_unit_assignment and i not in suspicious_page_numbers
        ]

        unaligned_openxml_indices = [
            i for i in range(len(openxml_units))
            if i not in filtered_openxml_to_pdf
        ]

        page_number_list = list(page_number_indices | suspicious_page_numbers)

        debug_info = {
            'pdf_concat_len': len(pdf_concat),
            'openxml_concat_len': len(openxml_concat),
            'pdf_concat_sample': pdf_concat[:200],
            'openxml_concat_sample': openxml_concat[:200],
            'pdf_unit_ranges': pdf_unit_ranges,
            'openxml_unit_ranges': openxml_unit_ranges,
            'matching_blocks_count': len(matching_blocks),
            'matching_blocks': [
                {
                    'block_num': i, 'pdf_pos': b.a, 'openxml_pos': b.b, 'size': b.size,
                    'text': pdf_concat[b.a:b.a + min(b.size, 50)]
                }
                for i, b in enumerate(matching_blocks) if b.size > 0
            ][:30],
            'matching_log': matching_log[:20],
            'traversal_log': traversal_log,
            'traversal_log_count': len(traversal_log),
            'unit_matching_summary': unit_matching_summary,
            'consumed_pdf_count': len(pdf_unit_assignment),
            'page_number_indices': page_number_list,
            'suspicious_page_numbers': list(suspicious_page_numbers),
            'unaligned_pdf_count': len(unaligned_pdf_indices),
            'unaligned_openxml_count': len(unaligned_openxml_indices),
            'unaligned_openxml_indices': unaligned_openxml_indices,
            'max_openxml_idx': max(pdf_unit_assignment.values()) if pdf_unit_assignment else min_openxml_idx
        }

        return alignments, unaligned_pdf_indices, unaligned_openxml_indices, debug_info

    def _build_alignments_from_matching(self, openxml_to_pdf, pdf_units, openxml_units, match_debug):
        """
        Build alignment structure organized by OpenXML element.
        Groups table cells under parent element and keeps text/image parts separate.
        """
        elem_alignments = {}
        non_table_units = {}

        for openxml_idx, pdf_counts in openxml_to_pdf.items():
            if not pdf_counts:
                continue

            openxml_unit = openxml_units[openxml_idx]
            elem_id = openxml_unit['elem_id']
            unit_id = openxml_unit['unit_id']

            matched_pdf = []
            for pdf_idx, matched_count in pdf_counts.items():
                pdf_unit = pdf_units[pdf_idx]
                score = matched_count / len(pdf_unit['text_normalized']) if pdf_unit['text_normalized'] else 0

                debug_key = (openxml_idx, pdf_idx)
                debug_info = match_debug.get(debug_key, {})

                matched_pdf.append({
                    'pdf_unit_id': pdf_unit['unit_id'],
                    'item_idx': pdf_unit['item_idx'],
                    'item_type': pdf_unit['item_type'],
                    'text': pdf_unit['text'],
                    'bbox': pdf_unit['bbox'],
                    'matched_count': matched_count,
                    'score': round(score, 3),
                    'is_cell': pdf_unit['is_cell'],
                    'is_hline_table_unit': pdf_unit.get('is_hline_table_unit', False),
                    'row': pdf_unit.get('row'),
                    'col': pdf_unit.get('col'),
                    'debug': {
                        'matched_str': ''.join(debug_info.get('matched_chars', []))
                    } if debug_info else {}
                })

            matched_pdf.sort(key=lambda x: x['item_idx'])

            is_image_part = openxml_unit.get('is_image_part', False)

            if is_image_part:
                for mp_idx, mp in enumerate(matched_pdf):
                    bbox = mp.get('bbox')
                    individual_unit_id = f"{unit_id}_m{mp_idx}"
                    non_table_units[individual_unit_id] = {
                        'element_id': elem_id,
                        'element_sequence': openxml_unit['elem_seq'],
                        'element_type': openxml_unit['elem_type'],
                        'is_table': False,
                        'element_text': openxml_unit['text'],
                        'matched_pdf_units': [mp],
                        'merged_bbox': list(bbox) if bbox and len(bbox) >= 4 else None,
                        'cells': None,
                        'is_text_part': False,
                        'is_image_part': True,
                        'unit_id': individual_unit_id,
                        'image_index': openxml_unit.get('image_index')
                    }
                continue

            merged_bbox = self._merge_bboxes([mp.get('bbox') for mp in matched_pdf])

            if openxml_unit['is_cell']:
                if elem_id not in elem_alignments:
                    elem_alignments[elem_id] = {
                        'element_id': elem_id,
                        'element_sequence': openxml_unit['elem_seq'],
                        'element_type': openxml_unit['elem_type'],
                        'is_table': True,
                        'element_text': '',
                        'matched_pdf_units': [],
                        'merged_bbox': None,
                        'cells': []
                    }

                elem_alignments[elem_id]['cells'].append({
                    'row': openxml_unit['row'],
                    'col': openxml_unit['col'],
                    'text': openxml_unit['text'],
                    'matched_pdf_units': matched_pdf,
                    'merged_bbox': merged_bbox
                })

                if merged_bbox:
                    parent_bbox = elem_alignments[elem_id]['merged_bbox']
                    if parent_bbox is None:
                        elem_alignments[elem_id]['merged_bbox'] = list(merged_bbox)
                    else:
                        elem_alignments[elem_id]['merged_bbox'] = self._merge_bboxes([parent_bbox, merged_bbox])
            else:
                is_text_part = openxml_unit.get('is_text_part', False)
                is_image_part = openxml_unit.get('is_image_part', False)

                non_table_units[unit_id] = {
                    'element_id': elem_id,
                    'element_sequence': openxml_unit['elem_seq'],
                    'element_type': openxml_unit['elem_type'],
                    'is_table': False,
                    'element_text': openxml_unit['text'],
                    'matched_pdf_units': matched_pdf,
                    'merged_bbox': merged_bbox,
                    'cells': None,
                    'is_text_part': is_text_part,
                    'is_image_part': is_image_part,
                    'unit_id': unit_id
                }

        for alignment in elem_alignments.values():
            if alignment.get('cells'):
                alignment['cells'].sort(key=lambda c: (c['row'], c['col']))

        result = list(elem_alignments.values()) + list(non_table_units.values())
        result.sort(key=lambda x: x['element_sequence'] or 0)

        return result

    def _detect_suspicious_page_numbers(self, pdf_units, assignment, ox_to_pdf):
        susp = set()
        for idx, u in enumerate(pdf_units):
            if idx in assignment and self._is_standalone_number(u.get('text', '')):
                oidx = assignment[idx]
                prev_ok = (idx > 0 and (idx-1) in assignment and assignment[idx-1] == oidx)
                next_ok = (idx < len(pdf_units)-1 and (idx+1) in assignment and assignment[idx+1] == oidx)
                if not prev_ok and not next_ok:
                    susp.add(idx)
        return susp
 
    def _is_standalone_number(self, text):
        if not text:
            return False
        
        cleaned = text.strip()
        cleaned = cleaned.strip('-').strip()
        cleaned = cleaned.strip('.').strip()
        
        if cleaned.isdigit() and len(cleaned) <= 4:
            return True
        
        import re
        page_patterns = [
            r'^-?\s*\d{1,4}\s*-?$',        # "7", "-7-", "- 7 -"
            r'^page\s*\d{1,4}$',           # "Page 7"
            r'^hal\.?\s*\d{1,4}$',         # "Hal. 7", "Hal 7"
            r'^\d{1,4}\s*/\s*\d{1,4}$'     # "7/10"
        ]
        for pattern in page_patterns:
            if re.match(pattern, cleaned.lower()):
                return True
        
        return False
    
    def _is_likely_page_number(self, text, bbox):
        return self._is_standalone_number(text)

    def _absorb_unaligned_into_alignments(self, alignments, unaligned_indices, pdf_units):
        absorbed = set()
        for al in alignments:
            mbbox = al.get('merged_bbox')
            if not mbbox: continue
            
            to_absorb = []
            for idx in unaligned_indices:
                if idx in absorbed: continue
                u = pdf_units[idx]
                if self._is_bbox_inside(u.get('bbox'), mbbox):
                    to_absorb.append(u)
                    absorbed.add(idx)
            
            if to_absorb:
                if 'matched_pdf_units' not in al: al['matched_pdf_units'] = []
                for u in to_absorb:
                    al['matched_pdf_units'].append({
                        'pdf_unit_id': u['unit_id'], 'item_idx': u['item_idx'], 'item_type': u['item_type'],
                        'text': u['text'], 'bbox': u['bbox'], 'score': 0, 'absorbed': True
                    })
                al['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
        
        rem = [i for i in unaligned_indices if i not in absorbed]
        return alignments, rem

    def _is_bbox_inside(self, inner, outer, tol=5):
        if not inner or not outer: return False
        cx = (inner[0]+inner[2])/2
        cy = (inner[1]+inner[3])/2
        return (outer[0]-tol <= cx <= outer[2]+tol) and (outer[1]-tol <= cy <= outer[3]+tol)

    def _match_remaining_with_unaligned_openxml(self, alignments, un_pdf_idx, un_ox_idx, pdf_units, openxml_units):
        if not un_pdf_idx or not un_ox_idx: return alignments, un_pdf_idx, un_ox_idx
        
        sub_pdf = [pdf_units[i] for i in un_pdf_idx]
        sub_ox = [openxml_units[i] for i in un_ox_idx]
        
        late_align, l_un_pdf_local, l_un_ox_local, _ = self._perform_char_alignment(sub_pdf, sub_ox)
        
        ex_map = {a['element_id']: a for a in alignments}
        for la in late_align:
            eid = la['element_id']
            for u in la['matched_pdf_units']: u['late_matched'] = True
            
            if eid in ex_map:
                ex = ex_map[eid]
                ex['matched_pdf_units'].extend(la['matched_pdf_units'])
                ex['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
                if la.get('merged_bbox'):
                    if ex.get('merged_bbox'):
                        ex['merged_bbox'] = self._merge_bboxes([ex['merged_bbox'], la['merged_bbox']])
                    else:
                        ex['merged_bbox'] = la['merged_bbox']
            else:
                la['late_matched'] = True
                alignments.append(la)
        
        alignments.sort(key=lambda x: x.get('element_sequence') or 0)
        
        final_un_pdf = [un_pdf_idx[i] for i in l_un_pdf_local]
        final_un_ox = [un_ox_idx[i] for i in l_un_ox_local]
        return alignments, final_un_pdf, final_un_ox

    def _get_alignment_min_item_idx(self, alignment):
        indices = []
        if alignment.get('is_table') and alignment.get('cells'):
            for cell in alignment['cells']:
                for u in cell.get('matched_pdf_units', []):
                    idx = u.get('item_idx')
                    if idx is not None:
                        indices.append(idx)
        else:
            for u in alignment.get('matched_pdf_units', []):
                idx = u.get('item_idx')
                if idx is not None:
                    indices.append(idx)
        return min(indices) if indices else None

    def _get_alignment_sequence(self, alignment):
        seq = alignment.get('element_sequence')
        if seq is None:
            return 0
        try:
            return int(seq)
        except (TypeError, ValueError):
            return 0

    def _is_bbox_fully_contained(self, inner_bbox, outer_bbox, tolerance=2):
        if not inner_bbox or not outer_bbox or len(inner_bbox) < 4 or len(outer_bbox) < 4:
            return False
        if (abs(inner_bbox[0] - outer_bbox[0]) < tolerance and
            abs(inner_bbox[1] - outer_bbox[1]) < tolerance and
            abs(inner_bbox[2] - outer_bbox[2]) < tolerance and
            abs(inner_bbox[3] - outer_bbox[3]) < tolerance):
            return False
        return (inner_bbox[0] >= outer_bbox[0] - tolerance and
                inner_bbox[1] >= outer_bbox[1] - tolerance and
                inner_bbox[2] <= outer_bbox[2] + tolerance and
                inner_bbox[3] <= outer_bbox[3] + tolerance)

    def _is_punctuation_only(self, text):
        if not text:
            return False
        cleaned = text.strip()
        if not cleaned:
            return False
        punctuation_chars = set('.:,;!?-')
        return all(c in punctuation_chars for c in cleaned)

    def _cleanup_punctuation_alignments(self, alignments):
        if not alignments or len(alignments) < 2:
            return alignments

        punct_alignments = []
        container_candidates = []

        for i, align in enumerate(alignments):
            merged_bbox = align.get('merged_bbox')
            if not merged_bbox or len(merged_bbox) < 4:
                continue

            all_text = ' '.join(u.get('text', '') for u in align.get('matched_pdf_units', []))

            if self._is_punctuation_only(all_text):
                punct_alignments.append((i, align, merged_bbox))
            else:
                area = (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1])
                container_candidates.append((i, align, merged_bbox, area))

        if not punct_alignments or not container_candidates:
            return alignments

        punct_to_remove = set()

        for punct_idx, punct_align, punct_bbox in punct_alignments:
            best_container = None
            best_area = float('inf')

            for cont_idx, cont_align, cont_bbox, cont_area in container_candidates:
                if cont_idx == punct_idx:
                    continue
                if self._is_bbox_fully_contained(punct_bbox, cont_bbox):
                    if cont_area < best_area:
                        best_container = (cont_idx, cont_align)
                        best_area = cont_area

            if best_container:
                _, cont_align = best_container
                for pdf_unit in punct_align.get('matched_pdf_units', []):
                    pdf_unit['absorbed'] = True
                    pdf_unit['absorbed_from_punctuation'] = True
                    cont_align['matched_pdf_units'].append(pdf_unit)

                cont_align['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))

                cont_bbox = cont_align['merged_bbox']
                cont_bbox[0] = min(cont_bbox[0], punct_bbox[0])
                cont_bbox[1] = min(cont_bbox[1], punct_bbox[1])
                cont_bbox[2] = max(cont_bbox[2], punct_bbox[2])
                cont_bbox[3] = max(cont_bbox[3], punct_bbox[3])

                punct_to_remove.add(punct_idx)

        return [a for i, a in enumerate(alignments) if i not in punct_to_remove]

    def _recompute_alignment_bboxes(self, alignment):
        if alignment.get('is_table') and alignment.get('cells'):
            cell_bboxes = []
            for cell in alignment['cells']:
                cell_units = cell.get('matched_pdf_units', [])
                cell_bbox = self._merge_bboxes([u.get('bbox') for u in cell_units])
                cell['merged_bbox'] = cell_bbox
                if cell_bbox:
                    cell_bboxes.append(cell_bbox)
            alignment['merged_bbox'] = self._merge_bboxes(cell_bboxes)
        else:
            units = alignment.get('matched_pdf_units', [])
            alignment['merged_bbox'] = self._merge_bboxes([u.get('bbox') for u in units])

    def _resolve_shape_alignment_conflicts(self, alignments, pdf_units):
        if not alignments:
            return alignments, []

        pdf_unit_by_id = {u.get('unit_id'): u for u in pdf_units if u.get('unit_id')}
        alignment_positions = []
        alignments_by_sequence = sorted(alignments, key=self._get_alignment_sequence)
        for alignment in alignments:
            min_idx = self._get_alignment_min_item_idx(alignment)
            if min_idx is not None:
                alignment_positions.append((min_idx, alignment))
        alignment_positions.sort(key=lambda x: x[0])

        shape_refs = {}
        for alignment in alignments:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment['cells']:
                    for unit in cell.get('matched_pdf_units', []):
                        unit_id = unit.get('pdf_unit_id')
                        if not unit_id:
                            continue
                        pdf_unit = pdf_unit_by_id.get(unit_id)
                        if not pdf_unit or pdf_unit.get('item_type') != 'shape':
                            continue
                        shape_refs.setdefault(unit_id, []).append((alignment, cell, unit))
            else:
                for unit in alignment.get('matched_pdf_units', []):
                    unit_id = unit.get('pdf_unit_id')
                    if not unit_id:
                        continue
                    pdf_unit = pdf_unit_by_id.get(unit_id)
                    if not pdf_unit or pdf_unit.get('item_type') != 'shape':
                        continue
                    shape_refs.setdefault(unit_id, []).append((alignment, None, unit))

        debug = []
        touched = set()

        for unit_id, refs in shape_refs.items():
            if len(refs) < 2:
                continue

            pdf_unit = pdf_unit_by_id.get(unit_id)
            if not pdf_unit:
                continue

            shape_item_idx = pdf_unit.get('item_idx')
            target_seq = None
            if shape_item_idx is not None:
                for min_idx, alignment in alignment_positions:
                    if min_idx > shape_item_idx:
                        target_seq = self._get_alignment_sequence(alignment)
                        break

            candidates = {}
            for alignment, cell, unit in refs:
                candidates.setdefault(id(alignment), {'alignment': alignment, 'cells': [], 'units': []})
                if cell:
                    candidates[id(alignment)]['cells'].append(cell)
                candidates[id(alignment)]['units'].append(unit)

            candidate_list = list(candidates.values())
            candidate_alignments = [c['alignment'] for c in candidate_list]

            chosen_alignment = None
            if target_seq is not None:
                prior_candidates = [a for a in candidate_alignments if self._get_alignment_sequence(a) < target_seq]
                if prior_candidates:
                    chosen_alignment = max(prior_candidates, key=self._get_alignment_sequence)
                else:
                    chosen_alignment = min(candidate_alignments, key=lambda a: abs(self._get_alignment_sequence(a) - target_seq))
            else:
                chosen_alignment = max(candidate_alignments, key=self._get_alignment_sequence)

            removed_from = []
            if chosen_alignment:
                for candidate in candidate_list:
                    alignment = candidate['alignment']
                    if alignment is chosen_alignment:
                        continue

                    if alignment.get('is_table') and alignment.get('cells'):
                        for cell in alignment['cells']:
                            cell_units = cell.get('matched_pdf_units', [])
                            new_units = [u for u in cell_units if u.get('pdf_unit_id') != unit_id]
                            if len(new_units) != len(cell_units):
                                cell['matched_pdf_units'] = new_units
                                touched.add(id(alignment))
                    else:
                        units = alignment.get('matched_pdf_units', [])
                        new_units = [u for u in units if u.get('pdf_unit_id') != unit_id]
                        if len(new_units) != len(units):
                            alignment['matched_pdf_units'] = new_units
                            touched.add(id(alignment))

                    removed_from.append(self._get_alignment_sequence(alignment))

            if removed_from:
                debug.append({
                    'pdf_unit_id': unit_id,
                    'shape_item_idx': shape_item_idx,
                    'target_sequence': target_seq,
                    'kept_sequence': chosen_alignment.get('element_sequence') if chosen_alignment else None,
                    'removed_sequences': removed_from
                })

        if touched:
            for alignment in alignments:
                if id(alignment) in touched:
                    self._recompute_alignment_bboxes(alignment)

        return alignments, debug

    def _attach_shape_clusters_to_next_alignment(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments or not unaligned_pdf_indices:
            return alignments, unaligned_pdf_indices, []

        shape_indices = [
            idx for idx in unaligned_pdf_indices
            if pdf_units[idx].get('item_type') == 'shape'
        ]
        if not shape_indices:
            return alignments, unaligned_pdf_indices, []

        alignment_positions = []
        alignments_by_sequence = sorted(alignments, key=self._get_alignment_sequence)
        for alignment in alignments:
            min_idx = self._get_alignment_min_item_idx(alignment)
            if min_idx is not None:
                alignment_positions.append((min_idx, alignment))
        alignment_positions.sort(key=lambda x: x[0])

        if not alignment_positions:
            return alignments, unaligned_pdf_indices, []

        shape_indices.sort()
        clusters = []
        cluster = []
        prev_idx = None
        for idx in shape_indices:
            if prev_idx is None or idx == prev_idx + 1:
                cluster.append(idx)
            else:
                clusters.append(cluster)
                cluster = [idx]
            prev_idx = idx
        if cluster:
            clusters.append(cluster)

        remaining_unaligned = [i for i in unaligned_pdf_indices if i not in shape_indices]
        debug = []
        attached_count = 0

        for cluster in clusters:
            cluster_units = [pdf_units[i] for i in cluster]
            cluster_bbox = self._merge_bboxes([u.get('bbox') for u in cluster_units])
            cluster_text = ' '.join(u.get('text', '') for u in cluster_units).strip()
            cluster_item_idx_min = min(u.get('item_idx', 0) for u in cluster_units)
            cluster_item_idx_max = max(u.get('item_idx', 0) for u in cluster_units)

            next_alignment = None
            for min_idx, alignment in alignment_positions:
                if min_idx > cluster_item_idx_max:
                    next_alignment = alignment
                    break

            target_alignment = None
            if next_alignment:
                next_seq = self._get_alignment_sequence(next_alignment)
                prev_candidates = [a for a in alignments_by_sequence if self._get_alignment_sequence(a) < next_seq]
                if prev_candidates:
                    target_alignment = max(prev_candidates, key=self._get_alignment_sequence)
            else:
                if alignments_by_sequence:
                    target_alignment = alignments_by_sequence[-1]

            if not target_alignment:
                remaining_unaligned.extend(cluster)
                continue

            merged_unit = {
                'pdf_unit_id': f"pdf_shape_cluster_{cluster[0]}",
                'item_idx': cluster_item_idx_min,
                'item_type': 'shape',
                'text': cluster_text,
                'bbox': cluster_bbox,
                'matched_count': 0,
                'score': 0.0,
                'is_cell': False,
                'row': None,
                'col': None,
                'debug': {
                    'shape_cluster_size': len(cluster)
                }
            }

            if target_alignment.get('is_table'):
                shape_alignment = {
                    'element_id': target_alignment.get('element_id'),
                    'element_sequence': target_alignment.get('element_sequence'),
                    'element_type': target_alignment.get('element_type'),
                    'is_table': False,
                    'element_text': target_alignment.get('element_text', ''),
                    'matched_pdf_units': [merged_unit],
                    'merged_bbox': list(cluster_bbox) if cluster_bbox else None,
                    'cells': None,
                    'is_text_part': False,
                    'is_image_part': False,
                    'is_shape_part': True,
                    'unit_id': f"{target_alignment.get('element_id')}_shape_{cluster[0]}"
                }
                alignments.append(shape_alignment)
            else:
                target_alignment.setdefault('matched_pdf_units', []).append(merged_unit)
                target_alignment['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))

                if cluster_bbox:
                    if target_alignment.get('merged_bbox'):
                        mb = target_alignment['merged_bbox']
                        mb[0] = min(mb[0], cluster_bbox[0])
                        mb[1] = min(mb[1], cluster_bbox[1])
                        mb[2] = max(mb[2], cluster_bbox[2])
                        mb[3] = max(mb[3], cluster_bbox[3])
                    else:
                        target_alignment['merged_bbox'] = list(cluster_bbox)

            attached_count += 1
            debug.append({
                'cluster_size': len(cluster),
                'cluster_item_idx_min': cluster_item_idx_min,
                'cluster_item_idx_max': cluster_item_idx_max,
                'target_element_id': target_alignment.get('element_id')
            })

        if attached_count:
            alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments, remaining_unaligned, debug

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
                'has_shape': all_units[i].get('has_shape', False)
            }
            for i in indices
        ]

    def _normalize_text(self, text):
        if not text: return ''
        result = []
        for char in text:
            code = ord(char)
            normalized = None
            if 0x1D400 <= code <= 0x1D419: normalized = chr(ord('A') + (code - 0x1D400))
            elif 0x1D41A <= code <= 0x1D433: normalized = chr(ord('a') + (code - 0x1D41A))
            elif 0x1D434 <= code <= 0x1D44D: normalized = chr(ord('A') + (code - 0x1D434))
            elif 0x1D44E <= code <= 0x1D467:
                 if code == 0x1D455: normalized = 'h'
                 else: normalized = chr(ord('a') + (code - 0x1D44E))
            elif 0x1D468 <= code <= 0x1D49B: normalized = chr(ord('A') + (code - 0x1D468)) if code <= 0x1D481 else chr(ord('a') + (code - 0x1D482))
            elif 0x1D49C <= code <= 0x1D4CF: normalized = chr(ord('A') + (code - 0x1D49C)) if code <= 0x1D4B5 else chr(ord('a') + (code - 0x1D4B6))
            elif 0x1D4D0 <= code <= 0x1D503: normalized = chr(ord('A') + (code - 0x1D4D0)) if code <= 0x1D4E9 else chr(ord('a') + (code - 0x1D4EA))
            elif 0x1D504 <= code <= 0x1D537: normalized = chr(ord('A') + (code - 0x1D504)) if code <= 0x1D51C else chr(ord('a') + (code - 0x1D51E))
            elif 0x1D538 <= code <= 0x1D56B: normalized = chr(ord('A') + (code - 0x1D538)) if code <= 0x1D550 else chr(ord('a') + (code - 0x1D552))
            elif 0x1D56C <= code <= 0x1D59F: normalized = chr(ord('A') + (code - 0x1D56C)) if code <= 0x1D585 else chr(ord('a') + (code - 0x1D586))
            elif 0x1D5A0 <= code <= 0x1D5D3: normalized = chr(ord('A') + (code - 0x1D5A0)) if code <= 0x1D5B9 else chr(ord('a') + (code - 0x1D5BA))
            elif 0x1D5D4 <= code <= 0x1D607: normalized = chr(ord('A') + (code - 0x1D5D4)) if code <= 0x1D5ED else chr(ord('a') + (code - 0x1D5EE))
            elif 0x1D608 <= code <= 0x1D63B: normalized = chr(ord('A') + (code - 0x1D608)) if code <= 0x1D621 else chr(ord('a') + (code - 0x1D622))
            elif 0x1D63C <= code <= 0x1D66F: normalized = chr(ord('A') + (code - 0x1D63C)) if code <= 0x1D655 else chr(ord('a') + (code - 0x1D656))
            elif 0x1D670 <= code <= 0x1D6A3: normalized = chr(ord('A') + (code - 0x1D670)) if code <= 0x1D689 else chr(ord('a') + (code - 0x1D68A))
            elif 0x1D6A8 <= code <= 0x1D6E1: normalized = chr(0x0391 + (code - 0x1D6A8)) if code <= 0x1D6C0 else chr(0x03B1 + (code - 0x1D6C2))
            elif 0x1D6E2 <= code <= 0x1D71B: normalized = chr(0x0391 + (code - 0x1D6E2)) if code <= 0x1D6FA else chr(0x03B1 + (code - 0x1D6FC))
            elif 0x1D71C <= code <= 0x1D755: normalized = chr(0x0391 + (code - 0x1D71C)) if code <= 0x1D734 else chr(0x03B1 + (code - 0x1D736))
            elif 0x1D756 <= code <= 0x1D78F: normalized = chr(0x0391 + (code - 0x1D756)) if code <= 0x1D76E else chr(0x03B1 + (code - 0x1D770))
            elif 0x1D790 <= code <= 0x1D7C9: normalized = chr(0x0391 + (code - 0x1D790)) if code <= 0x1D7A8 else chr(0x03B1 + (code - 0x1D7AA))
            elif code in [0x1D715, 0x1D6DB, 0x1D74F, 0x1D789, 0x1D7C3, 0x2202]: normalized = '∂'
            elif 0x1D7CE <= code <= 0x1D7FF: normalized = chr(ord('0') + (code - 0x1D7CE)) if code <= 0x1D7D7 else chr(ord('0') + (code - 0x1D7D8)) if code <= 0x1D7E1 else chr(ord('0') + (code - 0x1D7E2)) if code <= 0x1D7EB else chr(ord('0') + (code - 0x1D7EC)) if code <= 0x1D7F5 else chr(ord('0') + (code - 0x1D7F6))
            elif code == 0x2070: normalized = '0'
            elif code == 0x00B9: normalized = '1'
            elif code == 0x00B2: normalized = '2'
            elif code == 0x00B3: normalized = '3'
            elif 0x2074 <= code <= 0x2079: normalized = chr(ord('0') + (code - 0x2070))
            # Superscript letters
            elif code == 0x1D43: normalized = 'a'   # ᵃ
            elif code == 0x1D47: normalized = 'b'   # ᵇ
            elif code == 0x1D9C: normalized = 'c'   # ᶜ
            elif code == 0x1D48: normalized = 'd'   # ᵈ
            elif code == 0x1D49: normalized = 'e'   # ᵉ
            elif code == 0x1DA0: normalized = 'f'   # ᶠ
            elif code == 0x1D4D: normalized = 'g'   # ᵍ
            elif code == 0x02B0: normalized = 'h'   # ʰ
            elif code == 0x2071: normalized = 'i'   # ⁱ
            elif code == 0x02B2: normalized = 'j'   # ʲ
            elif code == 0x1D4F: normalized = 'k'   # ᵏ
            elif code == 0x02E1: normalized = 'l'   # ˡ
            elif code == 0x1D50: normalized = 'm'   # ᵐ
            elif code == 0x207F: normalized = 'n'   # ⁿ
            elif code == 0x1D52: normalized = 'o'   # ᵒ
            elif code == 0x1D56: normalized = 'p'   # ᵖ
            elif code == 0x02B3: normalized = 'r'   # ʳ
            elif code == 0x02E2: normalized = 's'   # ˢ
            elif code == 0x1D57: normalized = 't'   # ᵗ
            elif code == 0x1D58: normalized = 'u'   # ᵘ
            elif code == 0x1D5B: normalized = 'v'   # ᵛ
            elif code == 0x02B7: normalized = 'w'   # ʷ
            elif code == 0x02E3: normalized = 'x'   # ˣ
            elif code == 0x02B8: normalized = 'y'   # ʸ
            elif code == 0x1DBB: normalized = 'z'   # ᶻ
            # Subscript digits
            elif 0x2080 <= code <= 0x2089: normalized = chr(ord('0') + (code - 0x2080))
            # Subscript letters
            elif code == 0x2090: normalized = 'a'   # ₐ
            elif code == 0x2091: normalized = 'e'   # ₑ
            elif code == 0x2095: normalized = 'h'   # ₕ
            elif code == 0x1D62: normalized = 'i'   # ᵢ
            elif code == 0x2C7C: normalized = 'j'   # ⱼ
            elif code == 0x2096: normalized = 'k'   # ₖ
            elif code == 0x2097: normalized = 'l'   # ₗ
            elif code == 0x2098: normalized = 'm'   # ₘ
            elif code == 0x2099: normalized = 'n'   # ₙ
            elif code == 0x2092: normalized = 'o'   # ₒ
            elif code == 0x209A: normalized = 'p'   # ₚ
            elif code == 0x1D63: normalized = 'r'   # ᵣ
            elif code == 0x209B: normalized = 's'   # ₛ
            elif code == 0x209C: normalized = 't'   # ₜ
            elif code == 0x1D64: normalized = 'u'   # ᵤ
            elif code == 0x1D65: normalized = 'v'   # ᵥ
            elif code == 0x2093: normalized = 'x'   # ₓ
            # Math operators
            elif char in '−–—‐‑‒―': normalized = '-'
            elif char in '×∙·•⋅': normalized = '*'
            elif char in '÷∕': normalized = '/'
            elif char == '±': normalized = '+-'
            elif char == '∓': normalized = '-+'
            elif char in '＝⁼₌': normalized = '='
            elif char in '＜‹〈⟨': normalized = '<'
            elif char in '＞›〉⟩': normalized = '>'
            elif char in '≤≦⩽': normalized = '<='
            elif char in '≥≧⩾': normalized = '>='
            elif char in '→←↑↓↔↕⇒⇐⇑⇓⇔': normalized = ''
            elif char in '′': normalized = "'"
            elif char in '″': normalized = "''"
            elif char == '½': normalized = '1/2'
            elif char == '⅓': normalized = '1/3'
            elif char == '¼': normalized = '1/4'
            elif char == '⅔': normalized = '2/3'
            elif char == '¾': normalized = '3/4'
            elif 0xFF01 <= code <= 0xFF5E: normalized = chr(code - 0xFF00 + 0x20)
            
            result.append(normalized if normalized is not None else char)
        return ''.join(''.join(result).lower().split())

    def _extract_text_from_json_tree(self, json_tree):
        """Recursively extract text from dokumen_elemen_json_tree.
        
        Images are converted to count-based placeholders [IMG:1], [IMG:2], etc.
        Matches the legacy extract_text_from_json_tree from dokumen_elemen_routes.py
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
                # Recurse through all values EXCEPT certain keys
                for key, value in node.items():
                    if key not in ['text', 't', 'content', 'value', 'type', 'rId']:
                        collect_items(value)
            elif isinstance(node, list):
                for item in node:
                    collect_items(item)
        
        collect_items(json_tree)
        
        # Second pass: generate count-based placeholders for images
        result_parts = []
        image_counter = 0
        
        for item in items:
            if item['type'] == 'text':
                result_parts.append(item['value'])
            elif item['type'] == 'image':
                image_counter += 1
                result_parts.append(f'[IMG:{image_counter}]')
        
        return ' '.join(result_parts).strip()
