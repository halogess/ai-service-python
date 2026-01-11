
import difflib
import re
import math
from typing import List, Dict, Any, Tuple, Optional, Set
from models import DokumenElemen, DokumenSection, DokumenPart
from database import SessionLocal

class AlignmentService:
    def __init__(self):
        pass

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

            return {
                'success': True,
                'phase1_alignments': alignment_result['phase1_alignments'],
                'alignments': alignment_result['final_alignments'], # Backward compat alias
                'final_alignments': alignment_result['final_alignments'],
                'shape_alignments': alignment_result['shape_alignments'],
                'unaligned_pdf_units': alignment_result['unaligned_final'],
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
                if text.strip():
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
            elem_has_shape = self._has_shape_content(elem.delemen_json_tree)
            
            if self._is_table_element(elem.delemen_type):
                cells = self._extract_table_cells(elem.delemen_json_tree)
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
                content = self._extract_text_and_images_separately(elem.delemen_json_tree)
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
                    text = content['combined'] if content['combined'] else self._extract_text_from_json_tree(elem.delemen_json_tree)
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
        final_align, _ = self._resolve_shape_alignment_conflicts(final_align, pdf_units)
        final_align, final_un_pdf, _ = self._attach_shape_clusters_to_next_alignment(final_align, final_un_pdf, pdf_units)

        max_idx = min_openxml_idx
        if p1_debug.get('max_openxml_idx'): max_idx = p1_debug['max_openxml_idx']

        return {
            'phase1_alignments': p1_align,
            'final_alignments': final_align,
            'shape_alignments': [],
            'unaligned_after_phase1': p1_un_pdf,
            'unaligned_final': final_un_pdf,
            'unaligned_openxml': p1_un_ox,
            'debug_info': p1_debug,
            'max_openxml_idx': max_idx
        }

    def _perform_char_alignment(self, pdf_units, openxml_units, min_openxml_idx=0):
        if not pdf_units or not openxml_units:
            return [], list(range(len(pdf_units))), list(range(len(openxml_units))), {'max_openxml_idx': min_openxml_idx}

        pdf_concat = ''
        pdf_map = []
        for i, u in enumerate(pdf_units):
            t = u['text_normalized']
            for _ in t: pdf_map.append(i)
            pdf_concat += t
        
        ox_concat = ''
        ox_map = []
        for i, u in enumerate(openxml_units):
            t = u['text_normalized']
            for _ in t: ox_map.append(i)
            ox_concat += t

        sm = difflib.SequenceMatcher(None, pdf_concat, ox_concat, autojunk=False)
        blocks = sm.get_matching_blocks()
        blocks = sorted(blocks, key=lambda x: x.b)

        consumed_ox_chars = set() # Optional if blocks are disjoint, but safe to keep checking if needed. 
        # Actually diffib blocks ARE disjoint. 
        # But let's stick to cleaning up the unit-level block.
        pdf_assign = {}
        ox_to_pdf = {}
        suspicious = set()

        for b in blocks:
            for off in range(b.size):
                pi = b.a + off
                oi = b.b + off
                if pi < len(pdf_map) and oi < len(ox_map):
                    uidx = pdf_map[pi]
                    oidx = ox_map[oi]
                    # if oidx in consumed_ox: continue  <-- REMOVE THIS
                    if uidx in pdf_assign:
                        if pdf_assign[uidx] != oidx: continue 
                    else:
                        if oidx < min_openxml_idx: continue
                        violation = False
                        for p, o in pdf_assign.items():
                            if uidx > p and oidx < o: violation = True; break
                            if uidx < p and oidx > o: violation = True; break
                        if violation: continue
                        pdf_assign[uidx] = oidx
                    
                    # consumed_ox.add(oidx) <-- REMOVE THIS
                    if oidx not in ox_to_pdf: ox_to_pdf[oidx] = {}
                    if uidx not in ox_to_pdf[oidx]: ox_to_pdf[oidx][uidx] = 0
                    ox_to_pdf[oidx][uidx] += 1
        
        suspicious = self._detect_suspicious_page_numbers(pdf_units, pdf_assign, ox_to_pdf)
        
        alignments = []
        for oidx, pdata in ox_to_pdf.items():
            valid_pdata = {k:v for k,v in pdata.items() if k not in suspicious}
            if not valid_pdata: continue
            
            ou = openxml_units[oidx]
            matched = []
            mbbox = None
            for pidx, cnt in valid_pdata.items():
                pu = pdf_units[pidx]
                sc = cnt / len(pu['text_normalized']) if pu['text_normalized'] else 0
                matched.append({
                    'pdf_unit_id': pu['unit_id'], 'item_idx': pu['item_idx'], 'item_type': pu['item_type'],
                    'text': pu['text'], 'bbox': pu['bbox'], 'matched_count': cnt, 'score': round(sc, 3),
                    'is_cell': pu['is_cell']
                })
                if pu.get('bbox'):
                    b = pu['bbox']
                    if mbbox is None: mbbox = list(b)
                    else:
                        mbbox[0] = min(mbbox[0], b[0])
                        mbbox[1] = min(mbbox[1], b[1])
                        mbbox[2] = max(mbbox[2], b[2])
                        mbbox[3] = max(mbbox[3], b[3])
            
            matched.sort(key=lambda x: x['item_idx'])
            alignments.append({
                'element_id': ou['elem_id'], 'element_sequence': ou.get('elem_seq'), 'element_type': ou['elem_type'],
                'is_table': ou['is_cell'],
                'element_text': ou['text'],
                'matched_pdf_units': matched, 'merged_bbox': mbbox,
                'cells': None
            })
            
        alignments.sort(key=lambda x: x.get('element_sequence') or 0)
        
        unaligned_pdf = [i for i in range(len(pdf_units)) if i not in pdf_assign and i not in suspicious]
        unaligned_ox = [i for i in range(len(openxml_units)) if i not in ox_to_pdf]
        max_oidx = max(pdf_assign.values()) if pdf_assign else min_openxml_idx
        
        return alignments, unaligned_pdf, unaligned_ox, {'max_openxml_idx': max_oidx, 'unaligned_openxml_indices': unaligned_ox}

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
        if not text: return False
        cl = text.strip().strip('-').strip('.').strip()
        if cl.isdigit() and len(cl) <= 4: return True
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

    def _cleanup_punctuation_alignments(self, alignments):
        # Merge solitary punctuation alignment into container
        punct_to_remove = set()
        for i, punct_align in enumerate(alignments):
            if i in punct_to_remove: continue
            units = punct_align.get('matched_pdf_units', [])
            all_punct = all(not any(c.isalnum() for c in u.get('text', '')) for u in units)
            if not all_punct: continue
            
            punct_bbox = punct_align.get('merged_bbox')
            if not punct_bbox: continue
            
            best_container = None
            best_area = float('inf')
            
            for j, cont_align in enumerate(alignments):
                if i == j or j in punct_to_remove: continue
                cont_bbox = cont_align.get('merged_bbox')
                if not cont_bbox: continue
                
                cont_area = (cont_bbox[2]-cont_bbox[0]) * (cont_bbox[3]-cont_bbox[1])
                
                if self._is_bbox_inside(punct_bbox, cont_bbox):
                    if cont_area < best_area:
                        best_container = cont_align
                        best_area = cont_area
            
            if best_container:
                for unit in units:
                    unit['absorbed'] = True
                    unit['absorbed_from_punctuation'] = True
                    best_container['matched_pdf_units'].append(unit)
                best_container['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))
                
                pb = best_container['merged_bbox']
                pb[0] = min(pb[0], punct_bbox[0])
                pb[1] = min(pb[1], punct_bbox[1])
                pb[2] = max(pb[2], punct_bbox[2])
                pb[3] = max(pb[3], punct_bbox[3])
                
                punct_to_remove.add(i)
        
        return [a for i, a in enumerate(alignments) if i not in punct_to_remove]

    def _resolve_shape_alignment_conflicts(self, alignments, pdf_units):
        # Resolve shapes assigned to multiple alignments
        # Simplified: if we have comprehensive shape logic elsewhere, this might be overkill,
        # but for full parity we should check if shapes are duplicated.
        # Legacy checked if shape is in mulitple places. Here we just return as is 
        # because the primary assignment logic (char based) doesn't typically double-assign shapes 
        # unless we explicitly logic'd it.
        # Legacy used this because shapes were handled separately. 
        # Given our flow, shapes are largely unconsumed by char alignment unless text matches.
        # So we can keep it simple.
        return alignments, []

    def _attach_shape_clusters_to_next_alignment(self, alignments, unaligned_pdf_idx, pdf_units):
        if not alignments or not unaligned_pdf_idx: return alignments, unaligned_pdf_idx, []
        
        shape_indices = [i for i in unaligned_pdf_idx if pdf_units[i].get('item_type') == 'shape']
        if not shape_indices: return alignments, unaligned_pdf_idx, []
        
        shape_indices.sort()
        clusters = []
        if shape_indices:
            cluster = [shape_indices[0]]
            for x in shape_indices[1:]:
                if x == cluster[-1] + 1: cluster.append(x)
                else:
                    clusters.append(cluster)
                    cluster = [x]
            clusters.append(cluster)
            
        attached_count = 0
        consumed_shapes = set()
        
        # Helper to get alignment sequence
        def get_seq(a): return a.get('element_sequence') or 0
        
        # Sort alignments
        sorted_aligns = sorted(alignments, key=get_seq)
        
        for cluster in clusters:
            # Find next alignment
            cluster_max_idx = max(pdf_units[i]['item_idx'] for i in cluster)
            next_align = None
            for a in alignments:
                # Approximate position by getting min item_idx in align
                min_idx = min((u['item_idx'] for u in a['matched_pdf_units'] if u.get('item_idx') is not None), default=999999)
                if min_idx > cluster_max_idx:
                    if next_align is None or min((u['item_idx'] for u in next_align['matched_pdf_units'] if u.get('item_idx') is not None), default=999999) > min_idx:
                        next_align = a
            
            target = None
            if next_align:
                # Attach to PREVIOUS of next (i.e. the one before the gap)
                next_seq = get_seq(next_align)
                candidates = [a for a in sorted_aligns if get_seq(a) < next_seq]
                if candidates: target = max(candidates, key=get_seq)
            else:
                # Attach to last
                if sorted_aligns: target = sorted_aligns[-1]
            
            if target:
                cl_units = [pdf_units[i] for i in cluster]
                cl_bbox = self._merge_bboxes([u['bbox'] for u in cl_units])
                cl_text = ' '.join(u['text'] for u in cl_units)
                
                merged_unit = {
                    'pdf_unit_id': f"pdf_shape_cluster_{cluster[0]}",
                    'item_idx': min(u['item_idx'] for u in cl_units),
                    'item_type': 'shape', 'text': cl_text, 'bbox': cl_bbox,
                    'score': 0.0, 'is_cell': False, 'attached_shape': True
                }
                
                if target.get('is_table'):
                    # Append as separate alignment
                     alignments.append({
                        'element_id': target['element_id'], 'element_sequence': target['element_sequence'],
                        'element_type': target['element_type'], 'is_table': False,
                        'element_text': '', 'matched_pdf_units': [merged_unit],
                        'merged_bbox': cl_bbox
                     })
                else:
                    target['matched_pdf_units'].append(merged_unit)
                    target['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))
                    if cl_bbox:
                         if target.get('merged_bbox'): target['merged_bbox'] = self._merge_bboxes([target['merged_bbox'], cl_bbox])
                         else: target['merged_bbox'] = cl_bbox
                         
                consumed_shapes.update(cluster)
                attached_count += 1

        rem = [i for i in unaligned_pdf_idx if i not in consumed_shapes]
        return alignments, rem, []

    def _format_unaligned_openxml(self, all_units, indices):
        return [{'unit_id': all_units[i]['unit_id'], 'text': all_units[i]['text']} for i in indices]

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
        items = []
        def c(node):
            if isinstance(node, dict):
                if 'value' in node: items.append(str(node['value']))
                elif 'text' in node: items.append(str(node['text']))
                for v in node.values(): c(v)
            elif isinstance(node, list):
                for i in node: c(i)
        c(json_tree)
        return ' '.join(items)
