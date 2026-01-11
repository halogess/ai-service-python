
import difflib
import re
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
        """
        db = SessionLocal()
        try:
            # 1. Get OpenXML elements for this document (all body parts)
            # We fetch all because we don't know exactly which ones are on this page yet
            # But for optimization, we start matching from min_openxml_idx
            elements = self._get_openxml_elements(db, doc_id)
            
            # 2. Get Section Data for margin logic
            sections = self._get_doc_sections(db, doc_id)
            current_section = self._get_section_for_page(sections, page_num)
            
            # 3. Filter Header/Footer Items (Frontend display logic moved here)
            # In the original, the frontend filtered them for display but the alignment 
            # ran on "flattened items". Here we align everything but mark them.
            
            # 4. Flatten Extraction Items (PDF Units)
            pdf_units = self._flatten_extraction_items(extraction_items)
            
            # 5. Filter Header/Footer units from alignment candidates
            # We align ONLY body content first. Header/Footer are handled separately or ignored.
            body_units, header_footer_units = self._filter_header_footer_items(pdf_units, current_section, page_height)
            
            # 6. Perform Core Alignment
            alignment_result = self._perform_alignment(
                body_units, 
                elements, 
                min_openxml_idx
            )
            
            return {
                'success': True,
                'alignments': alignment_result['alignments'],
                'unaligned_pdf_units': alignment_result['unaligned_pdf_units'],
                'header_footer_units': header_footer_units,
                'max_openxml_idx': alignment_result['max_openxml_idx'],
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
        return DokumenSection.query(DokumenSection).filter_by(dokumen_id=doc_id).order_by(DokumenSection.dsec_index).all() if hasattr(DokumenSection, 'query') else db_session.query(DokumenSection).filter_by(dokumen_id=doc_id).order_by(DokumenSection.dsec_index).all()

    def _get_section_for_page(self, sections, page_num):
        # Simplistic logic: assuming 1 section per document or linear flow
        # Ideally we map page ranges to sections, but for now take the first or specific logic
        # if available. The legacy code passed all sections or handled it broadly.
        # We'll take the first section that seems to cover this page or just the first one.
        if not sections:
            return None
        return sections[0] # Fallback

    def _flatten_extraction_items(self, extraction_items):
        """Flatten extraction items to smallest alignment units."""
        # Ported from flatten_extraction_items in merging_alignment.py
        collected_items = []
        
        for item_idx, item in enumerate(extraction_items):
            item_type = item.get('type', '')
            item_data = item.get('data', {})
            item_bbox = item.get('bbox')
            
            if item_type == 'group':
                text = item_data.get('text', '')
                if text.strip():
                    collected_items.append({
                        'item_idx': item_idx, 'item_type': item_type, 'text': text,
                        'bbox': item_bbox, 'is_cell': False, 'row': None, 'col': None, 'is_image': False
                    })
            elif item_type == 'table':
                # flattened cells
                rows = item_data.get('rows', [])
                for row_idx, row in enumerate(rows):
                    cells = row.get('cells', [])
                    for col_idx, cell in enumerate(cells):
                         # extract text from cell content
                        cell_text = self._extract_cell_content_text(cell)
                        if cell_text.strip():
                            collected_items.append({
                                'item_idx': item_idx, 'item_type': item_type, 'text': cell_text,
                                'bbox': cell.get('bbox'), 'is_cell': True, 'row': row_idx, 'col': col_idx,
                                'table_bbox': item_bbox, 'is_image': False
                            })
            elif item_type == 'shape':
                text = item_data.get('text', '')
                if text.strip():
                     collected_items.append({
                        'item_idx': item_idx, 'item_type': item_type, 'text': text,
                        'bbox': item_bbox, 'is_cell': False, 'row': None, 'col': None, 'is_image': False
                    })
            elif item_type == 'image':
                 collected_items.append({
                    'item_idx': item_idx, 'item_type': item_type, 'text': None,
                    'bbox': item_bbox, 'is_cell': False, 'row': None, 'col': None, 'is_image': True
                })
        
        # Merge consecutive shapes
        collected_items = self._merge_consecutive_shape_items(collected_items)

        # Create PDF units
        pdf_units = []
        unit_counter = 0
        image_counter = 0
        
        for item in collected_items:
            if item['is_image']:
                image_counter += 1
                img_placeholder = '[IMG]'
                pdf_units.append({
                    'unit_id': f'pdf_{unit_counter}',
                    'item_idx': item['item_idx'],
                    'item_type': item['item_type'],
                    'text': img_placeholder,
                    'text_normalized': img_placeholder.lower(),
                    'bbox': item['bbox'],
                    'is_cell': False
                })
            else:
                pdf_units.append({
                   'unit_id': f'pdf_{unit_counter}',
                    'item_idx': item['item_idx'],
                    'item_type': item['item_type'],
                    'text': item['text'],
                    'text_normalized': self._normalize_text(item['text']),
                    'bbox': item['bbox'],
                    'is_cell': item['is_cell'],
                    'row': item['row'],
                    'col': item['col'],
                    'is_page_number': False # Simplified for now
                })
            unit_counter += 1
            
        return pdf_units

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
                continue
                
            merged_text = ' '.join(i.get('text', '') for i in cluster).strip()
            # Merge logic for bbox omitted for brevity, assuming simple union
            merged.append({
                'item_idx': cluster[0]['item_idx'],
                'item_type': 'shape',
                'text': merged_text,
                'bbox': cluster[0]['bbox'], # Placeholder bbox logic
                'is_cell': False,
                'is_image': False
            })
        return merged

    def _extract_cell_content_text(self, cell):
        # Simplified extraction
        texts = []
        content = cell.get('content', [])
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get('type') == 'text':
                    texts.append(c.get('text', ''))
        return ' '.join(texts)

    def _normalize_text(self, text):
        """Normalize text for matching: 
        - Convert Unicode mathematical characters to ASCII
        - Normalize Greek letters
        - Handle subscripts/superscripts
        - Lowercase
        - Remove whitespace
        """
        if not text:
            return ''
        
        result = []
        for char in text:
            code = ord(char)
            normalized = None
            
            # LATIN LETTERS (A-Z, a-z) - Mathematical variants
            if 0x1D400 <= code <= 0x1D419: normalized = chr(ord('A') + (code - 0x1D400)) # Bold A-Z
            elif 0x1D41A <= code <= 0x1D433: normalized = chr(ord('a') + (code - 0x1D41A)) # Bold a-z
            elif 0x1D434 <= code <= 0x1D467: # Italic
                if code == 0x1D455: normalized = 'h'
                elif code < 0x1D455: normalized = chr(ord('a') + (code - 0x1D44E))
                else: normalized = chr(ord('a') + (code - 0x1D44E)) # Logic adjusted to match original
            elif 0x1D468 <= code <= 0x1D49B: normalized = chr(ord('A') + (code - 0x1D468)) if code <= 0x1D481 else chr(ord('a') + (code - 0x1D482)) # Bold Italic
            elif 0x1D49C <= code <= 0x1D4CF: normalized = chr(ord('A') + (code - 0x1D49C)) if code <= 0x1D4B5 else chr(ord('a') + (code - 0x1D4B6)) # Script
            elif 0x1D4D0 <= code <= 0x1D503: normalized = chr(ord('A') + (code - 0x1D4D0)) if code <= 0x1D4E9 else chr(ord('a') + (code - 0x1D4EA)) # Bold Script
            elif 0x1D504 <= code <= 0x1D537: normalized = chr(ord('A') + (code - 0x1D504)) if code <= 0x1D51C else chr(ord('a') + (code - 0x1D51E)) # Fraktur
            elif 0x1D538 <= code <= 0x1D56B: normalized = chr(ord('A') + (code - 0x1D538)) if code <= 0x1D550 else chr(ord('a') + (code - 0x1D552)) # Double-Struck
            elif 0x1D56C <= code <= 0x1D59F: normalized = chr(ord('A') + (code - 0x1D56C)) if code <= 0x1D585 else chr(ord('a') + (code - 0x1D586)) # Bold Fraktur
            elif 0x1D5A0 <= code <= 0x1D5D3: normalized = chr(ord('A') + (code - 0x1D5A0)) if code <= 0x1D5B9 else chr(ord('a') + (code - 0x1D5BA)) # Sans
            elif 0x1D5D4 <= code <= 0x1D607: normalized = chr(ord('A') + (code - 0x1D5D4)) if code <= 0x1D5ED else chr(ord('a') + (code - 0x1D5EE)) # Sans Bold
            elif 0x1D608 <= code <= 0x1D63B: normalized = chr(ord('A') + (code - 0x1D608)) if code <= 0x1D621 else chr(ord('a') + (code - 0x1D622)) # Sans Italic
            elif 0x1D63C <= code <= 0x1D66F: normalized = chr(ord('A') + (code - 0x1D63C)) if code <= 0x1D655 else chr(ord('a') + (code - 0x1D656)) # Sans Bold Italic
            elif 0x1D670 <= code <= 0x1D6A3: normalized = chr(ord('A') + (code - 0x1D670)) if code <= 0x1D689 else chr(ord('a') + (code - 0x1D68A)) # Monospace

            # GREEK LETTERS
            elif 0x1D6A8 <= code <= 0x1D6E1: normalized = chr(0x0391 + (code - 0x1D6A8)) if code <= 0x1D6C0 else chr(0x03B1 + (code - 0x1D6C2)) # Bold
            elif 0x1D6E2 <= code <= 0x1D71B: normalized = chr(0x0391 + (code - 0x1D6E2)) if code <= 0x1D6FA else chr(0x03B1 + (code - 0x1D6FC)) # Italic
            elif 0x1D71C <= code <= 0x1D755: normalized = chr(0x0391 + (code - 0x1D71C)) if code <= 0x1D734 else chr(0x03B1 + (code - 0x1D736)) # Bold Italic
            elif 0x1D756 <= code <= 0x1D78F: normalized = chr(0x0391 + (code - 0x1D756)) if code <= 0x1D76E else chr(0x03B1 + (code - 0x1D770)) # Sans Bold
            elif 0x1D790 <= code <= 0x1D7C9: normalized = chr(0x0391 + (code - 0x1D790)) if code <= 0x1D7A8 else chr(0x03B1 + (code - 0x1D7AA)) # Sans Bold Italic

            # Common Math Symbols
            elif code in [0x1D715, 0x1D6DB, 0x1D74F, 0x1D789, 0x1D7C3, 0x2202]: normalized = '∂'
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
            
            # DIGITS (Bold, Double-Struck, Sans, Monospace, etc)
            elif 0x1D7CE <= code <= 0x1D7FF:
                base = 0x1D7CE
                if code >= 0x1D7F6: base = 0x1D7F6 # Monospace
                elif code >= 0x1D7EC: base = 0x1D7EC # Sans Bold
                elif code >= 0x1D7E2: base = 0x1D7E2 # Sans
                elif code >= 0x1D7D8: base = 0x1D7D8 # Double Struck
                normalized = chr(ord('0') + (code - base))

            # SUPERSCRIPTS / SUBSCRIPTS
            elif code == 0x2070: normalized = '0'
            elif code == 0x00B9: normalized = '1'
            elif code == 0x00B2: normalized = '2'
            elif code == 0x00B3: normalized = '3'
            elif 0x2074 <= code <= 0x2079: normalized = chr(ord('0') + (code - 0x2070))
            elif 0x2080 <= code <= 0x2089: normalized = chr(ord('0') + (code - 0x2080)) # Subscript digits
            
            # Fullwidth ASCII
            elif 0xFF01 <= code <= 0xFF5E: normalized = chr(code - 0xFF00 + 0x20)

            if normalized is None:
                normalized = char
            
            result.append(normalized)
        
        return ''.join(''.join(result).lower().split())

    def _filter_header_footer_items(self, pdf_units, section, page_height):
        # Placeholder: assume no header/footer for now or implement bounding box check
        # For this migration, we'll return all as body for simplicity unless strictness needed
        return pdf_units, [] 

    def _perform_alignment(self, pdf_units, elements, min_openxml_idx):
        # Use difflib to align normalized texts
        
        # Prepare PDF text stream
        pdf_concat = ""
        pdf_map = [] # char index -> unit index
        for i, u in enumerate(pdf_units):
            txt = u['text_normalized']
            for _ in txt:
                pdf_map.append(i)
            pdf_concat += txt
            
        # Prepare Elements text stream
        # Optimization: start checking from min_openxml_idx
        relevant_elements = elements[min_openxml_idx:]
        elem_concat = ""
        elem_map = [] # char index -> (relative_elem_index + min_openxml_idx)
        
        # Also need to extract text from element json tree
        extracted_elem_texts = []
        
        for i, elem in enumerate(relevant_elements):
            # We need a helper to extract text recursively from elem.delemen_json_tree
            # For now, let's assume a simplified extraction
            txt = self._extract_text_from_json_tree(elem.delemen_json_tree)
            norm_txt = self._normalize_text(txt)
            extracted_elem_texts.append({'id': elem.delemen_id, 'text': txt, 'norm': norm_txt})
            
            for _ in norm_txt:
                elem_map.append(i + min_openxml_idx)
            elem_concat += norm_txt
            
        # Matching
        sm = difflib.SequenceMatcher(None, pdf_concat, elem_concat, autojunk=False)
        blocks = sm.get_matching_blocks()
        
        # Assign matches
        # pdf_unit -> list of (elem_index, matched_chars)
        assignments = {}
        
        for block in blocks:
            for offset in range(block.size):
                pdf_char_idx = block.a + offset
                elem_char_idx = block.b + offset
                
                if pdf_char_idx < len(pdf_map) and elem_char_idx < len(elem_map):
                    u_idx = pdf_map[pdf_char_idx]
                    e_idx = elem_map[elem_char_idx]
                    
                    if u_idx not in assignments:
                        assignments[u_idx] = {}
                    if e_idx not in assignments[u_idx]:
                        assignments[u_idx][e_idx] = 0
                    assignments[u_idx][e_idx] += 1
                    
        # Finalize alignments
        final_alignments = []
        max_idx_used = min_openxml_idx
        
        for u_idx, matches in assignments.items():
            # Pick best element match
            best_e_idx = max(matches.items(), key=lambda x: x[1])[0]
            max_idx_used = max(max_idx_used, best_e_idx)
            
            unit = pdf_units[u_idx]
            final_alignments.append({
                'unit_id': unit['unit_id'],
                'text': unit['text'],
                'bbox': unit['bbox'],
                'element_id': elements[best_e_idx].delemen_id,
                'score': matches[best_e_idx] / len(unit['text_normalized']) if unit['text_normalized'] else 0
            })
            
        return {
            'alignments': final_alignments,
            'unaligned_pdf_units': [], # Todo: calculate unaligned
            'max_openxml_idx': max_idx_used,
            'debug_info': {}
        }

    def _extract_text_from_json_tree(self, json_tree):
        """Recursively extract text from dokumen_elemen_json_tree.
        
        Images are converted to context-based placeholders [IMG:1] based on order.
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
