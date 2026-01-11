from flask import Blueprint, render_template, jsonify, request
from models import db, TestingDokumen, DokumenElemen, DokumenSection

dokumen_elemen_bp = Blueprint('dokumen_elemen', __name__)

@dokumen_elemen_bp.route('/dokumen-elemen')
def dokumen_elemen_list():
    """List all documents for dokumen elemen viewing"""
    documents = TestingDokumen.query.order_by(TestingDokumen.testing_dokumen_id).all()
    return render_template('dokumen_elemen_documents.html', documents=documents)

@dokumen_elemen_bp.route('/dokumen-elemen/<int:doc_id>')
def dokumen_elemen_viewer(doc_id):
    """View document elements with accordion"""
    doc = TestingDokumen.query.get_or_404(doc_id)
    return render_template('dokumen_elemen_viewer.html', doc=doc)

@dokumen_elemen_bp.route('/dokumen-elemen-api/sections/<int:doc_id>')
def api_get_sections(doc_id):
    """API to get all sections for a document with margin info"""
    sections = DokumenSection.query.filter_by(dokumen_id=doc_id)\
        .order_by(DokumenSection.dsec_index).all()
    
    result = []
    for sec in sections:
        # Convert twips to points (1 twip = 1/20 point, 1 point = 1/72 inch)
        # For PDF rendering we need to know the margin positions in points
        twips_per_point = 20
        
        result.append({
            'dsec_id': sec.dsec_id,
            'dsec_index': sec.dsec_index,
            'page_width_twips': sec.dsec_page_width_twips,
            'page_height_twips': sec.dsec_page_height_twips,
            'page_width_pt': sec.dsec_page_width_twips / twips_per_point if sec.dsec_page_width_twips else None,
            'page_height_pt': sec.dsec_page_height_twips / twips_per_point if sec.dsec_page_height_twips else None,
            'orientation': sec.dsec_orientation,
            'margin_top_twips': sec.dsec_margin_top_twips,
            'margin_bottom_twips': sec.dsec_margin_bottom_twips,
            'margin_left_twips': sec.dsec_margin_left_twips,
            'margin_right_twips': sec.dsec_margin_right_twips,
            'margin_top_pt': sec.dsec_margin_top_twips / twips_per_point if sec.dsec_margin_top_twips else None,
            'margin_bottom_pt': sec.dsec_margin_bottom_twips / twips_per_point if sec.dsec_margin_bottom_twips else None,
            'margin_left_pt': sec.dsec_margin_left_twips / twips_per_point if sec.dsec_margin_left_twips else None,
            'margin_right_pt': sec.dsec_margin_right_twips / twips_per_point if sec.dsec_margin_right_twips else None,
            'header_margin_twips': sec.dsec_header_margin_twips,
            'footer_margin_twips': sec.dsec_footer_margin_twips,
            'header_margin_pt': sec.dsec_header_margin_twips / twips_per_point if sec.dsec_header_margin_twips else None,
            'footer_margin_pt': sec.dsec_footer_margin_twips / twips_per_point if sec.dsec_footer_margin_twips else None,
            'gutter_twips': sec.dsec_gutter_twips,
            'gutter_position': sec.dsec_gutter_position
        })
    
    return jsonify({
        'success': True,
        'count': len(result),
        'sections': result
    })

@dokumen_elemen_bp.route('/dokumen-elemen-api/elements/<int:doc_id>')
def api_get_elements(doc_id):
    """API to get all elements for a document (body parts only)"""
    from models import DokumenPart
    
    elements = db.session.query(DokumenElemen, DokumenSection)\
        .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)\
        .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)\
        .filter(DokumenSection.dokumen_id == doc_id)\
        .filter(DokumenPart.dpart_type == 'body')\
        .order_by(DokumenElemen.delemen_sequence)\
        .all()
    
    result = []
    for elem_row in elements:
        elem = elem_row.DokumenElemen
        section = elem_row.DokumenSection
        result.append({
            'dokumen_elemen_id': elem.delemen_id,
            'dokumen_elemen_sequence': elem.delemen_sequence,
            'dokumen_elemen_type': elem.delemen_type,
            'dokumen_elemen_json_tree': elem.delemen_json_tree or {},
            'section_margins': {
                'top_twips': section.dsec_margin_top_twips,
                'bottom_twips': section.dsec_margin_bottom_twips,
                'left_twips': section.dsec_margin_left_twips,
                'right_twips': section.dsec_margin_right_twips
            }
        })
    
    return jsonify({
        'success': True,
        'count': len(result),
        'elements': result
    })


def extract_text_from_json_tree(json_tree):
    """Recursively extract text from dokumen_elemen_json_tree.
    
    Images are converted to context-based placeholders [IMG:abc123] where the hash
    is derived from surrounding text. This enables matching even when image counts
    differ between PDF and Word (e.g., charts in Word but not in PDF).
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


def extract_text_and_images_separately(json_tree):
    """Extract text and images as separate items from dokumen_elemen_json_tree.
    
    Returns:
        dict with:
        - 'text_only': str - text content without image placeholders
        - 'images': list - list of {'placeholder': '[IMG:1]', 'index': 1}
        - 'has_images': bool - True if element contains images
        - 'combined': str - original combined text with placeholders (for fallback)
        - 'ordered_items': list - list of {'type': 'text'|'image', ...} in original order
    """
    if json_tree is None:
        return {'text_only': '', 'images': [], 'has_images': False, 'combined': '', 'ordered_items': []}
    
    # First pass: collect all items in order
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
    
    # Second pass: separate text and images, build ordered list
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
            images.append({
                'placeholder': placeholder,
                'index': image_counter
            })
            combined_parts.append(placeholder)
            ordered_items.append({'type': 'image', 'local_index': image_counter})
    
    return {
        'text_only': ' '.join(text_parts).strip(),
        'images': images,
        'has_images': len(images) > 0,
        'combined': ' '.join(combined_parts).strip(),
        'ordered_items': ordered_items
    }



def extract_cell_text(cell):
    """Extract text from a single table cell
    
    Cell can be:
    - string: "text"
    - dict: {"type": "text", "value": "text"}
    - list of dicts: [{"type": "text", "value": "text"}, ...]
    """
    if isinstance(cell, str):
        return cell
    if isinstance(cell, dict):
        # Single object with type/value
        if cell.get('type') == 'text' and 'value' in cell:
            return str(cell['value'])
        return extract_text_from_json_tree(cell)
    if isinstance(cell, list):
        # Array of objects - extract text from each
        texts = []
        for item in cell:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'value' in item:
                    texts.append(str(item['value']))
                elif item.get('type') == 'math' and 'text' in item:
                    texts.append(str(item['text']))
                else:
                    texts.append(extract_text_from_json_tree(item))
            elif isinstance(item, str):
                texts.append(item)
        return ' '.join(texts)
    return ""


def extract_table_cells(json_tree):
    """Extract cells from table json_tree as separate alignment units
    
    Returns list of {row, col, text} for each cell
    """
    if json_tree is None:
        return []
    
    cells = []
    
    # Try to find rows in content or directly in json_tree
    content = json_tree.get('content', {}) if isinstance(json_tree, dict) else {}
    if isinstance(content, dict):
        rows = content.get('rows', [])
    else:
        rows = json_tree.get('rows', []) if isinstance(json_tree, dict) else []
    
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        for col_idx, cell in enumerate(row.get('cells', [])):
            cell_text = extract_cell_text(cell)
            if cell_text.strip():  # Only include non-empty cells
                cells.append({
                    'row': row_idx,
                    'col': col_idx,
                    'text': cell_text
                })
    
    return cells


def is_table_element(elem_type):
    """Check if element type is a table"""
    if not elem_type:
        return False
    return 'table' in elem_type.lower()


# Register merging alignment routes
from routes.merging_alignment import register_merging_alignment_routes
register_merging_alignment_routes(dokumen_elemen_bp)


@dokumen_elemen_bp.route('/dokumen-elemen-api/simple-align/<int:doc_id>/<int:page>', methods=['POST'])
def api_simple_align(doc_id, page):
    """Simple difflib alignment - multiple groups can match to one element"""
    import difflib
    
    # Get groups from request
    data = request.get_json() or {}
    groups = data.get('groups', [])
    
    if not groups:
        return jsonify({
            'success': True,
            'page': page,
            'total_groups': 0,
            'total_elements': 0,
            'alignments': []
        })
    
    # Get all elements for this document (body parts only)
    from models import DokumenPart
    
    elements = db.session.query(DokumenElemen)\
        .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)\
        .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)\
        .filter(DokumenSection.dokumen_id == doc_id)\
        .filter(DokumenPart.dpart_type == 'body')\
        .order_by(DokumenElemen.delemen_sequence)\
        .all()
    
    # Build element texts (no spaces for matching)
    element_data = []
    for elem in elements:
        text = extract_text_from_json_tree(elem.delemen_json_tree)
        text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
        element_data.append({
            'id': elem.delemen_id,
            'sequence': elem.delemen_sequence,
            'type': elem.delemen_type,
            'text': text,
            'text_no_space': text_no_space.lower()
        })
    
    # Build concatenated groups text (no spaces)
    # Track which characters belong to which group
    groups_concat = ""
    group_char_map = []  # maps each char index to group index
    for i, g in enumerate(groups):
        text = g.get('text', '').replace(' ', '').replace('\n', '').replace('\t', '')
        for _ in text:
            group_char_map.append(i)
        groups_concat += text
    
    groups_concat_lower = groups_concat.lower()
    
    # Build concatenated element text 
    # Track which characters belong to which element
    elements_concat = ""
    elem_char_map = []  # maps each char index to element index
    for i, e in enumerate(element_data):
        for _ in e['text_no_space']:
            elem_char_map.append(i)
        elements_concat += e['text_no_space']
    
    # Run difflib SequenceMatcher
    sm = difflib.SequenceMatcher(None, groups_concat_lower, elements_concat, autojunk=False)
    matching_blocks = sm.get_matching_blocks()
    
    # Map each group to matched element(s)
    group_to_elem = {}  # group_idx -> set of elem_idx
    
    for block in matching_blocks:
        if block.size == 0:
            continue
        
        # For each matched character, map group to element
        for offset in range(block.size):
            g_char_idx = block.a + offset
            e_char_idx = block.b + offset
            
            if g_char_idx < len(group_char_map) and e_char_idx < len(elem_char_map):
                g_idx = group_char_map[g_char_idx]
                e_idx = elem_char_map[e_char_idx]
                
                if g_idx not in group_to_elem:
                    group_to_elem[g_idx] = {}
                if e_idx not in group_to_elem[g_idx]:
                    group_to_elem[g_idx][e_idx] = 0
                group_to_elem[g_idx][e_idx] += 1
    
    # Build alignments - for each group, pick the element with most matched chars
    alignments = []
    for g_idx, elem_counts in group_to_elem.items():
        if not elem_counts:
            continue
        
        # Pick element with highest count
        best_elem_idx = max(elem_counts.keys(), key=lambda x: elem_counts[x])
        matched_chars = elem_counts[best_elem_idx]
        group = groups[g_idx]
        elem = element_data[best_elem_idx]
        
        # Calculate score based on matched chars vs group text length
        group_text_len = len(group.get('text', '').replace(' ', ''))
        score = matched_chars / group_text_len if group_text_len > 0 else 0
        
        alignments.append({
            'group_id': group.get('id'),
            'group_text': group.get('text', '')[:80],
            'element_id': elem['id'],
            'element_sequence': elem['sequence'],
            'element_type': elem['type'],
            'element_text': elem['text'][:80],
            'matched_chars': matched_chars,
            'score': round(score, 3)
        })
    
    # Sort by group_id order
    alignments.sort(key=lambda x: int(x['group_id'].split('_')[-1]) if '_' in x['group_id'] else 0)
    
    return jsonify({
        'success': True,
        'page': page,
        'total_groups': len(groups),
        'total_elements': len(element_data),
        'alignments': alignments
    })


@dokumen_elemen_bp.route('/dokumen-elemen-api/elemen-alignment/<int:doc_id>/<int:page>', methods=['POST'])
def api_elemen_alignment(doc_id, page):
    """Element-centric alignment - shows each element with all matched PDF groups
    
    For tables: align each cell separately, parent table shows no text matching
    For non-tables: align element text as before
    """
    import difflib
    from datetime import datetime
    
    # Get groups from request
    data = request.get_json() or {}
    groups = data.get('groups', [])
    
    if not groups:
        return jsonify({
            'success': True,
            'page': page,
            'timestamp': datetime.now().isoformat(),
            'alignments': []
        })
    
    # Get all elements for this document (body parts only)
    from models import DokumenPart
    
    elements = db.session.query(DokumenElemen)\
        .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)\
        .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)\
        .filter(DokumenSection.dokumen_id == doc_id)\
        .filter(DokumenPart.dpart_type == 'body')\
        .order_by(DokumenElemen.delemen_sequence)\
        .all()
    
    # Build alignment units: for tables, create one unit per cell; for others, use element
    alignment_units = []  # Each unit: {unit_id, elem_id, elem_seq, elem_type, text, text_no_space, is_cell, row, col}
    
    for elem in elements:
        if is_table_element(elem.delemen_type):
            # Extract cells from table
            cells = extract_table_cells(elem.delemen_json_tree)
            for cell in cells:
                text = cell['text']
                text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
                alignment_units.append({
                    'unit_id': f"{elem.delemen_id}_r{cell['row']}_c{cell['col']}",
                    'elem_id': elem.delemen_id,
                    'elem_seq': elem.delemen_sequence,
                    'elem_type': elem.delemen_type,
                    'text': text,
                    'text_no_space': text_no_space.lower(),
                    'is_cell': True,
                    'row': cell['row'],
                    'col': cell['col']
                })
        else:
            # Non-table: use whole element
            text = extract_text_from_json_tree(elem.delemen_json_tree)
            text_no_space = text.replace(' ', '').replace('\n', '').replace('\t', '')
            alignment_units.append({
                'unit_id': str(elem.delemen_id),
                'elem_id': elem.delemen_id,
                'elem_seq': elem.delemen_sequence,
                'elem_type': elem.delemen_type,
                'text': text,
                'text_no_space': text_no_space.lower(),
                'is_cell': False,
                'row': None,
                'col': None
            })
    
    # Build concatenated groups text (no spaces) - keep punctuation as-is
    groups_concat = ""
    group_char_map = []
    for i, g in enumerate(groups):
        text = g.get('text', '').replace(' ', '').replace('\n', '').replace('\t', '')
        for _ in text:
            group_char_map.append(i)
        groups_concat += text
    
    groups_concat_lower = groups_concat.lower()
    
    # Build concatenated units text - strip trailing . and : from OpenXML elements
    units_concat = ""
    unit_char_map = []
    for i, u in enumerate(alignment_units):
        text_stripped = u['text_no_space'].rstrip('.:')
        for _ in text_stripped:
            unit_char_map.append(i)
        units_concat += text_stripped
    
    # Run difflib SequenceMatcher
    sm = difflib.SequenceMatcher(None, groups_concat_lower, units_concat, autojunk=False)
    matching_blocks = sm.get_matching_blocks()
    
    # DEBUG: Log matching blocks
    print(f"\n=== DIFFLIB MATCHING DEBUG (page {page}) ===")
    print(f"Groups concat length: {len(groups_concat_lower)}")
    print(f"Units concat length: {len(units_concat)}")
    print(f"Matching blocks count: {len(matching_blocks)}")
    for i, block in enumerate(matching_blocks[:10]):  # Show first 10 blocks
        if block.size > 0:
            g_text = groups_concat_lower[block.a:block.a+min(block.size, 30)]
            print(f"  Block {i}: g_pos={block.a}, u_pos={block.b}, size={block.size}, text='{g_text}...'")
    
    # Sort matching blocks by UNIT position (b) to ensure proper consumption order
    # This ensures that text appearing earlier in units gets consumed first
    sorted_blocks = sorted(matching_blocks, key=lambda x: x.b)
    
    # Map unit -> groups with CONSUMPTIVE matching
    # Once a unit character position is matched, it won't be matched again
    unit_to_groups = {}  # unit_idx -> { group_idx: matched_char_count }
    consumed_unit_positions = set()  # Track consumed character positions in units_concat
    consumed_groups = set()  # Track consumed groups (each group can only match 1 unit)
    skipped_positions = []  # DEBUG: track skipped positions
    
    # DEBUG: Track match details for each (unit_idx, group_idx) pair
    match_debug = {}  # (u_idx, g_idx) -> { matched_positions: [], matched_chars: "", block_info: [] }
    
    for block in sorted_blocks:
        if block.size == 0:
            continue
        
        for offset in range(block.size):
            g_char_idx = block.a + offset
            u_char_idx = block.b + offset
            
            # Skip if this unit position is already consumed
            if u_char_idx in consumed_unit_positions:
                # DEBUG: Record skipped positions (esp. periods)
                if g_char_idx < len(groups_concat_lower):
                    skipped_char = groups_concat_lower[g_char_idx]
                    if skipped_char in '.,:;':  # Log punctuation
                        skipped_positions.append({
                            'char': skipped_char,
                            'g_pos': g_char_idx,
                            'u_pos': u_char_idx
                        })
                continue
            
            if g_char_idx < len(group_char_map) and u_char_idx < len(unit_char_map):
                g_idx = group_char_map[g_char_idx]
                u_idx = unit_char_map[u_char_idx]
                
                # Skip if this group is already consumed (matched to another unit)
                if g_idx in consumed_groups:
                    continue
                
                # Mark this unit position as consumed
                consumed_unit_positions.add(u_char_idx)
                
                if u_idx not in unit_to_groups:
                    unit_to_groups[u_idx] = {}
                if g_idx not in unit_to_groups[u_idx]:
                    unit_to_groups[u_idx][g_idx] = 0
                unit_to_groups[u_idx][g_idx] += 1
                
                # Mark this group as consumed (can only match 1 unit)
                consumed_groups.add(g_idx)
                
                # DEBUG: Record match detail
                debug_key = (u_idx, g_idx)
                if debug_key not in match_debug:
                    match_debug[debug_key] = {
                        'u_positions': [],
                        'g_positions': [],
                        'matched_chars': []
                    }
                match_debug[debug_key]['u_positions'].append(u_char_idx)
                match_debug[debug_key]['g_positions'].append(g_char_idx)
                if g_char_idx < len(groups_concat_lower):
                    match_debug[debug_key]['matched_chars'].append(groups_concat_lower[g_char_idx])
    
    # Build alignments - organize by element
    elem_alignments = {}  # elem_id -> alignment data
    
    for u_idx, group_counts in unit_to_groups.items():
        if not group_counts:
            continue
        
        unit = alignment_units[u_idx]
        elem_id = unit['elem_id']
        
        # Build matched groups for this unit
        matched_groups = []
        for g_idx, matched_chars in group_counts.items():
            group = groups[g_idx]
            group_text = group.get('text', '')
            group_text_len = len(group_text.replace(' ', ''))
            score = matched_chars / group_text_len if group_text_len > 0 else 0
            
            # Get debug info for this match
            debug_key = (u_idx, g_idx)
            debug_info = match_debug.get(debug_key, {})
            matched_char_str = ''.join(debug_info.get('matched_chars', []))
            u_positions = debug_info.get('u_positions', [])
            g_positions = debug_info.get('g_positions', [])
            
            matched_groups.append({
                'group_id': group.get('id'),
                'text': group_text,
                'bbox': group.get('bbox'),
                'matched_chars': matched_chars,
                'score': round(score, 3),
                'debug': {
                    'matched_str': matched_char_str,  # Full matched sequence
                    'u_range': f"{min(u_positions)}-{max(u_positions)}" if u_positions else "",
                    'g_range': f"{min(g_positions)}-{max(g_positions)}" if g_positions else "",
                    'group_text_len': group_text_len
                }
            })
        
        # Sort matched groups by their original order
        matched_groups.sort(key=lambda x: int(x['group_id'].split('_')[-1]) if x['group_id'] and '_' in x['group_id'] else 0)
        
        # Merge bboxes
        merged_bbox = None
        for mg in matched_groups:
            bbox = mg.get('bbox')
            if bbox and len(bbox) >= 4:
                if merged_bbox is None:
                    merged_bbox = list(bbox)
                else:
                    merged_bbox[0] = min(merged_bbox[0], bbox[0])
                    merged_bbox[1] = min(merged_bbox[1], bbox[1])
                    merged_bbox[2] = max(merged_bbox[2], bbox[2])
                    merged_bbox[3] = max(merged_bbox[3], bbox[3])
        
        if unit['is_cell']:
            # Table cell: add to parent element's cells array
            if elem_id not in elem_alignments:
                elem_alignments[elem_id] = {
                    'element_id': elem_id,
                    'element_sequence': unit['elem_seq'],
                    'element_type': unit['elem_type'],
                    'is_table': True,
                    'element_text': '',  # No text for table parent
                    'matched_groups': [],  # No groups for table parent
                    'merged_bbox': None,
                    'cells': []
                }
            
            elem_alignments[elem_id]['cells'].append({
                'row': unit['row'],
                'col': unit['col'],
                'text': unit['text'],
                'matched_groups': matched_groups,
                'merged_bbox': merged_bbox
            })
            
            # Update parent merged_bbox to include all cells
            if merged_bbox:
                parent_bbox = elem_alignments[elem_id]['merged_bbox']
                if parent_bbox is None:
                    elem_alignments[elem_id]['merged_bbox'] = list(merged_bbox)
                else:
                    elem_alignments[elem_id]['merged_bbox'][0] = min(parent_bbox[0], merged_bbox[0])
                    elem_alignments[elem_id]['merged_bbox'][1] = min(parent_bbox[1], merged_bbox[1])
                    elem_alignments[elem_id]['merged_bbox'][2] = max(parent_bbox[2], merged_bbox[2])
                    elem_alignments[elem_id]['merged_bbox'][3] = max(parent_bbox[3], merged_bbox[3])
        else:
            # Non-table: add directly
            elem_alignments[elem_id] = {
                'element_id': elem_id,
                'element_sequence': unit['elem_seq'],
                'element_type': unit['elem_type'],
                'is_table': False,
                'element_text': unit['text'],
                'matched_groups': matched_groups,
                'merged_bbox': merged_bbox,
                'cells': None
            }
    
    # Sort cells within each table by row, then col
    for elem_id, alignment in elem_alignments.items():
        if alignment.get('cells'):
            alignment['cells'].sort(key=lambda c: (c['row'], c['col']))
    
    # Convert to list and sort by element sequence
    alignments = list(elem_alignments.values())
    alignments.sort(key=lambda x: x['element_sequence'] or 0)
    
    # Build page debug info for frontend overlay - NO TRUNCATION
    matching_blocks_info = []
    for i, block in enumerate(matching_blocks):
        if block.size > 0:
            g_text = groups_concat_lower[block.a:block.a+block.size]
            matching_blocks_info.append({
                'block_num': i,
                'g_pos': block.a,
                'u_pos': block.b,
                'size': block.size,
                'text': g_text  # Full text, no truncation
            })
    
    # Build step-by-step group info
    groups_step_info = []
    for i, g in enumerate(groups):
        g_text = g.get('text', '')
        g_text_no_space = g_text.replace(' ', '').replace('\n', '').replace('\t', '').lower()
        groups_step_info.append({
            'group_id': g.get('id'),
            'text': g_text,
            'text_no_space': g_text_no_space,
            'len': len(g_text_no_space)
        })
    
    # Build step-by-step unit info
    units_step_info = []
    for i, u in enumerate(alignment_units):
        units_step_info.append({
            'unit_id': u['unit_id'],
            'elem_type': u['elem_type'],
            'text': u['text'][:100],  # First 100 for preview
            'text_no_space': u['text_no_space'][:100],
            'len': len(u['text_no_space']),
            'is_cell': u['is_cell'],
            'row': u.get('row'),
            'col': u.get('col')
        })
    
    # ========================
    # TOKEN-BASED MATCHING
    # ========================
    import re
    
    # Tokenize function: split by whitespace and keep punctuation as separate tokens
    def tokenize(text):
        # Split text into words and punctuation marks
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    # Build token lists for groups (keep all punctuation)
    groups_tokens = []
    group_token_map = []  # Which group each token belongs to
    for i, g in enumerate(groups):
        g_text = g.get('text', '')
        tokens = tokenize(g_text)
        for t in tokens:
            groups_tokens.append(t)
            group_token_map.append(i)
    
    # Build token lists for units - strip trailing . or : from each unit (OpenXML)
    units_tokens = []
    unit_token_map = []  # Which unit each token belongs to
    for i, u in enumerate(alignment_units):
        tokens = tokenize(u['text'])
        # Remove trailing . or : from this unit's tokens
        while tokens and tokens[-1] in ['.', ':']:
            tokens.pop()
        for t in tokens:
            units_tokens.append(t)
            unit_token_map.append(i)
    
    # Run token-based SequenceMatcher
    token_sm = difflib.SequenceMatcher(None, groups_tokens, units_tokens, autojunk=False)
    token_matching_blocks = token_sm.get_matching_blocks()
    
    # Build token matching blocks info
    token_blocks_info = []
    for i, block in enumerate(token_matching_blocks):
        if block.size > 0:
            matched_tokens = groups_tokens[block.a:block.a+block.size]
            token_blocks_info.append({
                'block_num': i,
                'g_token_pos': block.a,
                'u_token_pos': block.b,
                'size': block.size,
                'tokens': matched_tokens[:20],  # First 20 tokens for preview
                'tokens_text': ' '.join(matched_tokens)
            })
    
    # ========================
    # BUILD TOKEN-BASED ALIGNMENTS (same format as character-based)
    # ========================
    token_unit_to_groups = {}  # unit_idx -> { group_idx: matched_token_count }
    token_consumed_groups = set()  # Track consumed groups (each group can only match 1 unit)
    
    for block in token_matching_blocks:
        if block.size == 0:
            continue
        for offset in range(block.size):
            g_token_idx = block.a + offset
            u_token_idx = block.b + offset
            
            if g_token_idx < len(group_token_map) and u_token_idx < len(unit_token_map):
                g_idx = group_token_map[g_token_idx]
                u_idx = unit_token_map[u_token_idx]
                
                # Skip if this group is already consumed (matched to another unit)
                if g_idx in token_consumed_groups:
                    continue
                
                if u_idx not in token_unit_to_groups:
                    token_unit_to_groups[u_idx] = {}
                if g_idx not in token_unit_to_groups[u_idx]:
                    token_unit_to_groups[u_idx][g_idx] = 0
                token_unit_to_groups[u_idx][g_idx] += 1
                
                # Mark this group as consumed
                token_consumed_groups.add(g_idx)
    
    # Build token alignments organized by element
    token_elem_alignments = {}
    
    for u_idx, group_counts in token_unit_to_groups.items():
        if not group_counts:
            continue
        
        unit = alignment_units[u_idx]
        elem_id = unit['elem_id']
        
        # Count tokens in unit
        unit_tokens = tokenize(unit['text'])
        unit_token_count = len(unit_tokens)
        
        # Build matched groups
        matched_groups = []
        for g_idx, matched_tokens in group_counts.items():
            group = groups[g_idx]
            group_text = group.get('text', '')
            group_tokens_list = tokenize(group_text)
            group_token_count = len(group_tokens_list)
            score = matched_tokens / group_token_count if group_token_count > 0 else 0
            
            matched_groups.append({
                'group_id': group.get('id'),
                'text': group_text,
                'bbox': group.get('bbox'),
                'matched_tokens': matched_tokens,
                'score': round(score, 3)
            })
        
        matched_groups.sort(key=lambda x: int(x['group_id'].split('_')[-1]) if x['group_id'] and '_' in x['group_id'] else 0)
        
        # Merge bboxes
        merged_bbox = None
        for mg in matched_groups:
            bbox = mg.get('bbox')
            if bbox and len(bbox) >= 4:
                if merged_bbox is None:
                    merged_bbox = list(bbox)
                else:
                    merged_bbox[0] = min(merged_bbox[0], bbox[0])
                    merged_bbox[1] = min(merged_bbox[1], bbox[1])
                    merged_bbox[2] = max(merged_bbox[2], bbox[2])
                    merged_bbox[3] = max(merged_bbox[3], bbox[3])
        
        if unit['is_cell']:
            if elem_id not in token_elem_alignments:
                token_elem_alignments[elem_id] = {
                    'element_id': elem_id,
                    'element_sequence': unit['elem_seq'],
                    'element_type': unit['elem_type'],
                    'is_table': True,
                    'cells': [],
                    'merged_bbox': None
                }
            token_elem_alignments[elem_id]['cells'].append({
                'row': unit.get('row'),
                'col': unit.get('col'),
                'text': unit['text'],
                'matched_groups': matched_groups,
                'merged_bbox': merged_bbox
            })
            if merged_bbox:
                parent_bbox = token_elem_alignments[elem_id]['merged_bbox']
                if parent_bbox is None:
                    token_elem_alignments[elem_id]['merged_bbox'] = list(merged_bbox)
                else:
                    token_elem_alignments[elem_id]['merged_bbox'][0] = min(parent_bbox[0], merged_bbox[0])
                    token_elem_alignments[elem_id]['merged_bbox'][1] = min(parent_bbox[1], merged_bbox[1])
                    token_elem_alignments[elem_id]['merged_bbox'][2] = max(parent_bbox[2], merged_bbox[2])
                    token_elem_alignments[elem_id]['merged_bbox'][3] = max(parent_bbox[3], merged_bbox[3])
        else:
            token_elem_alignments[elem_id] = {
                'element_id': elem_id,
                'element_sequence': unit['elem_seq'],
                'element_type': unit['elem_type'],
                'is_table': False,
                'element_text': unit['text'],
                'matched_groups': matched_groups,
                'merged_bbox': merged_bbox,
                'cells': None
            }
    
    for elem_id, alignment in token_elem_alignments.items():
        if alignment.get('cells'):
            alignment['cells'].sort(key=lambda c: (c['row'], c['col']))
    
    token_alignments = list(token_elem_alignments.values())
    token_alignments.sort(key=lambda x: x['element_sequence'] or 0)
    
    page_debug = {
        'groups_concat_len': len(groups_concat_lower),
        'units_concat_len': len(units_concat),
        'groups_concat': groups_concat_lower,
        'units_concat': units_concat,
        'matching_blocks_count': len(matching_blocks),
        'matching_blocks': matching_blocks_info,
        'skipped_punctuation': skipped_positions,
        'groups_step': groups_step_info,
        'units_step': units_step_info,
        'token_based': {
            'groups_tokens_count': len(groups_tokens),
            'units_tokens_count': len(units_tokens),
            'groups_tokens': groups_tokens[:100],
            'units_tokens': units_tokens[:100],
            'matching_blocks_count': len(token_matching_blocks),
            'matching_blocks': token_blocks_info
        }
    }
    
    # ========================
    # TRACK UNALIGNED GROUPS (CHARACTER-BASED)
    # ========================
    unaligned_groups = []
    for i, g in enumerate(groups):
        if i not in consumed_groups:
            unaligned_groups.append({
                'group_id': g.get('id'),
                'text': g.get('text', ''),
                'bbox': g.get('bbox')
            })
    
    # ========================
    # TRACK UNALIGNED GROUPS (TOKEN-BASED)
    # ========================
    token_unaligned_groups = []
    for i, g in enumerate(groups):
        if i not in token_consumed_groups:
            token_unaligned_groups.append({
                'group_id': g.get('id'),
                'text': g.get('text', ''),
                'bbox': g.get('bbox')
            })
    
    # ========================
    # CALCULATE MISSING WORDS PER ELEMENT/CELL
    # ========================
    def get_aligned_words(matched_groups_list):
        """Get all words from matched groups"""
        words = set()
        for mg in matched_groups_list:
            text = mg.get('text', '')
            for word in tokenize(text):
                if len(word) > 1 or word.isalnum():  # Skip single punctuation
                    words.add(word.lower())
        return words
    
    def find_missing_words(unit_text, aligned_words):
        """Find words in unit that are not in aligned groups"""
        unit_words = tokenize(unit_text)
        missing = []
        for word in unit_words:
            word_lower = word.lower()
            if len(word) > 1 or word.isalnum():  # Skip single punctuation
                if word_lower not in aligned_words:
                    missing.append(word)
        return missing
    
    # Add missing_words to each alignment
    for alignment in alignments:
        if alignment.get('is_table') and alignment.get('cells'):
            # Table: check each cell
            for cell in alignment['cells']:
                aligned_words = get_aligned_words(cell.get('matched_groups', []))
                cell['missing_words'] = find_missing_words(cell.get('text', ''), aligned_words)
        else:
            # Non-table: check element
            aligned_words = get_aligned_words(alignment.get('matched_groups', []))
            alignment['missing_words'] = find_missing_words(alignment.get('element_text', ''), aligned_words)
    
    return jsonify({
        'success': True,
        'page': page,
        'timestamp': datetime.now().isoformat(),
        'total_groups': len(groups),
        'total_elements': len(elements),
        'total_units': len(alignment_units),
        'alignments': alignments,
        'token_alignments': token_alignments,
        'unaligned_groups': unaligned_groups,  # Character-based unaligned
        'token_unaligned_groups': token_unaligned_groups,  # Token-based unaligned
        'page_debug': page_debug
    })

