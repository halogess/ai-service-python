"""
Merging Alignment API
Alignment using extraction data from merging API instead of raw char groups
"""
import re
import difflib
from datetime import datetime
from flask import Blueprint, jsonify, request
from models import db, DokumenElemen, DokumenSection


def is_item_in_header_footer_zone(bbox, section_data, page_height_pt=842):
    """
    Check if a bbox is in header or footer zone based on section margins.
    
    Args:
        bbox: [x0, y0, x1, y1] in PDF points
        section_data: DokumenSection record with margin info
        page_height_pt: page height in points (default A4)
    
    Returns:
        (is_header, is_footer) - tuple of booleans
    """
    if not bbox or len(bbox) < 4:
        return False, False
    
    if not section_data:
        return False, False
    
    # Get margin values in points (1 twip = 1/20 point)
    twips_per_point = 20
    
    # Word already handles landscape internally - width/height are swapped
    # So margin_top is always the top margin regardless of orientation
    margin_top_pt = (section_data.dsec_margin_top_twips or 0) / twips_per_point
    margin_bottom_pt = (section_data.dsec_margin_bottom_twips or 0) / twips_per_point
    page_height = (section_data.dsec_page_height_twips or 0) / twips_per_point or page_height_pt
    
    # bbox[1] = y0 (top), bbox[3] = y1 (bottom)
    y_center = (bbox[1] + bbox[3]) / 2
    
    # Item is in header zone if its center is above margin_top
    is_header = y_center < margin_top_pt
    
    # Item is in footer zone if its center is below (page_height - margin_bottom)
    footer_line = page_height - margin_bottom_pt
    is_footer = y_center > footer_line
    
    return is_header, is_footer


def filter_header_footer_items(pdf_units, section_data, page_height_pt=842):
    """
    Filter out PDF units that are in header or footer zones.
    
    Returns:
        (filtered_units, header_footer_units) - tuple of lists
    """
    if not section_data:
        return pdf_units, []
    
    # Log section data for debugging
    twips_per_point = 20
    margin_top_pt = (section_data.dsec_margin_top_twips or 0) / twips_per_point
    margin_bottom_pt = (section_data.dsec_margin_bottom_twips or 0) / twips_per_point
    page_height = (section_data.dsec_page_height_twips or 0) / twips_per_point or page_height_pt
    footer_line = page_height - margin_bottom_pt
    
    print(f"[Filter] Section: orientation={section_data.dsec_orientation}, "
          f"margin_top={margin_top_pt:.1f}pt, margin_bottom={margin_bottom_pt:.1f}pt, "
          f"page_height={page_height:.1f}pt, footer_line={footer_line:.1f}pt")
    
    filtered = []
    header_footer = []
    
    for unit in pdf_units:
        bbox = unit.get('bbox')
        is_header, is_footer = is_item_in_header_footer_zone(bbox, section_data, page_height_pt)
        
        if is_header or is_footer:
            unit['is_header_footer'] = True
            unit['zone'] = 'header' if is_header else 'footer'
            header_footer.append(unit)
            # Log filtered items
            y_center = (bbox[1] + bbox[3]) / 2 if bbox else 0
            print(f"[Filter] FILTERED as {'header' if is_header else 'footer'}: "
                  f"unit_id={unit.get('unit_id')}, type={unit.get('item_type')}, "
                  f"y_center={y_center:.1f}, text={unit.get('text', '')[:50]}")
        else:
            unit['is_header_footer'] = False
            filtered.append(unit)
    
    return filtered, header_footer


def has_shape_content(json_tree, elem_id=None):
    """
    Check if json_tree contains any shape content.
    Used to identify OpenXML elements that need proximity-based alignment.
    """
    if json_tree is None:
        return False
    
    # Track types found for debugging
    types_found = set()
    
    def check_node(node):
        if isinstance(node, dict):
            node_type = node.get('type')
            if node_type:
                types_found.add(node_type)
            if node_type == 'shape':
                return True
            for value in node.values():
                if check_node(value):
                    return True
        elif isinstance(node, list):
            for item in node:
                if check_node(item):
                    return True
        return False
    
    result = check_node(json_tree)
    
    # Debug logging for tables
    if elem_id and types_found:
        print(f"[ShapeCheck] Element {elem_id}: types={types_found}, has_shape={result}")
    
    return result



def flatten_extraction_items(extraction_items):
    """
    Flatten extraction items to smallest alignment units.
    
    Each unit has: {unit_id, item_idx, type, text, text_normalized, bbox, is_cell, row, col}
    
    Images are converted to context-based placeholders [IMG:hash] using surrounding text.
    This enables matching even when image counts differ between PDF and Word.
    """
    import hashlib
    
    # First pass: collect all items with their metadata
    collected_items = []
    
    for item_idx, item in enumerate(extraction_items):
        item_type = item.get('type', '')
        item_data = item.get('data', {})
        item_bbox = item.get('bbox')
        
        if item_type == 'group':
            text = item_data.get('text', '')
            if text.strip():
                collected_items.append({
                    'item_idx': item_idx,
                    'item_type': item_type,
                    'text': text,
                    'bbox': item_bbox,
                    'is_cell': False,
                    'row': None,
                    'col': None,
                    'is_image': False
                })
        
        elif item_type == 'paragraph':
            # Handle paragraph items (e.g., table captions like "Tabel 5.16")
            text = item_data.get('text', '')
            if text.strip():
                collected_items.append({
                    'item_idx': item_idx,
                    'item_type': item_type,
                    'text': text,
                    'bbox': item_bbox,
                    'is_cell': False,
                    'row': None,
                    'col': None,
                    'is_image': False
                })
        
        elif item_type == 'hline_table':
            # hline_table (from horizontal line detection): treat as single unit
            # This simplifies alignment for borderless tables where cell detection is unreliable
            all_cell_texts = []
            rows = item_data.get('rows', [])
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                for col_idx, cell in enumerate(cells):
                    cell_text = extract_cell_content_text(cell)
                    if cell_text.strip():
                        all_cell_texts.append(cell_text.strip())
            
            if all_cell_texts:
                table_text = ' '.join(all_cell_texts)
                collected_items.append({
                    'item_idx': item_idx,
                    'item_type': item_type,
                    'text': table_text,
                    'bbox': item_bbox,
                    'is_cell': False,
                    'is_hline_table_unit': True,  # Mark as hline_table unit
                    'row': None,
                    'col': None,
                    'is_image': False
                })
        
        elif item_type == 'table':
            # Regular table (from find_tables): cell-by-cell for precise alignment
            rows = item_data.get('rows', [])
            for row_idx, row in enumerate(rows):
                cells = row.get('cells', [])
                for col_idx, cell in enumerate(cells):
                    cell_text = extract_cell_content_text(cell)
                    if cell_text.strip():
                        collected_items.append({
                            'item_idx': item_idx,
                            'item_type': item_type,
                            'text': cell_text,
                            'bbox': cell.get('bbox'),
                            'is_cell': True,
                            'row': row_idx,
                            'col': col_idx,
                            'table_bbox': item_bbox,
                            'is_image': False
                        })
        
        elif item_type == 'shape':
            text = item_data.get('text', '')
            if text.strip():
                collected_items.append({
                    'item_idx': item_idx,
                    'item_type': item_type,
                    'text': text,
                    'bbox': item_bbox,
                    'is_cell': False,
                    'row': None,
                    'col': None,
                    'is_image': False
                })
        
        elif item_type == 'image':
            collected_items.append({
                'item_idx': item_idx,
                'item_type': item_type,
                'text': None,  # Will be replaced with placeholder
                'bbox': item_bbox,
                'is_cell': False,
                'row': None,
                'col': None,
                'is_image': True
            })

    # Merge consecutive shape items into a single unit to avoid
    # fragmenting a single visual element into many shapes.
    collected_items = merge_consecutive_shape_items(collected_items)

    # Second pass: generate count-based placeholders for images
    # Use simple numbering: [IMG:1], [IMG:2], etc. based on order on page
    pdf_units = []
    unit_counter = 0
    image_counter = 0  # Count images sequentially
    
    for i, item in enumerate(collected_items):
        if item['is_image']:
            image_counter += 1
            img_placeholder = '[IMG]'  # Generic placeholder - matching by position order
            
            pdf_units.append({
                'unit_id': f'pdf_{unit_counter}',
                'item_idx': item['item_idx'],
                'item_type': item['item_type'],
                'text': img_placeholder,
                'text_normalized': img_placeholder.lower(),
                'bbox': item['bbox'],
                'is_cell': False,
                'row': None,
                'col': None,
                'is_page_number': False
            })

        else:
            # Regular text item
            pdf_units.append({
                'unit_id': f'pdf_{unit_counter}',
                'item_idx': item['item_idx'],
                'item_type': item['item_type'],
                'text': item['text'],
                'text_normalized': normalize_text(item['text']),
                'bbox': item['bbox'],
                'is_cell': item['is_cell'],
                'is_hline_table_unit': item.get('is_hline_table_unit', False),
                'row': item['row'],
                'col': item['col'],
                'table_bbox': item.get('table_bbox'),
                'is_page_number': is_likely_page_number(item['text'], item['bbox']) if not item['is_cell'] else False
            })
        
        unit_counter += 1
    
    return pdf_units


def merge_bboxes(bboxes):
    """Merge multiple bboxes into a single encompassing bbox."""
    valid = [b for b in bboxes if b and len(b) >= 4]
    if not valid:
        return None
    x0 = min(b[0] for b in valid)
    y0 = min(b[1] for b in valid)
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    return [x0, y0, x1, y1]


def merge_consecutive_shape_items(items):
    """Merge consecutive shape items into a single item."""
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
        merged_bbox = merge_bboxes([i.get('bbox') for i in cluster])
        merged.append({
            'item_idx': cluster[0].get('item_idx'),
            'item_type': 'shape',
            'text': merged_text,
            'bbox': merged_bbox,
            'is_cell': False,
            'row': None,
            'col': None,
            'is_image': False,
            'shape_count': len(cluster),
            'merged_item_indices': [i.get('item_idx') for i in cluster]
        })

    return merged

def extract_cell_content_text(cell):
    """Extract text from cell content array.
    
    Images are converted to count-based placeholders [IMG:1], [IMG:2], etc.
    """
    texts = []
    image_counter = 0
    content = cell.get('content', [])
    
    if isinstance(content, list):
        for idx, c in enumerate(content):
            if isinstance(c, dict):
                if c.get('type') == 'text' and c.get('text'):
                    texts.append(c.get('text', ''))
                elif c.get('type') == 'image':
                    image_counter += 1
                    texts.append(f'[IMG:{image_counter}]')
    
    return ' '.join(texts)



def normalize_text(text):
    """Normalize text for matching: 
    - Convert Unicode mathematical characters to ASCII
    - Normalize Greek letters
    - Handle subscripts/superscripts
    - Lowercase
    - Remove whitespace
    """
    if not text:
        return ''
    
    # Build comprehensive mapping for mathematical alphanumeric symbols
    # Reference: Unicode Block U+1D400 - U+1D7FF (Mathematical Alphanumeric Symbols)
    
    result = []
    for char in text:
        code = ord(char)
        normalized = None
        
        # ============================================
        # LATIN LETTERS (A-Z, a-z)
        # ============================================
        
        # Mathematical Bold (U+1D400-U+1D433)
        if 0x1D400 <= code <= 0x1D419:      # Bold A-Z
            normalized = chr(ord('A') + (code - 0x1D400))
        elif 0x1D41A <= code <= 0x1D433:    # Bold a-z
            normalized = chr(ord('a') + (code - 0x1D41A))
        
        # Mathematical Italic (U+1D434-U+1D467)
        elif 0x1D434 <= code <= 0x1D44D:    # Italic A-Z
            normalized = chr(ord('A') + (code - 0x1D434))
        elif 0x1D44E <= code <= 0x1D467:    # Italic a-z
            # Note: U+1D455 is a hole (reserved), but Unicode sequence handles this
            # The codepoints are: a(0x1D44E)..g(0x1D454), HOLE(0x1D455), i(0x1D456)..z(0x1D467)
            # But the offset calculation still works because the hole is skipped in the standard
            if code == 0x1D455:  # Hole - shouldn't exist, but handle gracefully
                normalized = 'h'  # Map to regular 'h' just in case
            elif code < 0x1D455:  # a-g (before hole)
                normalized = chr(ord('a') + (code - 0x1D44E))
            else:  # i-z (after hole) - code is 0x1D456 to 0x1D467
                # 0x1D456 = i (letter index 8), so offset is (code - 0x1D44E) which gives correct letter
                # BUT we need to account for the missing 'h' position
                # Actually the Unicode maps: 0x1D456 → i, so offset = 8, and i is letter 8. Correct!
                # The issue is there's no 'h' in the sequence, so:
                # 0x1D456 = i (should be letter 8) → offset = 0x1D456 - 0x1D44E = 8 → chr(ord('a')+8) = 'i' ✓
                # 0x1D465 = x (should be letter 23) → offset = 0x1D465 - 0x1D44E = 23 → chr(ord('a')+23) = 'x' ✓
                normalized = chr(ord('a') + (code - 0x1D44E))
        
        # Mathematical Bold Italic (U+1D468-U+1D49B)
        elif 0x1D468 <= code <= 0x1D481:    # Bold Italic A-Z
            normalized = chr(ord('A') + (code - 0x1D468))
        elif 0x1D482 <= code <= 0x1D49B:    # Bold Italic a-z
            normalized = chr(ord('a') + (code - 0x1D482))
        
        # Mathematical Script (U+1D49C-U+1D4CF)
        elif 0x1D49C <= code <= 0x1D4B5:    # Script A-Z (with holes)
            normalized = chr(ord('A') + (code - 0x1D49C))
        elif 0x1D4B6 <= code <= 0x1D4CF:    # Script a-z (with holes)
            normalized = chr(ord('a') + (code - 0x1D4B6))
        
        # Mathematical Bold Script (U+1D4D0-U+1D503)
        elif 0x1D4D0 <= code <= 0x1D4E9:    # Bold Script A-Z
            normalized = chr(ord('A') + (code - 0x1D4D0))
        elif 0x1D4EA <= code <= 0x1D503:    # Bold Script a-z
            normalized = chr(ord('a') + (code - 0x1D4EA))
        
        # Mathematical Fraktur (U+1D504-U+1D537)
        elif 0x1D504 <= code <= 0x1D51C:    # Fraktur A-Z (with holes)
            normalized = chr(ord('A') + (code - 0x1D504))
        elif 0x1D51E <= code <= 0x1D537:    # Fraktur a-z
            normalized = chr(ord('a') + (code - 0x1D51E))
        
        # Mathematical Double-Struck (U+1D538-U+1D56B)
        elif 0x1D538 <= code <= 0x1D550:    # Double-Struck A-Z (with holes)
            normalized = chr(ord('A') + (code - 0x1D538))
        elif 0x1D552 <= code <= 0x1D56B:    # Double-Struck a-z
            normalized = chr(ord('a') + (code - 0x1D552))
        
        # Mathematical Bold Fraktur (U+1D56C-U+1D59F)
        elif 0x1D56C <= code <= 0x1D585:    # Bold Fraktur A-Z
            normalized = chr(ord('A') + (code - 0x1D56C))
        elif 0x1D586 <= code <= 0x1D59F:    # Bold Fraktur a-z
            normalized = chr(ord('a') + (code - 0x1D586))
        
        # Mathematical Sans-Serif (U+1D5A0-U+1D5D3)
        elif 0x1D5A0 <= code <= 0x1D5B9:    # Sans A-Z
            normalized = chr(ord('A') + (code - 0x1D5A0))
        elif 0x1D5BA <= code <= 0x1D5D3:    # Sans a-z
            normalized = chr(ord('a') + (code - 0x1D5BA))
        
        # Mathematical Sans-Serif Bold (U+1D5D4-U+1D607)
        elif 0x1D5D4 <= code <= 0x1D5ED:    # Sans Bold A-Z
            normalized = chr(ord('A') + (code - 0x1D5D4))
        elif 0x1D5EE <= code <= 0x1D607:    # Sans Bold a-z
            normalized = chr(ord('a') + (code - 0x1D5EE))
        
        # Mathematical Sans-Serif Italic (U+1D608-U+1D63B)
        elif 0x1D608 <= code <= 0x1D621:    # Sans Italic A-Z
            normalized = chr(ord('A') + (code - 0x1D608))
        elif 0x1D622 <= code <= 0x1D63B:    # Sans Italic a-z
            normalized = chr(ord('a') + (code - 0x1D622))
        
        # Mathematical Sans-Serif Bold Italic (U+1D63C-U+1D66F)
        elif 0x1D63C <= code <= 0x1D655:    # Sans Bold Italic A-Z
            normalized = chr(ord('A') + (code - 0x1D63C))
        elif 0x1D656 <= code <= 0x1D66F:    # Sans Bold Italic a-z
            normalized = chr(ord('a') + (code - 0x1D656))
        
        # Mathematical Monospace (U+1D670-U+1D6A3)
        elif 0x1D670 <= code <= 0x1D689:    # Monospace A-Z
            normalized = chr(ord('A') + (code - 0x1D670))
        elif 0x1D68A <= code <= 0x1D6A3:    # Monospace a-z
            normalized = chr(ord('a') + (code - 0x1D68A))
        
        # ============================================
        # GREEK LETTERS
        # ============================================
        
        # Mathematical Bold Greek (U+1D6A8-U+1D6E1)
        elif 0x1D6A8 <= code <= 0x1D6C0:    # Bold Greek capitals
            normalized = chr(0x0391 + (code - 0x1D6A8))  # Map to regular Greek
        elif 0x1D6C2 <= code <= 0x1D6DA:    # Bold Greek small
            normalized = chr(0x03B1 + (code - 0x1D6C2))
        
        # Mathematical Italic Greek (U+1D6E2-U+1D71B)
        elif 0x1D6E2 <= code <= 0x1D6FA:    # Italic Greek capitals
            normalized = chr(0x0391 + (code - 0x1D6E2))
        elif 0x1D6FC <= code <= 0x1D714:    # Italic Greek small
            normalized = chr(0x03B1 + (code - 0x1D6FC))
        
        # Mathematical Bold Italic Greek (U+1D71C-U+1D755)
        elif 0x1D71C <= code <= 0x1D734:    # Bold Italic Greek capitals
            normalized = chr(0x0391 + (code - 0x1D71C))
        elif 0x1D736 <= code <= 0x1D74E:    # Bold Italic Greek small
            normalized = chr(0x03B1 + (code - 0x1D736))
        
        # Mathematical Sans-Serif Bold Greek (U+1D756-U+1D78F)
        elif 0x1D756 <= code <= 0x1D76E:    # Sans Bold Greek capitals
            normalized = chr(0x0391 + (code - 0x1D756))
        elif 0x1D770 <= code <= 0x1D788:    # Sans Bold Greek small
            normalized = chr(0x03B1 + (code - 0x1D770))
        
        # Mathematical Sans-Serif Bold Italic Greek (U+1D790-U+1D7C9)
        elif 0x1D790 <= code <= 0x1D7A8:    # Sans Bold Italic Greek capitals
            normalized = chr(0x0391 + (code - 0x1D790))
        elif 0x1D7AA <= code <= 0x1D7C2:    # Sans Bold Italic Greek small
            normalized = chr(0x03B1 + (code - 0x1D7AA))
        
        # ============================================
        # PARTIAL DIFFERENTIAL SYMBOLS (various styles)
        # ============================================
        
        # Mathematical Italic Partial Differential (U+1D715)
        elif code == 0x1D715:
            normalized = '∂'
        # Mathematical Bold Partial Differential (U+1D6DB)
        elif code == 0x1D6DB:
            normalized = '∂'
        # Mathematical Bold Italic Partial Differential (U+1D74F)
        elif code == 0x1D74F:
            normalized = '∂'
        # Mathematical Sans-Serif Bold Partial Differential (U+1D789)
        elif code == 0x1D789:
            normalized = '∂'
        # Mathematical Sans-Serif Bold Italic Partial Differential (U+1D7C3)
        elif code == 0x1D7C3:
            normalized = '∂'
        # Regular Partial Differential (U+2202)
        elif code == 0x2202:
            normalized = '∂'
        
        # ============================================
        # DIGITS
        # ============================================
        
        # Mathematical Bold Digits (U+1D7CE-U+1D7D7)
        elif 0x1D7CE <= code <= 0x1D7D7:
            normalized = chr(ord('0') + (code - 0x1D7CE))
        # Mathematical Double-Struck Digits (U+1D7D8-U+1D7E1)
        elif 0x1D7D8 <= code <= 0x1D7E1:
            normalized = chr(ord('0') + (code - 0x1D7D8))
        # Mathematical Sans-Serif Digits (U+1D7E2-U+1D7EB)
        elif 0x1D7E2 <= code <= 0x1D7EB:
            normalized = chr(ord('0') + (code - 0x1D7E2))
        # Mathematical Sans-Serif Bold Digits (U+1D7EC-U+1D7F5)
        elif 0x1D7EC <= code <= 0x1D7F5:
            normalized = chr(ord('0') + (code - 0x1D7EC))
        # Mathematical Monospace Digits (U+1D7F6-U+1D7FF)
        elif 0x1D7F6 <= code <= 0x1D7FF:
            normalized = chr(ord('0') + (code - 0x1D7F6))
        
        # ============================================
        # SUPERSCRIPTS AND SUBSCRIPTS
        # ============================================
        
        # Superscript digits
        elif code == 0x2070:  # ⁰
            normalized = '0'
        elif code == 0x00B9:  # ¹
            normalized = '1'
        elif code == 0x00B2:  # ²
            normalized = '2'
        elif code == 0x00B3:  # ³
            normalized = '3'
        elif 0x2074 <= code <= 0x2079:  # ⁴-⁹
            normalized = chr(ord('0') + (code - 0x2070))
        
        # Superscript letters
        elif code == 0x1D43:  # ᵃ
            normalized = 'a'
        elif code == 0x1D47:  # ᵇ
            normalized = 'b'
        elif code == 0x1D9C:  # ᶜ
            normalized = 'c'
        elif code == 0x1D48:  # ᵈ
            normalized = 'd'
        elif code == 0x1D49:  # ᵉ
            normalized = 'e'
        elif code == 0x1DA0:  # ᶠ
            normalized = 'f'
        elif code == 0x1D4D:  # ᵍ
            normalized = 'g'
        elif code == 0x02B0:  # ʰ
            normalized = 'h'
        elif code == 0x2071:  # ⁱ
            normalized = 'i'
        elif code == 0x02B2:  # ʲ
            normalized = 'j'
        elif code == 0x1D4F:  # ᵏ
            normalized = 'k'
        elif code == 0x02E1:  # ˡ
            normalized = 'l'
        elif code == 0x1D50:  # ᵐ
            normalized = 'm'
        elif code == 0x207F:  # ⁿ
            normalized = 'n'
        elif code == 0x1D52:  # ᵒ
            normalized = 'o'
        elif code == 0x1D56:  # ᵖ
            normalized = 'p'
        elif code == 0x02B3:  # ʳ
            normalized = 'r'
        elif code == 0x02E2:  # ˢ
            normalized = 's'
        elif code == 0x1D57:  # ᵗ
            normalized = 't'
        elif code == 0x1D58:  # ᵘ
            normalized = 'u'
        elif code == 0x1D5B:  # ᵛ
            normalized = 'v'
        elif code == 0x02B7:  # ʷ
            normalized = 'w'
        elif code == 0x02E3:  # ˣ
            normalized = 'x'
        elif code == 0x02B8:  # ʸ
            normalized = 'y'
        elif code == 0x1DBB:  # ᶻ
            normalized = 'z'
        
        # Subscript digits (U+2080-U+2089)
        elif 0x2080 <= code <= 0x2089:
            normalized = chr(ord('0') + (code - 0x2080))
        
        # Subscript letters
        elif code == 0x2090:  # ₐ
            normalized = 'a'
        elif code == 0x2091:  # ₑ
            normalized = 'e'
        elif code == 0x2095:  # ₕ
            normalized = 'h'
        elif code == 0x1D62:  # ᵢ
            normalized = 'i'
        elif code == 0x2C7C:  # ⱼ
            normalized = 'j'
        elif code == 0x2096:  # ₖ
            normalized = 'k'
        elif code == 0x2097:  # ₗ
            normalized = 'l'
        elif code == 0x2098:  # ₘ
            normalized = 'm'
        elif code == 0x2099:  # ₙ
            normalized = 'n'
        elif code == 0x2092:  # ₒ
            normalized = 'o'
        elif code == 0x209A:  # ₚ
            normalized = 'p'
        elif code == 0x1D63:  # ᵣ
            normalized = 'r'
        elif code == 0x209B:  # ₛ
            normalized = 's'
        elif code == 0x209C:  # ₜ
            normalized = 't'
        elif code == 0x1D64:  # ᵤ
            normalized = 'u'
        elif code == 0x1D65:  # ᵥ
            normalized = 'v'
        elif code == 0x2093:  # ₓ
            normalized = 'x'
        
        # ============================================
        # MATH OPERATORS AND SYMBOLS
        # ============================================
        
        # Minus variants
        elif char in '−–—‐‑‒―':
            normalized = '-'
        # Multiplication
        elif char in '×∙·•⋅':
            normalized = '*'
        # Division
        elif char in '÷∕':
            normalized = '/'
        # Plus/Minus
        elif char == '±':
            normalized = '+-'
        elif char == '∓':
            normalized = '-+'
        # Equals variants
        elif char in '＝⁼₌':
            normalized = '='
        # Less/Greater variants
        elif char in '＜‹〈⟨':
            normalized = '<'
        elif char in '＞›〉⟩':
            normalized = '>'
        elif char in '≤≦⩽':
            normalized = '<='
        elif char in '≥≧⩾':
            normalized = '>='
        # Arrows to nothing (remove)
        elif char in '→←↑↓↔↕⇒⇐⇑⇓⇔':
            normalized = ''
        # Prime variants
        elif char in '′':
            normalized = "'"
        elif char in '″':
            normalized = "''"
        # Fractions
        elif char == '½':
            normalized = '1/2'
        elif char == '⅓':
            normalized = '1/3'
        elif char == '¼':
            normalized = '1/4'
        elif char == '⅔':
            normalized = '2/3'
        elif char == '¾':
            normalized = '3/4'
        # Fullwidth ASCII (U+FF01-U+FF5E)
        elif 0xFF01 <= code <= 0xFF5E:
            normalized = chr(code - 0xFF00 + 0x20)
        
        # Default: keep character as-is
        if normalized is None:
            normalized = char
        
        result.append(normalized)
    
    # Remove all whitespace and lowercase
    return ''.join(''.join(result).lower().split())


def is_standalone_number(text):
    """
    Check if text is a standalone short number (potential page number candidate).
    Returns True if it's a 1-4 digit number that needs context checking.
    """
    if not text:
        return False
    
    # Clean the text - remove common decorations
    cleaned = text.strip()
    cleaned = cleaned.strip('-').strip()  # Remove leading/trailing dashes
    cleaned = cleaned.strip('.').strip()  # Remove leading/trailing dots
    
    # Check if purely numeric and short
    if cleaned.isdigit() and len(cleaned) <= 4:
        return True
    
    # Check patterns like "- 7 -", "Page 7", etc.
    import re
    page_patterns = [
        r'^-?\s*\d{1,4}\s*-?$',  # "7", "-7-", "- 7 -"
        r'^page\s*\d{1,4}$',     # "Page 7"
        r'^hal\.?\s*\d{1,4}$',   # "Hal. 7", "Hal 7"
        r'^\d{1,4}\s*/\s*\d{1,4}$',  # "7/10"
    ]
    for pattern in page_patterns:
        if re.match(pattern, cleaned.lower()):
            return True
    
    return False


def detect_suspicious_page_numbers(pdf_units, pdf_unit_assignment, openxml_to_pdf):
    """
    Post-process to detect PDF units that are likely page numbers based on context.
    
    A match is suspicious if:
    1. PDF unit is a standalone short number (1-4 digits)
    2. The PDF units before and after don't match the same OpenXML element
    3. OR the match ratio is very low (single digit matched to 100+ char element)
    """
    suspicious_indices = set()
    
    for pdf_idx, u in enumerate(pdf_units):
        if pdf_idx not in pdf_unit_assignment:
            continue
            
        if not is_standalone_number(u.get('text', '')):
            continue
        
        openxml_idx = pdf_unit_assignment[pdf_idx]
        
        # Check context: do surrounding PDF units match the same OpenXML element?
        prev_match_same = False
        next_match_same = False
        
        # Check previous PDF unit
        if pdf_idx > 0:
            prev_idx = pdf_idx - 1
            if prev_idx in pdf_unit_assignment and pdf_unit_assignment[prev_idx] == openxml_idx:
                prev_match_same = True
        
        # Check next PDF unit
        if pdf_idx < len(pdf_units) - 1:
            next_idx = pdf_idx + 1
            if next_idx in pdf_unit_assignment and pdf_unit_assignment[next_idx] == openxml_idx:
                next_match_same = True
        
        # If neither neighbor matches the same OpenXML element, it's suspicious
        if not prev_match_same and not next_match_same:
            suspicious_indices.add(pdf_idx)
    
    return suspicious_indices


def is_likely_page_number(text, bbox, page_height=None):
    """Simple wrapper for backward compatibility - uses is_standalone_number."""
    return is_standalone_number(text)


def build_openxml_units(elements, page_sequence_range=None):
    """
    Build alignment units from OpenXML elements.
    Similar to existing logic but extracted for clarity.
    Adds has_shape flag to identify elements needing proximity-based alignment.
    
    For non-table paragraphs with both text and images, creates separate units
    for text and each image, all referencing the same element_id.
    
    Args:
        elements: List of DokumenElemen objects
        page_sequence_range: Optional tuple (min_seq, max_seq) to filter image counting.
                            When provided, only images from elements in this range are numbered.
                            This ensures per-page image numbering matches PDF extraction.
    
    Returns: (units, table_debug) where table_debug is a list of table processing info
    """
    from routes.dokumen_elemen_routes import (
        is_table_element, extract_table_cells, extract_text_from_json_tree,
        extract_text_and_images_separately
    )

    
    units = []
    table_debug = []  # Track table processing for debugging
    global_image_counter = 0  # Counter for image numbering
    
    for elem in elements:
        elem_has_shape = has_shape_content(elem.delemen_json_tree, elem.delemen_id)
        
        
        if is_table_element(elem.delemen_type):
            # Table: one unit per cell
            cells = extract_table_cells(elem.delemen_json_tree)
            
            # Record debug info for this table
            table_info = {
                'elem_id': elem.delemen_id,
                'cells_count': len(cells),
                'has_shape': elem_has_shape,
                'units_created': 0,
                'action': ''
            }
            
            # Debug logging for tables
            print(f"[BuildUnits] Table {elem.delemen_id}: cells={len(cells)}, has_shape={elem_has_shape}")
            
            if cells:
                table_info['action'] = f'created {len(cells)} cell units'
                table_info['units_created'] = len(cells)
                table_info['cell_details'] = []  # Debug: track what cells contain
                for cell in cells:
                    text = cell['text']
                    unit_id = f"{elem.delemen_id}_r{cell['row']}_c{cell['col']}"
                    table_info['cell_details'].append({
                        'unit_id': unit_id,
                        'row': cell['row'],
                        'col': cell['col'],
                        'text_preview': text[:50] if text else '(empty)'
                    })
                    units.append({
                        'unit_id': unit_id,
                        'elem_id': elem.delemen_id,
                        'elem_seq': elem.delemen_sequence,
                        'elem_type': elem.delemen_type,
                        'text': text,
                        'text_normalized': normalize_text(text).rstrip('.:'),
                        'is_cell': True,
                        'row': cell['row'],
                        'col': cell['col'],
                        'has_shape': elem_has_shape
                    })
            elif elem_has_shape:
                # Table with shapes but no extractable text cells
                # Create a placeholder unit so Phase 2 can align it
                table_info['action'] = 'created shape placeholder'
                table_info['units_created'] = 1
                print(f"[BuildUnits] Creating placeholder for shape-only table {elem.delemen_id}")
                units.append({
                    'unit_id': str(elem.delemen_id),
                    'elem_id': elem.delemen_id,
                    'elem_seq': elem.delemen_sequence,
                    'elem_type': elem.delemen_type,
                    'text': '',  # No text, just shape
                    'text_normalized': '',
                    'is_cell': False,
                    'row': None,
                    'col': None,
                    'has_shape': True  # This is a shape-only table
                })
            else:
                # Table with no cells and no shapes - THIS SHOULD NOT HAPPEN OFTEN
                table_info['action'] = 'SKIPPED (no cells, no shapes)'
                print(f"[BuildUnits] WARNING: Table {elem.delemen_id} has no cells and no shapes - SKIPPED!")
            
            table_debug.append(table_info)
        else:
            # Non-table: check if element has both text and images
            content = extract_text_and_images_separately(elem.delemen_json_tree)
            
            if content['has_images']:
                # Element has images - create units IN ORIGINAL ORDER from content
                # This handles both: images-first-then-text AND text-first-then-images
                
                text_unit_created = False  # Track if we already created text unit
                
                for item in content['ordered_items']:
                    if item['type'] == 'image':
                        # Use generic [IMG] placeholder - matching by position order
                        global_image_counter += 1
                        global_placeholder = '[IMG]'
                        
                        units.append({
                            'unit_id': f"{elem.delemen_id}_img{global_image_counter}",
                            'elem_id': elem.delemen_id,
                            'elem_seq': elem.delemen_sequence,
                            'elem_type': elem.delemen_type,
                            'text': global_placeholder,
                            'text_normalized': global_placeholder.lower(),
                            'is_cell': False,
                            'row': None,
                            'col': None,
                            'has_shape': True,
                            'is_text_part': False,
                            'is_image_part': True,
                            'image_index': global_image_counter
                        })
                    elif item['type'] == 'text' and not text_unit_created:
                        # Create text unit (only once, combining all text)
                        if content['text_only']:
                            units.append({
                                'unit_id': f"{elem.delemen_id}_text",
                                'elem_id': elem.delemen_id,
                                'elem_seq': elem.delemen_sequence,
                                'elem_type': elem.delemen_type,
                                'text': content['text_only'],
                                'text_normalized': normalize_text(content['text_only']).rstrip('.:'),
                                'is_cell': False,
                                'row': None,
                                'col': None,
                                'has_shape': elem_has_shape,
                                'is_text_part': True,
                                'is_image_part': False
                            })
                            text_unit_created = True
            else:
                # Element has only text (no images) - use combined text as before
                text = content['combined'] if content['combined'] else extract_text_from_json_tree(elem.delemen_json_tree)
                units.append({
                    'unit_id': str(elem.delemen_id),
                    'elem_id': elem.delemen_id,
                    'elem_seq': elem.delemen_sequence,
                    'elem_type': elem.delemen_type,
                    'text': text,
                    'text_normalized': normalize_text(text).rstrip('.:'),
                    'is_cell': False,
                    'row': None,
                    'col': None,
                    'has_shape': elem_has_shape
                })

    
    return units, table_debug




def perform_char_alignment(pdf_units, openxml_units, min_openxml_idx=0):
    """
    Perform character-based alignment using difflib SequenceMatcher.
    
    Args:
        pdf_units: List of PDF units to align
        openxml_units: List of OpenXML units to align against
        min_openxml_idx: Minimum OpenXML unit index to match (for cross-page tracking)
                         Units with index < min_openxml_idx will be skipped
    
    Returns (alignments, unaligned_pdf_indices, debug_info)
    """
    # Build concatenated strings with position tracking
    pdf_concat = ''
    pdf_char_map = []  # maps char index -> pdf unit index
    pdf_unit_ranges = []  # [(start, end, unit_idx), ...]
    page_number_indices = set()  # Track potential page number unit indices for debug
    
    for i, u in enumerate(pdf_units):
        # Track potential page numbers for debug, but DON'T skip them
        # Pre-filtering was too aggressive - let post-processing handle it
        if u.get('is_page_number', False):
            page_number_indices.add(i)
            # Continue processing normally - don't skip!
            
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
    openxml_char_map = []  # maps char index -> openxml unit index
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
    
    # Run SequenceMatcher
    sm = difflib.SequenceMatcher(None, pdf_concat, openxml_concat, autojunk=False)
    matching_blocks = sm.get_matching_blocks()
    
    # Sort by openxml position for consumptive matching
    sorted_blocks = sorted(matching_blocks, key=lambda x: x.b)
    
    # Log gap analysis to file
    with open('gap_analysis.log', 'w', encoding='utf-8') as gap_log:
        gap_log.write("=" * 80 + "\n")
        gap_log.write("GAP ANALYSIS - What OpenXML content is NOT being matched\n")
        gap_log.write("=" * 80 + "\n\n")
        
        prev_end_ox = 0
        for i, block in enumerate(sorted_blocks):
            if block.size == 0:
                continue
            gap = block.b - prev_end_ox
            if gap > 50:  # Only log significant gaps
                gap_log.write(f"\n[GAP {i}] OX positions {prev_end_ox} to {block.b} (size: {gap} chars)\n")
                gap_content = openxml_concat[prev_end_ox:block.b]
                gap_log.write(f"  Content: \"{gap_content[:200]}...\"\n")
                # Find which units are in this gap
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
    
    # Consumptive matching with detailed logging
    consumed_openxml_positions = set()
    pdf_unit_assignment = {}  # pdf_idx -> openxml_idx (first assignment)
    last_assigned_openxml_idx = -1  # Track the highest openxml_idx assigned so far (to prevent backward matching)
    
    # Map: openxml_unit_idx -> {pdf_unit_idx: matched_char_count}
    openxml_to_pdf = {}
    match_debug = {}
    matching_log = []  # Block-level log
    traversal_log = []  # Detailed step-by-step log for each char
    
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
            
            # Build traversal log entry
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

                # Check if this PDF unit is already assigned to a different openxml unit
                if pdf_idx in pdf_unit_assignment and not is_shape_pdf:
                    # Already assigned - only allow matching to the same openxml unit
                    if pdf_unit_assignment[pdf_idx] != openxml_idx:
                        log_entry['action'] = 'SKIP'
                        log_entry['reason'] = f'pdf_assigned_to_different: {pdf_unit_assignment[pdf_idx]}'
                        traversal_log.append(log_entry)
                        continue
                    else:
                        log_entry['reason'] = f'continue_existing_assignment'
                else:
                    # First assignment for this PDF unit
                    # CROSS-PAGE CHECK: Don't allow matching to OpenXML units 
                    # that come BEFORE the last matched unit from previous pages
                    if openxml_idx < min_openxml_idx:
                        log_entry['action'] = 'SKIP'
                        log_entry['reason'] = f'cross_page_backward: openxml_idx={openxml_idx} < min_from_prev_page={min_openxml_idx}'
                        traversal_log.append(log_entry)
                        continue
                    
                    # WITHIN-PAGE CHECK: Ensure ordering consistency
                    # If PDF[A] < PDF[B] then OpenXML[A] <= OpenXML[B]
                    # Check against ALL existing assignments
                    if not is_shape_pdf:
                        backward_violation = False
                        violation_reason = None
                        
                        for other_pdf_idx, other_openxml_idx in pdf_unit_assignment.items():
                            # If this PDF unit comes AFTER another assigned one
                            if pdf_idx > other_pdf_idx:
                                # Then this must match to OpenXML >= that one
                                if openxml_idx < other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] > pdf[{other_pdf_idx}] but openxml[{openxml_idx}] < openxml[{other_openxml_idx}]'
                                    break
                            # If this PDF unit comes BEFORE another assigned one
                            elif pdf_idx < other_pdf_idx:
                                # Then this must match to OpenXML <= that one
                                if openxml_idx > other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] < pdf[{other_pdf_idx}] but openxml[{openxml_idx}] > openxml[{other_openxml_idx}]'
                                    break
                        
                        if backward_violation:
                            log_entry['action'] = 'SKIP'
                            log_entry['reason'] = f'backward_match_prevented: {violation_reason}'
                            traversal_log.append(log_entry)
                            continue
                        
                        pdf_unit_assignment[pdf_idx] = openxml_idx
                        log_entry['reason'] = f'new_assignment'
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
                
                # Debug info
                debug_key = (openxml_idx, pdf_idx)
                if debug_key not in match_debug:
                    match_debug[debug_key] = {'matched_chars': []}
                match_debug[debug_key]['matched_chars'].append(pdf_concat[pdf_char_idx])
                
                # Add to block log (first occurrence only)
                if len(block_log['matches']) < 5:
                    block_log['matches'].append({
                        'char': pdf_concat[pdf_char_idx],
                        'pdf_unit': pdf_idx,
                        'openxml_unit': openxml_idx
                    })
        
        if block_log['matches']:
            matching_log.append(block_log)
    
    # Build per-unit matching summary
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
    
    # Detect suspicious page numbers based on context
    suspicious_page_numbers = detect_suspicious_page_numbers(pdf_units, pdf_unit_assignment, openxml_to_pdf)
    
    # Update unit_matching_summary with suspicious flag
    for entry in unit_matching_summary:
        entry['is_suspicious_page_number'] = entry['pdf_unit_idx'] in suspicious_page_numbers
    
    # Build alignments organized by OpenXML element (exclude suspicious matches)
    # Filter openxml_to_pdf to remove suspicious page number matches
    filtered_openxml_to_pdf = {}
    for openxml_idx, pdf_counts in openxml_to_pdf.items():
        filtered_counts = {pdf_idx: count for pdf_idx, count in pdf_counts.items() 
                          if pdf_idx not in suspicious_page_numbers}
        if filtered_counts:
            filtered_openxml_to_pdf[openxml_idx] = filtered_counts
    
    alignments = build_alignments_from_matching(
        filtered_openxml_to_pdf, pdf_units, openxml_units, match_debug, 'char'
    )
    
    # Unaligned PDF units (excluding suspicious page numbers detected by context)
    unaligned_pdf_indices = [i for i in range(len(pdf_units)) 
                             if i not in pdf_unit_assignment 
                             and i not in suspicious_page_numbers]
    
    # Unaligned OpenXML units (not matched to any PDF unit)
    unaligned_openxml_indices = [i for i in range(len(openxml_units)) 
                                  if i not in filtered_openxml_to_pdf]
    
    # Page numbers detected (both pre-filtered and suspicious)
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
            {'block_num': i, 'pdf_pos': b.a, 'openxml_pos': b.b, 'size': b.size, 
             'text': pdf_concat[b.a:b.a+min(b.size, 50)]}
            for i, b in enumerate(matching_blocks) if b.size > 0
        ][:30],
        'matching_log': matching_log[:20],
        'traversal_log': traversal_log,  # Full detailed log for copy
        'traversal_log_count': len(traversal_log),
        'unit_matching_summary': unit_matching_summary,
        'consumed_pdf_count': len(pdf_unit_assignment),
        'page_number_indices': page_number_list,
        'suspicious_page_numbers': list(suspicious_page_numbers),
        'unaligned_pdf_count': len(unaligned_pdf_indices),
        'unaligned_openxml_count': len(unaligned_openxml_indices),
        'unaligned_openxml_indices': unaligned_openxml_indices,
        # Max OpenXML index assigned - used for cross-page tracking
        'max_openxml_idx': max(pdf_unit_assignment.values()) if pdf_unit_assignment else min_openxml_idx,

    }
    
    return alignments, unaligned_pdf_indices, unaligned_openxml_indices, debug_info


def build_alignments_from_matching(openxml_to_pdf, pdf_units, openxml_units, match_debug, mode):
    """
    Build alignment structure organized by OpenXML element.
    Groups table cells under parent element.
    
    For non-table paragraphs with multiple units (text + images), creates SEPARATE
    alignment entries for each unit to preserve individual bboxes.
    """
    elem_alignments = {}
    # Track non-table units separately to avoid overwriting
    # Key: (elem_id, unit_id) for non-tables to keep text/image separate
    non_table_units = {}
    
    for openxml_idx, pdf_counts in openxml_to_pdf.items():
        if not pdf_counts:
            continue
        
        openxml_unit = openxml_units[openxml_idx]
        elem_id = openxml_unit['elem_id']
        unit_id = openxml_unit['unit_id']
        
        # Build matched PDF units for this OpenXML unit
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
        
        # Sort by original order
        matched_pdf.sort(key=lambda x: x['item_idx'])
        
        is_image_part = openxml_unit.get('is_image_part', False)
        
        # For image parts: DON'T merge bboxes - each PDF unit becomes separate alignment
        # For text parts: merge bboxes as before
        if is_image_part:
            # Create separate alignment for each matched PDF unit (each image bbox stays separate)
            for mp_idx, mp in enumerate(matched_pdf):
                bbox = mp.get('bbox')
                individual_unit_id = f"{unit_id}_m{mp_idx}"
                non_table_units[individual_unit_id] = {
                    'element_id': elem_id,
                    'element_sequence': openxml_unit['elem_seq'],
                    'element_type': openxml_unit['elem_type'],
                    'is_table': False,
                    'element_text': openxml_unit['text'],
                    'matched_pdf_units': [mp],  # Only this single PDF unit
                    'merged_bbox': list(bbox) if bbox and len(bbox) >= 4 else None,
                    'cells': None,
                    'is_text_part': False,
                    'is_image_part': True,
                    'unit_id': individual_unit_id,
                    'image_index': openxml_unit.get('image_index')
                }
            continue  # Skip the rest of the loop for image parts
        
        # For non-image parts: merge bboxes as before
        merged_bbox = None
        for mp in matched_pdf:
            bbox = mp.get('bbox')
            if bbox and len(bbox) >= 4:
                if merged_bbox is None:
                    merged_bbox = list(bbox)
                else:
                    merged_bbox[0] = min(merged_bbox[0], bbox[0])
                    merged_bbox[1] = min(merged_bbox[1], bbox[1])
                    merged_bbox[2] = max(merged_bbox[2], bbox[2])
                    merged_bbox[3] = max(merged_bbox[3], bbox[3])
        
        if openxml_unit['is_cell']:
            # Table cell: add to parent element (group by elem_id)
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
            
            # Update parent bbox
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
            # Non-table element: keep each unit SEPARATE (don't merge text + image)
            # Use unit_id as key to preserve separate bboxes for text vs image
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
                # Add flags to identify text vs image parts
                'is_text_part': is_text_part,
                'is_image_part': is_image_part,
                'unit_id': unit_id  # Include for reference
            }
    
    # Sort cells within tables
    for alignment in elem_alignments.values():
        if alignment.get('cells'):
            alignment['cells'].sort(key=lambda c: (c['row'], c['col']))
    
    # Combine table alignments with non-table units
    result = list(elem_alignments.values()) + list(non_table_units.values())
    result.sort(key=lambda x: x['element_sequence'] or 0)
    
    return result


def is_bbox_inside(inner_bbox, outer_bbox, tolerance=5):
    """
    Check if inner_bbox is inside or mostly overlapping with outer_bbox.
    Returns True if the center of inner_bbox is within outer_bbox.
    """
    if not inner_bbox or not outer_bbox or len(inner_bbox) < 4 or len(outer_bbox) < 4:
        return False
    
    # Calculate center of inner bbox
    center_x = (inner_bbox[0] + inner_bbox[2]) / 2
    center_y = (inner_bbox[1] + inner_bbox[3]) / 2
    
    # Check if center is within outer bbox (with tolerance)
    return (outer_bbox[0] - tolerance <= center_x <= outer_bbox[2] + tolerance and
            outer_bbox[1] - tolerance <= center_y <= outer_bbox[3] + tolerance)


def is_bbox_fully_contained(inner_bbox, outer_bbox, tolerance=2):
    """
    Check if inner_bbox is 100% contained within outer_bbox.
    Returns True if all four corners of inner_bbox are inside outer_bbox.
    
    Args:
        inner_bbox: [x0, y0, x1, y1] - the potentially contained bbox
        outer_bbox: [x0, y0, x1, y1] - the potentially containing bbox
        tolerance: small margin for floating point comparison
    """
    if not inner_bbox or not outer_bbox or len(inner_bbox) < 4 or len(outer_bbox) < 4:
        return False
    
    # inner_bbox must be strictly smaller (not the same as outer_bbox)
    if (abs(inner_bbox[0] - outer_bbox[0]) < tolerance and 
        abs(inner_bbox[1] - outer_bbox[1]) < tolerance and
        abs(inner_bbox[2] - outer_bbox[2]) < tolerance and
        abs(inner_bbox[3] - outer_bbox[3]) < tolerance):
        return False  # Same bbox, not contained
    
    # Check if all edges of inner are within outer
    return (inner_bbox[0] >= outer_bbox[0] - tolerance and  # left edge
            inner_bbox[1] >= outer_bbox[1] - tolerance and  # top edge
            inner_bbox[2] <= outer_bbox[2] + tolerance and  # right edge
            inner_bbox[3] <= outer_bbox[3] + tolerance)     # bottom edge


def is_punctuation_only(text):
    """
    Check if text contains only punctuation characters (., :, etc.)
    """
    if not text:
        return False
    # Remove whitespace and check if remaining chars are all punctuation
    cleaned = text.strip()
    if not cleaned:
        return False
    # Common punctuation that might appear as separate alignment
    punctuation_chars = set('.:,;!?-–—…')
    return all(c in punctuation_chars for c in cleaned)


def cleanup_punctuation_alignments(alignments):
    """
    Post-process alignments to merge punctuation-only alignments into larger
    containing alignments.
    
    If an alignment only contains punctuation (. or : etc.) and its bbox is
    fully contained within another alignment's bbox, merge it into that alignment.
    
    This handles cases where punctuation gets aligned to wrong OpenXML elements.
    
    Returns:
        updated_alignments with punctuation merged into containing alignments
    """
    if not alignments or len(alignments) < 2:
        return alignments
    
    # Find punctuation-only alignments and their potential containers
    punct_alignments = []  # (index, alignment, bbox)
    container_candidates = []  # (index, alignment, bbox, area)
    
    for i, align in enumerate(alignments):
        merged_bbox = align.get('merged_bbox')
        if not merged_bbox or len(merged_bbox) < 4:
            continue
        
        # Get all text from matched_pdf_units
        all_text = ' '.join(
            u.get('text', '') for u in align.get('matched_pdf_units', [])
        )
        
        if is_punctuation_only(all_text):
            punct_alignments.append((i, align, merged_bbox))
        else:
            # Calculate area for potential container
            area = (merged_bbox[2] - merged_bbox[0]) * (merged_bbox[3] - merged_bbox[1])
            container_candidates.append((i, align, merged_bbox, area))
    
    if not punct_alignments or not container_candidates:
        return alignments
    
    # Map punctuation alignments to their containers
    punct_to_remove = set()
    
    for punct_idx, punct_align, punct_bbox in punct_alignments:
        # Find the smallest container that fully contains this punctuation bbox
        best_container = None
        best_area = float('inf')
        
        for cont_idx, cont_align, cont_bbox, cont_area in container_candidates:
            if cont_idx == punct_idx:
                continue
            
            if is_bbox_fully_contained(punct_bbox, cont_bbox):
                if cont_area < best_area:
                    best_container = (cont_idx, cont_align)
                    best_area = cont_area
        
        if best_container:
            cont_idx, cont_align = best_container
            
            # Move PDF units from punctuation alignment to container
            for pdf_unit in punct_align.get('matched_pdf_units', []):
                pdf_unit['absorbed'] = True  # Mark as absorbed
                pdf_unit['absorbed_from_punctuation'] = True
                cont_align['matched_pdf_units'].append(pdf_unit)
            
            # Re-sort container's matched_pdf_units by item_idx
            cont_align['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))
            
            # Update container's merged_bbox to include punctuation bbox
            cont_bbox = cont_align['merged_bbox']
            cont_bbox[0] = min(cont_bbox[0], punct_bbox[0])
            cont_bbox[1] = min(cont_bbox[1], punct_bbox[1])
            cont_bbox[2] = max(cont_bbox[2], punct_bbox[2])
            cont_bbox[3] = max(cont_bbox[3], punct_bbox[3])
            
            punct_to_remove.add(punct_idx)
            
            print(f"[PunctCleanup] Merged punctuation alignment (elem_id={punct_align.get('elem_id')}) "
                  f"into container (elem_id={cont_align.get('elem_id')})")
    
    # Remove absorbed punctuation alignments
    if punct_to_remove:
        alignments = [a for i, a in enumerate(alignments) if i not in punct_to_remove]
    
    return alignments



def absorb_unaligned_into_alignments(alignments, unaligned_pdf_indices, pdf_units):
    """
    Post-process alignments to absorb unaligned PDF units that fall within
    an alignment's merged_bbox. This ensures that PDF units spatially within
    an aligned element's bbox are included, maintaining correct order by item_idx.
    
    Returns:
        (updated_alignments, remaining_unaligned_indices)
    """
    if not alignments or not unaligned_pdf_indices:
        return alignments, unaligned_pdf_indices
    
    absorbed_indices = set()
    
    for alignment in alignments:
        merged_bbox = alignment.get('merged_bbox')
        if not merged_bbox:
            continue
        
        if alignment.get('is_table') and alignment.get('cells'):
            # For tables, absorb into individual cells
            for cell in alignment['cells']:
                cell_bbox = cell.get('merged_bbox')
                if not cell_bbox:
                    continue
                
                # Find unaligned units that fall within this cell's bbox
                units_to_absorb = []
                for idx in unaligned_pdf_indices:
                    if idx in absorbed_indices:
                        continue
                    pdf_unit = pdf_units[idx]
                    unit_bbox = pdf_unit.get('bbox')
                    if is_bbox_inside(unit_bbox, cell_bbox):
                        units_to_absorb.append((idx, pdf_unit))
                        absorbed_indices.add(idx)
                
                if units_to_absorb:
                    # Add to matched_pdf_units with score 0 (absorbed)
                    for idx, pdf_unit in units_to_absorb:
                        cell['matched_pdf_units'].append({
                            'pdf_unit_id': pdf_unit['unit_id'],
                            'item_idx': pdf_unit['item_idx'],
                            'item_type': pdf_unit['item_type'],
                            'text': pdf_unit['text'],
                            'bbox': pdf_unit['bbox'],
                            'matched_count': 0,
                            'score': 0,  # Absorbed, not matched
                            'is_cell': pdf_unit['is_cell'],
                            'is_hline_table_unit': pdf_unit.get('is_hline_table_unit', False),
                            'row': pdf_unit.get('row'),
                            'col': pdf_unit.get('col'),
                            'absorbed': True,  # Flag to indicate this was absorbed
                            'debug': {}
                        })
                    
                    # Re-sort by item_idx to maintain correct order
                    cell['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
                    
                    # Update cell's merged_bbox
                    for idx, pdf_unit in units_to_absorb:
                        unit_bbox = pdf_unit.get('bbox')
                        if unit_bbox and len(unit_bbox) >= 4:
                            cell_bbox[0] = min(cell_bbox[0], unit_bbox[0])
                            cell_bbox[1] = min(cell_bbox[1], unit_bbox[1])
                            cell_bbox[2] = max(cell_bbox[2], unit_bbox[2])
                            cell_bbox[3] = max(cell_bbox[3], unit_bbox[3])
        else:
            # For non-table elements
            units_to_absorb = []
            for idx in unaligned_pdf_indices:
                if idx in absorbed_indices:
                    continue
                pdf_unit = pdf_units[idx]
                unit_bbox = pdf_unit.get('bbox')
                if is_bbox_inside(unit_bbox, merged_bbox):
                    units_to_absorb.append((idx, pdf_unit))
                    absorbed_indices.add(idx)
            
            if units_to_absorb:
                # Add to matched_pdf_units
                for idx, pdf_unit in units_to_absorb:
                    alignment['matched_pdf_units'].append({
                        'pdf_unit_id': pdf_unit['unit_id'],
                        'item_idx': pdf_unit['item_idx'],
                        'item_type': pdf_unit['item_type'],
                        'text': pdf_unit['text'],
                        'bbox': pdf_unit['bbox'],
                        'matched_count': 0,
                        'score': 0,  # Absorbed, not matched
                        'is_cell': pdf_unit['is_cell'],
                        'is_hline_table_unit': pdf_unit.get('is_hline_table_unit', False),
                        'row': pdf_unit.get('row'),
                        'col': pdf_unit.get('col'),
                        'absorbed': True,  # Flag to indicate this was absorbed
                        'debug': {}
                    })
                
                # Re-sort by item_idx to maintain correct order
                alignment['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
                
                # Update merged_bbox
                for idx, pdf_unit in units_to_absorb:
                    unit_bbox = pdf_unit.get('bbox')
                    if unit_bbox and len(unit_bbox) >= 4:
                        merged_bbox[0] = min(merged_bbox[0], unit_bbox[0])
                        merged_bbox[1] = min(merged_bbox[1], unit_bbox[1])
                        merged_bbox[2] = max(merged_bbox[2], unit_bbox[2])
                        merged_bbox[3] = max(merged_bbox[3], unit_bbox[3])
    
    # Calculate remaining unaligned
    remaining_unaligned = [idx for idx in unaligned_pdf_indices if idx not in absorbed_indices]
    
    print(f"[Absorb] Absorbed {len(absorbed_indices)} unaligned PDF units into alignments, "
          f"{len(remaining_unaligned)} remaining unaligned")
    
    return alignments, remaining_unaligned


def get_alignment_min_item_idx(alignment):
    """Get the smallest PDF item_idx used by an alignment (including table cells)."""
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


def get_alignment_sequence(alignment):
    """Normalize element_sequence for comparison."""
    seq = alignment.get('element_sequence')
    if seq is None:
        return 0
    try:
        return int(seq)
    except (TypeError, ValueError):
        return 0


def attach_shape_clusters_to_next_alignment(alignments, unaligned_pdf_indices, pdf_units):
    """
    Attach unaligned shape clusters to the element immediately before the next
    alignment (by OpenXML sequence). Shapes are merged into a single unit.
    """
    if not alignments or not unaligned_pdf_indices:
        return alignments, unaligned_pdf_indices, []

    shape_indices = [
        idx for idx in unaligned_pdf_indices
        if pdf_units[idx].get('item_type') == 'shape'
    ]
    if not shape_indices:
        return alignments, unaligned_pdf_indices, []

    alignment_positions = []
    alignments_by_sequence = sorted(alignments, key=get_alignment_sequence)
    for alignment in alignments:
        min_idx = get_alignment_min_item_idx(alignment)
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
        cluster_bbox = merge_bboxes([u.get('bbox') for u in cluster_units])
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
            next_seq = get_alignment_sequence(next_alignment)
            prev_candidates = [a for a in alignments_by_sequence if get_alignment_sequence(a) < next_seq]
            if prev_candidates:
                target_alignment = max(prev_candidates, key=get_alignment_sequence)
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
            # Create a separate non-table alignment so shapes are visible in UI.
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
        print(f"[ShapeAttach] Attached {attached_count} shape clusters to previous alignment")

    return alignments, remaining_unaligned, debug


def recompute_alignment_bboxes(alignment):
    """Recompute merged_bbox for an alignment after unit removal."""
    if alignment.get('is_table') and alignment.get('cells'):
        cell_bboxes = []
        for cell in alignment['cells']:
            cell_units = cell.get('matched_pdf_units', [])
            cell_bbox = merge_bboxes([u.get('bbox') for u in cell_units])
            cell['merged_bbox'] = cell_bbox
            if cell_bbox:
                cell_bboxes.append(cell_bbox)
        alignment['merged_bbox'] = merge_bboxes(cell_bboxes)
    else:
        units = alignment.get('matched_pdf_units', [])
        alignment['merged_bbox'] = merge_bboxes([u.get('bbox') for u in units])


def resolve_shape_alignment_conflicts(alignments, pdf_units):
    """
    Reassign shape pdf units to the element immediately before the next
    alignment in PDF order (based on OpenXML sequence).
    """
    if not alignments:
        return alignments, []

    pdf_unit_by_id = {u.get('unit_id'): u for u in pdf_units if u.get('unit_id')}
    alignment_positions = []
    alignments_by_sequence = sorted(alignments, key=get_alignment_sequence)
    for alignment in alignments:
        min_idx = get_alignment_min_item_idx(alignment)
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
                    target_seq = get_alignment_sequence(alignment)
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
            prior_candidates = [a for a in candidate_alignments if get_alignment_sequence(a) < target_seq]
            if prior_candidates:
                chosen_alignment = max(prior_candidates, key=get_alignment_sequence)
            else:
                chosen_alignment = min(candidate_alignments, key=lambda a: abs(get_alignment_sequence(a) - target_seq))
        else:
            chosen_alignment = max(candidate_alignments, key=get_alignment_sequence)

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

                removed_from.append(get_alignment_sequence(alignment))

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
                recompute_alignment_bboxes(alignment)
        print(f"[ShapeResolve] Resolved {len(debug)} shape conflicts")

    return alignments, debug


def match_remaining_with_unaligned_openxml(alignments, remaining_pdf_indices, unaligned_openxml_indices, 
                                            pdf_units, openxml_units):
    """
    Try to match remaining unaligned PDF units with unaligned OpenXML elements
    using the SAME character-based alignment as Phase 1 (perform_char_alignment).
    
    This re-runs the alignment algorithm on just the remaining unaligned items.
    
    Returns:
        (updated_alignments, remaining_pdf_indices, remaining_openxml_indices)
    """
    if not remaining_pdf_indices or not unaligned_openxml_indices:
        return alignments, remaining_pdf_indices, unaligned_openxml_indices
    
    # Build subset units from remaining indices
    remaining_pdf_units = [pdf_units[i] for i in remaining_pdf_indices]
    unaligned_openxml_units = [openxml_units[i] for i in unaligned_openxml_indices]
    
    print(f"[LateMtch] Re-running char alignment on {len(remaining_pdf_units)} PDF units "
          f"and {len(unaligned_openxml_units)} OpenXML units")
    
    # Run the same char alignment on remaining items
    # Note: perform_char_alignment already returns ready-to-use alignments via build_alignments_from_matching
    late_alignments, late_unaligned_pdf, late_unaligned_openxml, late_debug = perform_char_alignment(
        remaining_pdf_units, unaligned_openxml_units
    )
    
    # late_alignments is already in the right format, just use it directly
    new_alignments = late_alignments
    
    # Mark as late matched
    for alignment in new_alignments:
        alignment['late_matched'] = True
        for unit in alignment.get('matched_pdf_units', []):
            unit['late_matched'] = True
    
    if new_alignments:
        # Merge new alignments into existing ones
        existing_elem_ids = {a['element_id'] for a in alignments}
        
        for new_align in new_alignments:
            elem_id = new_align['element_id']
            
            if elem_id in existing_elem_ids:
                # Merge into existing alignment
                for existing in alignments:
                    if existing['element_id'] == elem_id:
                        existing['matched_pdf_units'].extend(new_align['matched_pdf_units'])
                        existing['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
                        
                        if new_align['merged_bbox']:
                            if existing['merged_bbox']:
                                existing['merged_bbox'][0] = min(existing['merged_bbox'][0], new_align['merged_bbox'][0])
                                existing['merged_bbox'][1] = min(existing['merged_bbox'][1], new_align['merged_bbox'][1])
                                existing['merged_bbox'][2] = max(existing['merged_bbox'][2], new_align['merged_bbox'][2])
                                existing['merged_bbox'][3] = max(existing['merged_bbox'][3], new_align['merged_bbox'][3])
                            else:
                                existing['merged_bbox'] = new_align['merged_bbox']
                        break
            else:
                # Add as new alignment
                alignments.append(new_align)
        
        alignments.sort(key=lambda x: x['element_sequence'] or 0)
    
    # Map late_unaligned indices back to original indices
    final_remaining_pdf = [remaining_pdf_indices[i] for i in late_unaligned_pdf]
    final_remaining_openxml = [unaligned_openxml_indices[i] for i in late_unaligned_openxml]
    
    matched_pdf_count = len(remaining_pdf_indices) - len(final_remaining_pdf)
    matched_openxml_count = len(unaligned_openxml_indices) - len(final_remaining_openxml)
    
    print(f"[LateMtch] Created {len(new_alignments)} late alignments, "
          f"matched {matched_pdf_count} PDF units and {matched_openxml_count} OpenXML units, "
          f"{len(final_remaining_pdf)} PDF and {len(final_remaining_openxml)} OpenXML still unaligned")
    
    return alignments, final_remaining_pdf, final_remaining_openxml

def perform_two_pass_alignment(pdf_units, openxml_units, min_openxml_idx=0):
    """
    Two-pass alignment strategy:
    Pass 1: Align ALL PDF units with NON-SHAPE OpenXML elements (sequential char-based)
    Pass 2: Align SHAPE OpenXML elements with remaining unaligned PDF units (proximity-based)
    
    Args:
        pdf_units: List of PDF units to align
        openxml_units: List of OpenXML units to align against
        min_openxml_idx: Minimum OpenXML index to match (for cross-page tracking)
    
    Returns dict with phase1_alignments, final_alignments, etc.
    """
    # No filtering - all OpenXML units participate in Phase 1 alignment
    # (Shape filtering disabled as it caused tables with shapes to be excluded)
    
    print(f"[TwoPass] OpenXML: {len(openxml_units)} total units (no shape filtering)")
    
    # =============================
    # PASS 1: Align ALL PDF units with ALL OpenXML units
    # =============================
    if openxml_units:
        pass1_alignments, pass1_unaligned_pdf, pass1_unaligned_openxml, pass1_debug = perform_char_alignment(
            pdf_units, openxml_units, min_openxml_idx
        )
    else:
        pass1_alignments = []
        pass1_unaligned_pdf = list(range(len(pdf_units)))
        pass1_unaligned_openxml = []
        pass1_debug = {}
    
    # Build mapping: elem_id -> merged_bbox from pass 1
    aligned_elem_to_bbox = {}
    aligned_elem_to_seq = {}
    for alignment in pass1_alignments:
        elem_id = alignment['element_id']
        aligned_elem_to_bbox[elem_id] = alignment.get('merged_bbox')
        aligned_elem_to_seq[elem_id] = alignment.get('element_sequence')
    
    print(f"[TwoPass] Pass 1 done: {len(pass1_alignments)} aligned, "
          f"{len(pass1_unaligned_pdf)} PDF units unaligned")

    
    # =============================
    # PASS 2: Disabled (shape filtering removed)
    # =============================
    # All units are now processed in Pass 1, so no Pass 2 needed
    shape_alignments = []
    pass2_consumed_pdf = set()
    pass2_debug = []
    
    print(f"[TwoPass] Pass 2 disabled - all units processed in Pass 1")
    
    # Combine pass 1 and pass 2 alignments for final
    final_alignments = pass1_alignments + shape_alignments
    final_alignments.sort(key=lambda x: x['element_sequence'] or 0)
    
    # Update unaligned PDF indices
    final_unaligned = [i for i in pass1_unaligned_pdf if i not in pass2_consumed_pdf]
    
    # =============================
    # POST-PROCESS 1: Absorb unaligned PDF units into alignments
    # =============================
    # If an unaligned PDF unit's bbox falls within an alignment's merged_bbox,
    # absorb it into that alignment (maintaining correct order by item_idx)
    final_alignments, final_unaligned = absorb_unaligned_into_alignments(
        final_alignments, final_unaligned, pdf_units
    )
    
    # =============================
    # POST-PROCESS 2: Match remaining unaligned with unaligned OpenXML
    # =============================
    # Try to match remaining PDF units with OpenXML elements that have no bbox
    # using the same char-based alignment algorithm
    final_alignments, final_unaligned, pass1_unaligned_openxml = match_remaining_with_unaligned_openxml(
        final_alignments, final_unaligned, pass1_unaligned_openxml, 
        pdf_units, openxml_units
    )
    
    # =============================
    # POST-PROCESS 3: Cleanup punctuation alignments
    # =============================
    # If an alignment only contains punctuation (. or :) and its bbox is fully
    # contained within another alignment's bbox, merge it into that alignment
    final_alignments = cleanup_punctuation_alignments(final_alignments)

    # =============================
    # POST-PROCESS 4: Resolve shape units matched to multiple elements
    # =============================
    final_alignments, shape_conflict_debug = resolve_shape_alignment_conflicts(
        final_alignments, pdf_units
    )

    # =============================
    # POST-PROCESS 5: Attach unaligned shapes to next element
    # =============================
    final_alignments, final_unaligned, shape_attach_debug = attach_shape_clusters_to_next_alignment(
        final_alignments, final_unaligned, pdf_units
    )

    
    # Build unaligned PDF units for phase 1
    unaligned_after_phase1 = [
        {
            'pdf_unit_id': pdf_units[i]['unit_id'],
            'item_idx': pdf_units[i]['item_idx'],
            'item_type': pdf_units[i]['item_type'],
            'text': pdf_units[i]['text'],
            'bbox': pdf_units[i].get('bbox')
        }
        for i in pass1_unaligned_pdf
    ]
    
    # Combine debug info
    pass1_debug['pass2_shape_debug'] = pass2_debug
    pass1_debug['pass2_shape_matched'] = len(shape_alignments)
    pass1_debug['pass2_consumed_pdf'] = list(pass2_consumed_pdf)
    pass1_debug['shape_openxml_count'] = 0  # Shape filtering disabled
    pass1_debug['non_shape_openxml_count'] = len(openxml_units)
    pass1_debug['shape_conflict_debug'] = shape_conflict_debug
    pass1_debug['shape_conflict_count'] = len(shape_conflict_debug)
    pass1_debug['shape_attach_debug'] = shape_attach_debug
    pass1_debug['shape_attach_count'] = len(shape_attach_debug)
    
    # Filter unaligned OpenXML to only those within this page's sequence range
    # (elements between min and max sequence of aligned elements on this page)
    page_unaligned_openxml = []
    if pass1_alignments and pass1_unaligned_openxml:
        # Get sequence range from aligned elements
        aligned_sequences = [a.get('element_sequence') or 0 for a in pass1_alignments]
        min_seq = min(aligned_sequences)
        max_seq = max(aligned_sequences)
        
        # Filter to only unaligned units within this range
        for idx in pass1_unaligned_openxml:
            unit = openxml_units[idx]
            unit_seq = unit.get('elem_seq') or 0
            if min_seq <= unit_seq <= max_seq:
                page_unaligned_openxml.append(idx)
        
        print(f"[TwoPass] Unaligned OpenXML: {len(pass1_unaligned_openxml)} total, "
              f"{len(page_unaligned_openxml)} in page range (seq {min_seq}-{max_seq})")
    else:
        page_unaligned_openxml = pass1_unaligned_openxml
    
    # Return dict with both phases
    return {
        'phase1_alignments': pass1_alignments,
        'final_alignments': final_alignments,
        'shape_alignments': shape_alignments,
        'unaligned_after_phase1': pass1_unaligned_pdf,
        'unaligned_openxml': page_unaligned_openxml,  # Only OpenXML units in this page's range
        'unaligned_final': final_unaligned,
        'debug_info': pass1_debug,
        'openxml_units': openxml_units  # Pass through for building response
    }




def register_merging_alignment_routes(bp):
    """Register the merging alignment API route on the given blueprint"""
    
    @bp.route('/dokumen-elemen-api/merging-alignment/<int:doc_id>/<int:page>', methods=['POST'])

    def api_merging_alignment(doc_id, page):
        """
        Alignment API using merging extraction data.
        
        Request body:
        {
            "extraction_items": [{type, data, bbox}, ...]
        }
        
        Response:
        {
            "success": true,
            "alignments": [...],           // Character-based alignments
            "unaligned_pdf_units": [...],  // PDF units not matched
            'page_debug': {...}
        }
        """
        data = request.get_json() or {}
        extraction_items = data.get('extraction_items', [])
        
        if not extraction_items:
            return jsonify({
                'success': True,
                'page': page,
                'timestamp': datetime.now().isoformat(),
                'alignments': [],
                'unaligned_pdf_units': [],
                'header_footer_units': [],
                'page_debug': {}
            })
        
        # Log extraction_items received from frontend
        print(f"[Alignment] Received {len(extraction_items)} extraction_items:")
        for idx, item in enumerate(extraction_items):
            item_type = item.get('type', 'UNKNOWN')
            item_data = item.get('data', {})
            text_preview = ''
            if item_type == 'group':
                text_preview = item_data.get('text', '')[:50]
            elif item_type in ['table', 'hline_table']:
                text_preview = f"rows={len(item_data.get('rows', []))}"
            elif item_type == 'shape':
                text_preview = item_data.get('text', '')[:50]
            print(f"  [{idx}] {item_type}: {text_preview}")
        
        # Flatten extraction items to PDF units
        all_pdf_units = flatten_extraction_items(extraction_items)
        
        # Get page dimensions from extraction data (sent from frontend)
        page_width = data.get('page_width', 0)
        page_height = data.get('page_height', 0)
        
        # Get min_openxml_idx for cross-page backward matching prevention
        # This should be the max_openxml_idx from the previous page
        min_openxml_idx = data.get('min_openxml_idx', 0)
        
        # Get all sections for this document
        sections = DokumenSection.query.filter_by(dokumen_id=doc_id)\
            .order_by(DokumenSection.dsec_index).all()
        
        # Find the section that matches current page dimensions
        section_data = None
        if sections and page_width and page_height:
            twips_per_point = 20
            for sec in sections:
                sec_width = (sec.dsec_page_width_twips or 0) / twips_per_point
                sec_height = (sec.dsec_page_height_twips or 0) / twips_per_point
                # Match with tolerance of 10 points
                if abs(sec_width - page_width) < 10 and abs(sec_height - page_height) < 10:
                    section_data = sec
                    print(f"[Alignment] Matched section: orientation={sec.dsec_orientation}, "
                          f"width={sec_width:.1f}pt, height={sec_height:.1f}pt")
                    break
        
        # Fallback to first section if no match
        if not section_data and sections:
            section_data = sections[0]
            print(f"[Alignment] Using fallback section (first): orientation={section_data.dsec_orientation}")
        
        # Filter out header/footer items
        pdf_units, header_footer_units = filter_header_footer_items(all_pdf_units, section_data)
        
        # Log filtering result
        print(f"[Alignment] doc_id={doc_id}, page={page}: {len(all_pdf_units)} total units, "
              f"{len(header_footer_units)} in header/footer, {len(pdf_units)} for alignment")
        
        # Get OpenXML elements (body parts only)
        from models import DokumenPart, DokumenElemen
        
        elements = db.session.query(DokumenElemen)\
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)\
            .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)\
            .filter(DokumenSection.dokumen_id == doc_id)\
            .filter(DokumenPart.dpart_type == 'body')\
            .order_by(DokumenElemen.delemen_sequence)\
            .all()
        
        # Debug: Log elements from query
        elem_ids_from_query = [e.delemen_id for e in elements]
        print(f"[Alignment] Query returned {len(elements)} elements: {elem_ids_from_query[:5]}...{elem_ids_from_query[-5:] if len(elem_ids_from_query) > 5 else ''}")
        
        # Estimate page sequence range for proper image numbering
        # This ensures OpenXML images are numbered starting from 1 for current page,
        # matching the per-page numbering used by PDF extraction
        total_elements = len(elements)
        total_pages = data.get('total_pages', 1)
        if total_pages < 1:
            total_pages = 1
        
        # Estimate elements per page and calculate range for current page
        elements_per_page = max(1, total_elements // total_pages)
        # Add buffer to ensure we capture all relevant elements
        buffer = max(10, elements_per_page // 2)
        
        # Get sequence numbers from elements
        page_sequence_range = None
        if elements:
            all_sequences = sorted([e.delemen_sequence for e in elements])
            if len(all_sequences) > 0:
                estimated_start_idx = max(0, min((page - 1) * elements_per_page - buffer, len(all_sequences) - 1))
                estimated_end_idx = max(0, min(page * elements_per_page + buffer, len(all_sequences) - 1))
                
                # Ensure start <= end
                if estimated_start_idx > estimated_end_idx:
                    estimated_start_idx, estimated_end_idx = estimated_end_idx, estimated_start_idx
                
                page_sequence_range = (all_sequences[estimated_start_idx], all_sequences[estimated_end_idx])
                print(f"[Alignment] Estimated page {page} sequence range: {page_sequence_range} (total_pages={total_pages})")
        
        # Build OpenXML units with page sequence range for proper image numbering
        openxml_units, table_debug = build_openxml_units(elements, page_sequence_range)
        
        # Debug: Log openxml_units count and first few IDs
        unit_ids = [u['elem_id'] for u in openxml_units]
        unique_elem_ids = list(set(unit_ids))
        print(f"[Alignment] Built {len(openxml_units)} OpenXML units from {len(unique_elem_ids)} unique elements: {unique_elem_ids[:5]}...")

        
        # Two-pass alignment: non-shape elements first, then shapes by proximity
        alignment_result = perform_two_pass_alignment(pdf_units, openxml_units, min_openxml_idx)
        
        # Unpack results
        phase1_alignments = alignment_result['phase1_alignments']
        final_alignments = alignment_result['final_alignments']
        shape_alignments = alignment_result['shape_alignments']
        unaligned_after_phase1 = alignment_result['unaligned_after_phase1']
        unaligned_openxml_indices = alignment_result['unaligned_openxml']
        unaligned_final = alignment_result['unaligned_final']
        char_debug = alignment_result['debug_info']

        
        # Build unaligned PDF units response for final
        unaligned_pdf_units = [
            {
                'pdf_unit_id': pdf_units[i]['unit_id'],
                'item_idx': pdf_units[i]['item_idx'],
                'item_type': pdf_units[i]['item_type'],
                'text': pdf_units[i]['text'],
                'bbox': pdf_units[i]['bbox'],
                'is_cell': pdf_units[i]['is_cell'],
                'row': pdf_units[i].get('row'),
                'col': pdf_units[i].get('col')
            }
            for i in unaligned_final
        ]
        
        # Build unaligned PDF units for phase 1
        unaligned_phase1_pdf_units = [
            {
                'pdf_unit_id': pdf_units[i]['unit_id'],
                'item_idx': pdf_units[i]['item_idx'],
                'item_type': pdf_units[i]['item_type'],
                'text': pdf_units[i]['text'],
                'bbox': pdf_units[i]['bbox'],
                'is_cell': pdf_units[i]['is_cell'],
                'row': pdf_units[i].get('row'),
                'col': pdf_units[i].get('col')
            }
            for i in unaligned_after_phase1
        ]
        
        # Build unaligned OpenXML units (elements that couldn't find matching bbox)
        unaligned_openxml_units_response = [
            {
                'openxml_unit_id': openxml_units[i]['unit_id'],
                'elem_id': openxml_units[i]['elem_id'],
                'elem_type': openxml_units[i]['elem_type'],
                'text': openxml_units[i]['text'],
                'text_normalized': openxml_units[i]['text_normalized'],
                'is_cell': openxml_units[i]['is_cell'],
                'row': openxml_units[i].get('row'),
                'col': openxml_units[i].get('col'),
                'has_shape': openxml_units[i].get('has_shape', False)
            }
            for i in unaligned_openxml_indices
        ]
        
        # Build debug info
        pdf_units_debug = [
            {
                'unit_id': u['unit_id'],
                'item_type': u['item_type'],
                'text': u['text'][:100],
                'text_normalized': u['text_normalized'][:100],
                'is_cell': u['is_cell'],
                'row': u.get('row'),
                'col': u.get('col')
            }
            for u in pdf_units
        ]
        
        openxml_units_debug = [
            {
                'unit_id': u['unit_id'],
                'elem_type': u['elem_type'],
                'text': u['text'][:100],
                'text_normalized': u['text_normalized'][:100],
                'is_cell': u['is_cell'],
                'row': u.get('row'),
                'col': u.get('col'),
                'has_shape': u.get('has_shape', False)
            }
            for u in openxml_units
        ]
        
        # Debug: extraction items received (to trace what was sent by frontend)
        extraction_items_debug = []
        for idx, item in enumerate(extraction_items):
            debug_entry = {
                'idx': idx,
                'type': item.get('type', ''),
                'has_data': 'data' in item,
                'has_bbox': 'bbox' in item,
            }
            
            item_data = item.get('data', {})
            item_type = item.get('type', '')
            
            if item_type == 'group':
                debug_entry['text_preview'] = str(item_data.get('text', ''))[:50]
            elif item_type in ('table', 'hline_table'):
                rows = item_data.get('rows', [])
                debug_entry['rows_count'] = len(rows)
                debug_entry['cells_breakdown'] = []
                for row_idx, row in enumerate(rows):
                    cells = row.get('cells', [])
                    cells_with_content = [c for c in cells if c.get('content')]
                    cells_with_text = []
                    for c in cells:
                        cell_text = extract_cell_content_text(c)
                        if cell_text.strip():
                            cells_with_text.append(cell_text[:30])
                    debug_entry['cells_breakdown'].append({
                        'row': row_idx,
                        'cells': len(cells),
                        'cells_with_content': len(cells_with_content),
                        'texts': cells_with_text[:3]  # First 3 texts
                    })
            elif item_type == 'shape':
                debug_entry['text_preview'] = str(item_data.get('text', ''))[:50]
            
            extraction_items_debug.append(debug_entry)
        
        page_debug = {
            'extraction_items_received': len(extraction_items),
            'extraction_items_summary': extraction_items_debug,
            'total_pdf_units_before_filter': len(all_pdf_units),
            'header_footer_filtered_count': len(header_footer_units),
            'pdf_units_for_alignment': len(pdf_units),
            'openxml_units_count': len(openxml_units),
            'pdf_units': pdf_units_debug,
            'openxml_units': openxml_units_debug,
            'char_based': char_debug,
            'table_processing': table_debug  # Debug info for table element processing
        }
        
        # Build header/footer units response
        header_footer_response = [
            {
                'pdf_unit_id': u['unit_id'],
                'item_idx': u['item_idx'],
                'item_type': u['item_type'],
                'text': u['text'],
                'bbox': u['bbox'],
                'zone': u.get('zone', 'unknown')
            }
            for u in header_footer_units
        ]
        
        return jsonify({
            'success': True,
            'page': page,
            'timestamp': datetime.now().isoformat(),
            'total_pdf_units': len(pdf_units),
            'total_openxml_units': len(openxml_units),
            # Phase 1 results (non-shape alignment only)
            'phase1_alignments': phase1_alignments,
            'unaligned_phase1': unaligned_phase1_pdf_units,
            # Final results (with shape alignment)
            'alignments': final_alignments,  # Keep 'alignments' for backward compat
            'final_alignments': final_alignments,
            'shape_alignments': shape_alignments,
            'unaligned_pdf_units': unaligned_pdf_units,
            'unaligned_openxml_units': unaligned_openxml_units_response,  # OpenXML elements without matching bbox
            'header_footer_units': header_footer_response,
            'page_debug': page_debug,
            # For cross-page tracking - frontend should pass this as min_openxml_idx for next page
            'max_openxml_idx': char_debug.get('max_openxml_idx', 0)
        })

