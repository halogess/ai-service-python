"""difflib_alignment.py - Global difflib-based alignment between OpenXML and PDF"""

import difflib
import fitz
import re
import unicodedata
from collections import defaultdict


# Token regex (dipakai untuk tokenisasi, tapi juga butuh posisi via finditer)
# PENTING: Pisahkan footnote (huruf besar + angka 1-2 digit di akhir), tapi gabung subscript (huruf kecil + angka)
# PENTING: Include underscore (_) di tengah identifier
TOKEN_RE = re.compile(r"\d*[A-Z\u00c0-\u00df\u0391-\u03a9][A-Za-z\u00c0-\u00ff\u0370-\u03ff_]*(?=\d{1,2}(?!\d))|\d*[A-Za-z\u00c0-\u00ff\u0370-\u03ff_]+\d*|\d+(?:\.\d+)*|[^\w\s]", flags=re.UNICODE)


def _split_token_with_trailing_digits(token: str) -> list:
    """Split token HANYA untuk huruf + 2 digit.
    
    Examples:
        "predikat24" -> ["predikat", "2", "4"]
        "Z24" -> ["Z", "2", "4"]
        "roa1" -> ["roa1"]  (keep subscript)
        "24" -> ["24"]  (keep pure numbers)
        "247.44" -> ["247.44"]  (keep decimals)
    """
    # HANYA split: huruf (1+) + exactly 2 digits
    match = re.match(r'^([a-zA-Zα-ωΑ-Ω]+)(\d{2})$', token, re.UNICODE)
    if match:
        return [match.group(1)] + list(match.group(2))
    
    return [token]


def iter_pdf_tokens_with_bboxes(page: "fitz.Page", page_index: int):
    """Yield (token, bbox, page_index) dengan bbox *per token*.

    Kenapa perlu ini?
    - `page.get_text('words')` mengembalikan bbox per *word*.
      Di code lama, satu word bisa di-split menjadi beberapa token, tapi semua token
      diberi bbox yang sama. Akibatnya, karakter tambahan dalam word yang tidak
      seharusnya ter-align (mis. footnote mark, bullet, numbering yang nempel,
      atau suffix/prefix lain) ikut kebawa ketika bbox di-union.

    Dengan `rawdict` (char-level), kita bisa hitung bbox yang lebih ketat untuk
    setiap token berdasarkan bbox per karakter.
    """

    raw = page.get_text("rawdict")
    blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
    # Pastikan urutan baca stabil (atas->bawah, kiri->kanan)
    blocks.sort(key=lambda b: (b.get("bbox", [0, 0])[1], b.get("bbox", [0, 0])[0]))

    for b in blocks:
        for line in b.get("lines", []):
            chars = []
            char_boxes = []
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    chars.append(ch.get("c", ""))
                    char_boxes.append(ch.get("bbox"))

            if not chars:
                continue

            line_text = "".join(chars)
            
            # NORMALISASI dengan preserve whitespace untuk bbox mapping
            line_text_normalized = normalize_text(line_text, preserve_whitespace=True)
            
            for m in TOKEN_RE.finditer(line_text_normalized):
                token = m.group(0)
                if not token or not token.strip():
                    continue

                start, end = m.span()
                # Ambil bbox langsung dari positions (1:1 mapping karena whitespace preserved)
                token_boxes = []
                for idx in range(start, end):
                    if idx < len(char_boxes) and char_boxes[idx]:
                        token_boxes.append(char_boxes[idx])
                
                if not token_boxes:
                    continue

                # POST-PROCESS: DISABLED - jangan split PDF
                # split_tokens = _split_token_with_trailing_digits(token)
                split_tokens = [token]
                
                if len(split_tokens) > 1:
                    # Token di-split, hitung bbox per sub-token
                    char_idx = 0
                    for sub_token in split_tokens:
                        sub_len = len(sub_token)
                        sub_boxes = token_boxes[char_idx:char_idx + sub_len]
                        
                        if sub_boxes:
                            x0 = min(bb[0] for bb in sub_boxes)
                            y0 = min(bb[1] for bb in sub_boxes)
                            x1 = max(bb[2] for bb in sub_boxes)
                            y1 = max(bb[3] for bb in sub_boxes)
                            yield sub_token, [x0, y0, x1, y1], page_index
                        
                        char_idx += sub_len
                else:
                    # Token tidak di-split
                    x0 = min(bb[0] for bb in token_boxes)
                    y0 = min(bb[1] for bb in token_boxes)
                    x1 = max(bb[2] for bb in token_boxes)
                    y1 = max(bb[3] for bb in token_boxes)
                    yield token, [x0, y0, x1, y1], page_index


def merge_bboxes_token_level(items, x_gap: float = 2.0, y_overlap_min: float = 0.5, is_formula: bool = False):
    """Merge bbox token-level jadi beberapa segmen (tidak dipaksa union 1 kotak besar).

    Ini penting kalau kamu ingin **karakter/word yang tidak ter-align** tidak ikut
    "ketutup" hanya karena berada di antara dua token yang align.

    Args:
        items: list of dict {bbox: {x0,y0,x1,y1}, page: int, ...}
        x_gap: jarak horizontal maksimal untuk dianggap masih satu segmen
        y_overlap_min: minimal overlap ratio vertikal untuk dianggap satu baris
        is_formula: jika True, filter token dari baris lain

    Returns:
        list of {page, bbox}
    """

    def overlap_ratio(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        denom = min(a1 - a0, b1 - b0) if min(a1 - a0, b1 - b0) > 0 else 1.0
        return inter / denom

    if not items:
        return []

    # FILTER: Remove items yang Y-nya terlalu jauh dari median (HANYA untuk formula)
    if is_formula and len(items) > 1:
        y_mids = [(item['bbox']['y0'] + item['bbox']['y1']) / 2 for item in items]
        y_median = sorted(y_mids)[len(y_mids) // 2]
        y_threshold = 15  # Max 15px dari median
        items = [item for item in items if abs((item['bbox']['y0'] + item['bbox']['y1']) / 2 - y_median) <= y_threshold]
    
    if not items:
        return []

    # Sort stabil
    items = sorted(
        items,
        key=lambda w: (
            w.get("page", 0),
            w["bbox"]["y0"],
            w["bbox"]["x0"],
        ),
    )

    merged = []
    cur = None

    for w in items:
        p = w.get("page", 0)
        b = w["bbox"]
        if cur is None:
            cur = {"page": p, "bbox": dict(b)}
            continue

        cb = cur["bbox"]
        same_page = p == cur["page"]
        y_ok = overlap_ratio(cb["y0"], cb["y1"], b["y0"], b["y1"]) >= y_overlap_min
        x_ok = b["x0"] <= cb["x1"] + x_gap

        if same_page and y_ok and x_ok:
            # merge
            cb["x0"] = min(cb["x0"], b["x0"])
            cb["y0"] = min(cb["y0"], b["y0"])
            cb["x1"] = max(cb["x1"], b["x1"])
            cb["y1"] = max(cb["y1"], b["y1"])
        else:
            merged.append(cur)
            cur = {"page": p, "bbox": dict(b)}

    if cur is not None:
        merged.append(cur)
    return merged


def normalize_text(s: str, preserve_whitespace: bool = False) -> str:
    """Normalisasi teks supaya ekstraksi DOCX dan PDF lebih mudah di-align.
    
    Args:
        s: Text to normalize
        preserve_whitespace: If True, don't collapse multiple spaces (for bbox mapping)
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # Normalisasi simbol matematika umum
    s = s.replace("×", "*").replace("÷", "/")
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("≠", "!=").replace("≈", "~=")
    
    # Normalisasi mathematical alphanumeric symbols
    # Mathematical Italic: A-Z (U+1D434-U+1D44D), a-z (U+1D44E-U+1D467)
    # Mathematical Bold: A-Z (U+1D400-U+1D419), a-z (U+1D41A-U+1D433)
    # Mathematical Bold Italic, Script, dll
    
    # Greek mu (common in formulas)
    s = s.replace('\U0001D707', 'μ')  # Mathematical Italic Small Mu -> Greek mu
    s = s.replace('\U0001D6CD', 'μ')  # Mathematical Bold Small Mu -> Greek mu
    
    # Italic uppercase A-Z -> normal A-Z
    for i in range(26):
        s = s.replace(chr(0x1D434 + i), chr(0x41 + i))
    # Italic lowercase a-z -> normal a-z (skip h karena U+210E)
    for i in range(26):
        if i != 7:  # skip h
            s = s.replace(chr(0x1D44E + i), chr(0x61 + i))
    s = s.replace('ℎ', 'h')  # U+210E -> h
    
    # Bold uppercase A-Z -> normal A-Z
    for i in range(26):
        s = s.replace(chr(0x1D400 + i), chr(0x41 + i))
    # Bold lowercase a-z -> normal a-z
    for i in range(26):
        s = s.replace(chr(0x1D41A + i), chr(0x61 + i))
    
    # Bold Italic uppercase A-Z -> normal A-Z
    for i in range(26):
        s = s.replace(chr(0x1D468 + i), chr(0x41 + i))
    # Bold Italic lowercase a-z -> normal a-z
    for i in range(26):
        s = s.replace(chr(0x1D482 + i), chr(0x61 + i))
    
    # Script uppercase A-Z -> normal A-Z
    for i in range(26):
        if i not in [1, 4, 5, 7, 8, 11, 12, 17]:  # skip special chars
            s = s.replace(chr(0x1D49C + i), chr(0x41 + i))
    # Script lowercase a-z -> normal a-z  
    for i in range(26):
        if i not in [4, 6, 11]:  # skip special chars
            s = s.replace(chr(0x1D4B6 + i), chr(0x61 + i))
    
    # Greek letters: Bold/italic Greek -> normal Greek
    for i in range(25):  # Greek has 25 letters
        s = s.replace(chr(0x1D6FC + i), chr(0x03B1 + i))  # lowercase
        s = s.replace(chr(0x1D6E2 + i), chr(0x0391 + i))  # uppercase
    
    # FIX: Gabungkan Greek letter + subscript yang terpisah spasi
    # Pattern: "μ LO" -> "μLO", "α MD" -> "αMD"
    s = re.sub(r'([α-ωΑ-Ω])\s+([A-Z]{1,3})(?=\s|$|[^A-Za-z])', r'\1\2', s)
    
    if not preserve_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def has_formula_in_tree(json_tree):
    """Cek apakah json_tree mengandung formula (type: math)."""
    if not json_tree:
        return False
    
    if isinstance(json_tree, dict):
        if json_tree.get('type') == 'math':
            return True
        for v in json_tree.values():
            if has_formula_in_tree(v):
                return True
    elif isinstance(json_tree, list):
        for item in json_tree:
            if has_formula_in_tree(item):
                return True
    
    return False


def has_shape_in_cells(json_tree):
    """Cek apakah table mengandung shape di cells."""
    if not json_tree or not isinstance(json_tree, dict):
        return False
    
    if "rows" not in json_tree:
        return False
    
    for row in json_tree.get("rows", []):
        if isinstance(row, dict):
            for cell in row.get("cells", []):
                if isinstance(cell, list):
                    # Cell berupa list - cek apakah ada shape
                    for item in cell:
                        if isinstance(item, dict) and item.get("type") == "shape":
                            return True
                elif isinstance(cell, dict):
                    # Cell berupa dict - cek apakah ini shape atau ada shape di dalamnya
                    if cell.get("type") == "shape":
                        return True
    return False


def extract_all_images_recursive(json_tree, images_list=None):
    """Extract all images recursively from any nested structure."""
    if images_list is None:
        images_list = []
    
    if not json_tree:
        return images_list
    
    # Unwrap content wrapper
    if isinstance(json_tree, dict) and "content" in json_tree and not json_tree.get("type"):
        if isinstance(json_tree["content"], dict):
            return extract_all_images_recursive(json_tree["content"], images_list)
        elif isinstance(json_tree["content"], list):
            for item in json_tree["content"]:
                extract_all_images_recursive(item, images_list)
            return images_list
    
    # Handle image
    if isinstance(json_tree, dict) and json_tree.get("type") == "image":
        if json_tree.get("rId"):
            images_list.append(json_tree)
        return images_list
    
    # Handle shape - recursive into content
    if isinstance(json_tree, dict) and json_tree.get("type") == "shape":
        if isinstance(json_tree.get("content"), list):
            for item in json_tree["content"]:
                extract_all_images_recursive(item, images_list)
        return images_list
    
    # Handle table
    if isinstance(json_tree, dict) and "rows" in json_tree:
        for row in json_tree.get("rows", []):
            if isinstance(row, dict):
                for cell in row.get("cells", []):
                    if isinstance(cell, list):
                        for item in cell:
                            extract_all_images_recursive(item, images_list)
                    elif isinstance(cell, dict):
                        extract_all_images_recursive(cell, images_list)
        return images_list
    
    # Handle list
    if isinstance(json_tree, list):
        for item in json_tree:
            extract_all_images_recursive(item, images_list)
    
    return images_list


def extract_all_shapes_recursive(json_tree, shapes_list=None):
    """Extract all shapes recursively from any nested structure."""
    if shapes_list is None:
        shapes_list = []
    
    if not json_tree:
        return shapes_list
    
    # Unwrap content wrapper
    if isinstance(json_tree, dict) and "content" in json_tree and not json_tree.get("type"):
        if isinstance(json_tree["content"], dict):
            return extract_all_shapes_recursive(json_tree["content"], shapes_list)
        elif isinstance(json_tree["content"], list):
            for item in json_tree["content"]:
                extract_all_shapes_recursive(item, shapes_list)
            return shapes_list
    
    # Handle shape
    if isinstance(json_tree, dict) and json_tree.get("type") == "shape":
        if json_tree.get("content"):
            shapes_list.append(json_tree)
            # Recursive into shape content
            if isinstance(json_tree["content"], list):
                for item in json_tree["content"]:
                    extract_all_shapes_recursive(item, shapes_list)
        return shapes_list
    
    # Handle table
    if isinstance(json_tree, dict) and "rows" in json_tree:
        for row in json_tree.get("rows", []):
            if isinstance(row, dict):
                for cell in row.get("cells", []):
                    if isinstance(cell, list):
                        for item in cell:
                            extract_all_shapes_recursive(item, shapes_list)
                    else:
                        extract_all_shapes_recursive(cell, shapes_list)
        return shapes_list
    
    # Handle list
    if isinstance(json_tree, list):
        for item in json_tree:
            extract_all_shapes_recursive(item, shapes_list)
    
    return shapes_list


def extract_shape_content_items(json_tree):
    """Extract content items from shape structure recursively."""
    items = []
    
    # Unwrap content if exists
    if isinstance(json_tree, dict) and "content" in json_tree:
        if isinstance(json_tree["content"], dict):
            json_tree = json_tree["content"]
        elif isinstance(json_tree["content"], list):
            # Shape content array
            for item in json_tree["content"]:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        # Add image marker
                        items.append("[IMAGE]")
                    elif item.get("type") in ["text", "list"] and "value" in item:
                        items.append(item["value"])
                    else:
                        # Recursive for nested structures (including nested tables)
                        items.extend(extract_shape_content_items(item))
            return items
    
    # Handle nested tables
    if isinstance(json_tree, dict) and "rows" in json_tree:
        for row in json_tree.get("rows", []):
            if isinstance(row, dict):
                for cell in row.get("cells", []):
                    if isinstance(cell, str):
                        items.append(cell)
                    elif isinstance(cell, list):
                        for item in cell:
                            items.extend(extract_shape_content_items(item))
                    else:
                        items.extend(extract_shape_content_items(cell))
        return items
    
    return items


def tokenize(s: str):
    """Tokenizer untuk alignment."""
    s = normalize_text(s)
    if not s:
        return []
    
    # NORMALISASI: Hapus spasi di antara huruf dan 2 digit untuk match dengan PDF
    # Contoh: "predikat 2 4" -> "predikat24"
    s = re.sub(r'([a-zα-ω]{2,})\s+(\d)\s+(\d)(?=\s|$)', r'\1\2\3', s)
    # Gabung 2 digit terpisah: "2 4" -> "24"
    s = re.sub(r'(?<=\s)(\d)\s+(\d)(?=\s|$)', r'\1\2', s)
    # FORMULA FIX: Gabung single digit yang terpisah spasi setelah operator
    # Pattern: "- 1 2" -> "- 12", "e - 1 2" -> "e - 12"
    s = re.sub(r'([\-=e])\s+(\d)\s+(\d)', r'\1 \2\3', s)
    
    # WORD WRAP FIX: Gabung kata yang terpisah hyphen
    # Pattern: "pertan- yaan" -> "pertanyaan"
    s = re.sub(r'([a-zA-Zα-ωΑ-Ω]+)-\s+([a-zA-Zα-ωΑ-Ω]+)', r'\1\2', s)
    
    # Tokenize dengan support Greek letters (U+0370-U+03FF), subscript (₀-₉), superscript (⁰-⁹), underscore (_)
    # PENTING: Pisahkan footnote untuk kata huruf besar, tapi gabung subscript dan underscore
    tokens = re.findall(r'\d*[A-Z\u00c0-\u00df\u0391-\u03a9][A-Za-z\u00c0-\u00ff\u0370-\u03ff\u2080-\u2089\u2070-\u2079_]*(?=\d{1,2}(?!\d))|\d*[A-Za-z\u00c0-\u00ff\u0370-\u03ff\u2080-\u2089\u2070-\u2079_]+\d*|\d+(?:\.\d+)*|[^\w\s]', s, flags=re.UNICODE)
    # Hapus titik atau titik dua di akhir
    if tokens and tokens[-1] in ('.', ':'):
        tokens = tokens[:-1]
    return tokens


def extract_text_from_json_tree(json_tree, return_cells=False, return_is_shape_table=False):
    """Ekstrak teks dari dokumen_elemen_json_tree (OpenXML).
    
    Args:
        json_tree: JSON tree dari OpenXML
        return_cells: Jika True dan ini table, return list of cell texts
        return_is_shape_table: Jika True, return tuple (result, is_shape_table)
    
    Returns:
        String teks gabungan, atau list of cell texts jika return_cells=True dan ada table
        Jika return_is_shape_table=True, return tuple (result, is_shape_table)
    """
    if not json_tree:
        result = "" if not return_cells else []
        return (result, False) if return_is_shape_table else result

    is_shape_table = False
    
    # Unwrap content if exists
    if isinstance(json_tree, dict) and "content" in json_tree and isinstance(json_tree["content"], dict):
        json_tree = json_tree["content"]
    
    # Cek apakah ini table
    if isinstance(json_tree, dict) and "rows" in json_tree:
        cells = []
        for row in json_tree.get("rows", []):
            if isinstance(row, dict):
                for cell in row.get("cells", []):
                    if isinstance(cell, str):
                        cells.append(cell)
                    elif isinstance(cell, list):
                        # Cell berupa list - gabungkan semua items jadi 1 cell text
                        cell_texts = []
                        for item in cell:
                            if isinstance(item, dict):
                                # Handle text/list type directly
                                if item.get("type") in ["text", "list"] and "value" in item:
                                    cell_texts.append(item["value"])
                                # Skip images
                                elif item.get("type") == "image":
                                    continue
                                # Handle other types (shape, nested table, etc)
                                else:
                                    item_text = extract_text_from_json_tree(item, return_cells=False)
                                    if item_text and item_text.strip():
                                        cell_texts.append(item_text)
                                        if item.get("type") == "shape" and "content" in item:
                                            is_shape_table = True
                        # Gabung jadi 1 cell
                        if cell_texts:
                            cells.append(" ".join(cell_texts))
                    elif isinstance(cell, dict):
                        # Cell berupa dict
                        cell_text = extract_text_from_json_tree(cell, return_cells=False)
                        if cell_text and cell_text.strip():
                            cells.append(cell_text)
                            if cell.get("type") == "shape" and "content" in cell:
                                is_shape_table = True
        
        if return_cells:
            result = cells
        else:
            result = " ".join(cells)
        
        return (result, is_shape_table) if return_is_shape_table else result

    texts = []

    def rec(node):
        if isinstance(node, dict):
            if "rows" in node:
                for row in node.get("rows", []):
                    if isinstance(row, dict):
                        for cell in row.get("cells", []):
                            if isinstance(cell, str):
                                texts.append(cell)
                            elif isinstance(cell, list):
                                # Cell bisa berupa list of shapes
                                for item in cell:
                                    if isinstance(item, dict):
                                        rec(item)
                            elif isinstance(cell, dict):
                                rec(cell)
                return

            # Handle shape dengan content
            if node.get("type") == "shape" and "content" in node:
                content = node.get("content")
                if isinstance(content, list):
                    for item in content:
                        rec(item)
                return

            if node.get("type") == "text" and "value" in node:
                texts.append(node["value"])
            elif node.get("type") == "list" and "value" in node:
                texts.append(node["value"])
            elif node.get("type") == "math" and "text" in node:
                texts.append(node["text"])

            # Iterasi semua values
            for k, v in node.items():
                rec(v)

        elif isinstance(node, list):
            for x in node:
                rec(x)

    rec(json_tree)
    result = " ".join(texts)
    # Fallback: jika tidak ada teks, coba ambil dari 'content' array langsung
    if not result and isinstance(json_tree, dict) and "content" in json_tree:
        content = json_tree["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and "value" in item:
                    texts.append(item["value"])
            result = " ".join(texts)
    
    return (result, False) if return_is_shape_table else result


def align_document(pdf_path: str, elements: list, log_file=None) -> dict:
    """
    Align OpenXML elements dengan PDF menggunakan global difflib alignment.

    Args:
        pdf_path: Path ke file PDF
        elements: List of DokumenElemen objects (sudah sorted by sequence)
        log_file: Optional file handle untuk logging

    Returns:
        dict dengan:
        - aligned_words: list of {text, bbox, page, element_id, before_align_bboxes}
        - stats: {total_words, assigned_words, coverage}
    """
    import sys

    if log_file:
        log_file.write("\n=== ALIGNMENT TRACE LOG ===\n")
        log_file.flush()
    sys.stderr.write("\n=== ALIGNMENT TRACE LOG ===\n")
    sys.stderr.flush()

    # Extract images dari PDF
    pdf_images = {}  # page_index -> list of image bboxes
    with fitz.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            image_list = page.get_images(full=True)
            page_images = []
            for img_index, img in enumerate(image_list):
                # Get image bbox
                img_rects = page.get_image_rects(img[0])
                if img_rects:
                    for rect_idx, rect in enumerate(img_rects):
                        page_images.append({
                            'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                            'xref': img[0],
                            'img_index': img_index,
                            'rect_index': rect_idx,
                        })
            pdf_images[page_index] = page_images

    # Build global DOCX token stream
    docx_tokens = []
    docx_owner = []
    docx_cell_index = []  # Track cell index untuk table
    docx_is_formula = []  # Track apakah token dari formula

    for elem in elements:
        elem_text = extract_text_from_json_tree(elem.delemen_json_tree)
        
        # Cek apakah ini table
        cells, is_shape_table = extract_text_from_json_tree(elem.delemen_json_tree, return_cells=True, return_is_shape_table=True)
        # Table jika: multiple cells ATAU shape table (flowchart)
        is_table = isinstance(cells, list) and (len(cells) > 1 or is_shape_table)
        
        # Cek apakah mengandung formula
        has_formula = has_formula_in_tree(elem.delemen_json_tree)
        
        if is_table:
            # Table: tokenize per cell
            for cell_idx, cell_text in enumerate(cells):
                toks = tokenize(cell_text)
                docx_tokens.extend(toks)
                docx_owner.extend([elem.delemen_id] * len(toks))
                docx_cell_index.extend([cell_idx] * len(toks))
                docx_is_formula.extend([has_formula] * len(toks))
        else:
            # Non-table: tokenize biasa
            toks = tokenize(elem_text)
            docx_tokens.extend(toks)
            docx_owner.extend([elem.delemen_id] * len(toks))
            docx_cell_index.extend([-1] * len(toks))  # -1 = bukan table
            docx_is_formula.extend([has_formula] * len(toks))

    if log_file:
        log_file.write(f"Total DOCX tokens: {len(docx_tokens)}\n")
        log_file.flush()
    sys.stderr.write(f"Total DOCX tokens: {len(docx_tokens)}\n")
    sys.stderr.flush()

    # Build global PDF token stream (token-level bbox)
    pdf_tokens = []
    pdf_bboxes = []
    pdf_pages = []

    # NOTE: pakai rawdict supaya bbox per token (bukan per word), agar karakter tambahan
    # yang tidak ter-align tidak ikut "ketarik" ke bbox elemen.
    with fitz.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]

            for tok, bbox, _ in iter_pdf_tokens_with_bboxes(page, page_index):
                pdf_tokens.append(tok)
                pdf_bboxes.append(bbox)
                pdf_pages.append(page_index)

    if log_file:
        log_file.write(f"Total PDF tokens: {len(pdf_tokens)}\n")
        log_file.flush()
    sys.stderr.write(f"Total PDF tokens: {len(pdf_tokens)}\n")
    sys.stderr.flush()

    # Global alignment dengan spatial constraint untuk table cells
    sm = difflib.SequenceMatcher(a=docx_tokens, b=pdf_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    docx_to_pdf = [None] * len(docx_tokens)
    docx_to_pdf_multi = {}  # Track 1 DOCX token -> multiple PDF tokens
    last_pdf_idx = -1
    last_cell_idx = -1
    last_elem_id = -1
    
    # DEBUG: Find table 39498 tokens
    table_39498_indices = []
    for i, elem_id in enumerate(docx_owner):
        if elem_id == 39498:
            table_39498_indices.append(i)
    
    if table_39498_indices and log_file:
        log_file.write(f"\nTable 39498 DOCX tokens: {table_39498_indices[0]} to {table_39498_indices[-1]}\n")
        log_file.write(f"Cell indices: {[docx_cell_index[i] for i in table_39498_indices[:20]]}\n")
        log_file.write(f"Tokens: {[docx_tokens[i] for i in table_39498_indices[:20]]}\n")
        log_file.flush()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                pdf_idx = j1 + k
                curr_cell_idx = docx_cell_index[i1 + k]
                curr_elem_id = docx_owner[i1 + k]
                
                # Detect cell transition in same table
                if curr_elem_id == last_elem_id and curr_cell_idx != last_cell_idx and curr_cell_idx != -1 and last_cell_idx != -1:
                    # New cell in same table - COMPLETELY RESET monotonic constraint
                    # This allows the new cell to start from any earlier PDF position
                    last_pdf_idx = -1  # RESET to allow going back
                    last_cell_idx = curr_cell_idx
                    last_elem_id = curr_elem_id
                
                # Apply monotonic constraint (or skip if just reset)
                if pdf_idx > last_pdf_idx:
                    docx_to_pdf[i1 + k] = pdf_idx
                    last_pdf_idx = pdf_idx
                    if curr_cell_idx != last_cell_idx or curr_elem_id != last_elem_id:
                        last_cell_idx = curr_cell_idx
                        last_elem_id = curr_elem_id
        elif tag == "replace" and i2 - i1 == 1 and j2 - j1 > 1:
            # WORD WRAP FIX: 1 DOCX token match ke multiple PDF tokens
            docx_token = docx_tokens[i1]
            pdf_segment = pdf_tokens[j1:j2]
            
            # Cek apakah PDF tokens bisa digabung jadi DOCX token
            combined = "".join(pdf_segment)
            if combined == docx_token or combined.replace("-", "") == docx_token:
                # Match! Assign semua PDF tokens ke DOCX token ini
                pdf_indices = list(range(j1, j2))
                curr_cell_idx = docx_cell_index[i1]
                curr_elem_id = docx_owner[i1]
                
                # Detect cell transition
                if curr_elem_id == last_elem_id and curr_cell_idx != last_cell_idx and curr_cell_idx != -1 and last_cell_idx != -1:
                    # New cell - RESET
                    last_pdf_idx = -1
                    last_cell_idx = curr_cell_idx
                    last_elem_id = curr_elem_id
                
                # Apply monotonic constraint
                if all(idx > last_pdf_idx for idx in pdf_indices):
                    docx_to_pdf[i1] = j1
                    docx_to_pdf_multi[i1] = pdf_indices
                    last_pdf_idx = j2 - 1
                    if curr_cell_idx != last_cell_idx or curr_elem_id != last_elem_id:
                        last_cell_idx = curr_cell_idx
                        last_elem_id = curr_elem_id

    # Build aligned_words
    element_groups = defaultdict(lambda: defaultdict(list))  # elem_id -> cell_idx -> words
    element_texts = {}  # Store RAW text without normalization
    element_is_table = {}
    element_cell_texts = {}  # elem_id -> list of cell texts (RAW, tidak dinormalisasi)

    for elem in elements:
        elem_text = extract_text_from_json_tree(elem.delemen_json_tree)
        # Store RAW text, tidak dinormalisasi
        element_texts[elem.delemen_id] = elem_text
        
        # DEBUG: Log semua element texts
        if log_file and elem.delemen_id >= 20111 and elem.delemen_id <= 20115:
            log_file.write(f"Stored element {elem.delemen_id}: '{elem_text}'\n")
        
        # DEBUG: Log jika text kosong
        if not elem_text or not elem_text.strip():
            if log_file:
                log_file.write(f"WARNING: Element {elem.delemen_id} has empty text\n")
        
        cells = extract_text_from_json_tree(elem.delemen_json_tree, return_cells=True)
        element_is_table[elem.delemen_id] = isinstance(cells, list) and (len(cells) > 1 or has_shape_in_cells(elem.delemen_json_tree))
        if element_is_table[elem.delemen_id]:
            # Store RAW cell texts (tidak dinormalisasi)
            element_cell_texts[elem.delemen_id] = cells

    for i, elem_id in enumerate(docx_owner):
        j = docx_to_pdf[i]
        if j is None:
            continue

        cell_idx = docx_cell_index[i]
        is_formula = docx_is_formula[i]
        
        # Check if this DOCX token maps to multiple PDF tokens (word wrap)
        if i in docx_to_pdf_multi:
            # Add all PDF tokens
            for pdf_idx in docx_to_pdf_multi[i]:
                bbox = pdf_bboxes[pdf_idx]
                element_groups[elem_id][cell_idx].append({
                    "text": pdf_tokens[pdf_idx],
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "page": pdf_pages[pdf_idx],
                    "is_formula": is_formula,
                    "pdf_index": pdf_idx,
                    "docx_token_index": i,
                })
        else:
            # Single PDF token
            bbox = pdf_bboxes[j]
            element_groups[elem_id][cell_idx].append({
                "text": pdf_tokens[j],
                "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                "page": pdf_pages[j],
                "is_formula": is_formula,
                "pdf_index": j,
                "docx_token_index": i,
            })

    # FILTER: Remove tokens dari baris lain HANYA untuk formula
    # Cek apakah formula mengambil token dari baris bawah yang seharusnya milik elemen lain
    for elem_id in element_groups:
        for cell_idx in element_groups[elem_id]:
            words = element_groups[elem_id][cell_idx]
            if len(words) > 1 and words[0].get('is_formula', False):
                # Group by baris berdasarkan Y position
                y_mids = [(w['bbox']['y0'] + w['bbox']['y1']) / 2 for w in words]
                y_sorted = sorted(set(y_mids))
                
                # Jika ada multiple baris
                if len(y_sorted) > 1:
                    # Hitung gap antar baris
                    gaps = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]
                    max_gap = max(gaps)
                    
                    # Gap dianggap besar jika > tinggi font rata-rata
                    heights = [w['bbox']['y1'] - w['bbox']['y0'] for w in words]
                    avg_height = sum(heights) / len(heights)
                    
                    if max_gap > avg_height:
                        # Identifikasi baris dengan token terbanyak (main line)
                        from collections import Counter
                        y_groups = Counter(round(y) for y in y_mids)
                        main_y = y_groups.most_common(1)[0][0]
                        
                        # Pisahkan token main line dan outlier
                        main_tokens = [w for w in words if abs(round((w['bbox']['y0'] + w['bbox']['y1']) / 2) - main_y) <= avg_height * 0.5]
                        outlier_tokens = [w for w in words if abs(round((w['bbox']['y0'] + w['bbox']['y1']) / 2) - main_y) > avg_height * 0.5]
                        
                        if outlier_tokens:
                            # Cek apakah ada elemen SETELAHNYA yang memiliki Y sama dengan outlier
                            outlier_y_avg = sum((w['bbox']['y0'] + w['bbox']['y1']) / 2 for w in outlier_tokens) / len(outlier_tokens)
                            page_num = words[0]['page']
                            
                            # Cari elemen dengan ID lebih besar (elemen setelahnya)
                            next_elem_has_same_y = False
                            for other_elem_id in element_groups:
                                if other_elem_id <= elem_id:
                                    continue
                                
                                for other_cell_idx in element_groups[other_elem_id]:
                                    other_words = element_groups[other_elem_id][other_cell_idx]
                                    if not other_words or other_words[0]['page'] != page_num:
                                        continue
                                    
                                    # Cek apakah elemen setelahnya punya token di Y yang sama dengan outlier
                                    for ow in other_words:
                                        ow_y = (ow['bbox']['y0'] + ow['bbox']['y1']) / 2
                                        if abs(ow_y - outlier_y_avg) <= avg_height:
                                            next_elem_has_same_y = True
                                            break
                                    
                                    if next_elem_has_same_y:
                                        break
                                
                                if next_elem_has_same_y:
                                    break
                            
                            # Jika elemen setelahnya punya Y yang sama, buang token outlier
                            if next_elem_has_same_y and main_tokens:
                                element_groups[elem_id][cell_idx] = main_tokens

    # Create final aligned_words
    final_aligned = []
    
    # Track which PDF images have been used
    used_images = set()
    
    # Track unmapped table cells untuk fallback matching
    unmapped_table_cells = []  # (elem_id, cell_idx, cell_text)
    
    # Check ALL table elements for unmapped cells (not just those in element_groups)
    for elem in elements:
        elem_id = elem.delemen_id
        if not element_is_table.get(elem_id, False):
            continue
        
        cells = extract_text_from_json_tree(elem.delemen_json_tree, return_cells=True)
        if not cells:
            continue
        
        # Check which cells are unmapped
        cell_groups = element_groups.get(elem_id, {})
        for cell_idx, cell_text in enumerate(cells):
            if cell_idx not in cell_groups or not cell_groups[cell_idx]:
                unmapped_table_cells.append((elem_id, cell_idx, cell_text))
    
    # Track which elements were aligned (including parent IDs for tables)
    aligned_elem_ids = set(element_groups.keys())
    aligned_parent_ids = set()  # Track parent IDs for table cells
    
    # Add all table elements to aligned_parent_ids (even if no cells aligned yet)
    for elem_id in element_is_table:
        if element_is_table[elem_id]:
            aligned_parent_ids.add(elem_id)
    
    # Build hierarchy: table -> cell -> content (recursive)
    def add_hierarchy_recursive(elem_id, cell_groups, parent_id=None, depth=0):
        is_table = element_is_table.get(elem_id, False)
        
        if is_table:
            # Add table container
            first_page = 0
            all_cell_bboxes = []
            for cell_idx, words in cell_groups.items():
                if words:
                    if first_page == 0:
                        first_page = words[0]["page"]
                    x0 = min(w["bbox"]["x0"] for w in words)
                    y0 = min(w["bbox"]["y0"] for w in words)
                    x1 = max(w["bbox"]["x1"] for w in words)
                    y1 = max(w["bbox"]["y1"] for w in words)
                    all_cell_bboxes.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
            
            if all_cell_bboxes:
                table_x0 = min(b["x0"] for b in all_cell_bboxes)
                table_y0 = min(b["y0"] for b in all_cell_bboxes)
                table_x1 = max(b["x1"] for b in all_cell_bboxes)
                table_y1 = max(b["y1"] for b in all_cell_bboxes)
                
                table_text = element_texts.get(elem_id, "")
                container_id = f"{elem_id}_table" if parent_id else elem_id
                
                final_aligned.append({
                    "text": table_text,
                    "matched_text": table_text,
                    "bbox": {"x0": table_x0, "y0": table_y0, "x1": table_x1, "y1": table_y1},
                    "bboxes": [],
                    "page": first_page,
                    "element_id": container_id,
                    "parent_element_id": parent_id,
                    "confidence": 1.0,
                    "is_table_container": True,
                    "depth": depth,
                    "before_align_bboxes": [],
                })
                
                # Add cells
                cell_order = [(min(w.get('pdf_index', float('inf')) for w in words), cell_idx) 
                              for cell_idx, words in cell_groups.items() if words]
                cell_order.sort()
                
                for _, cell_idx in cell_order:
                    words = cell_groups[cell_idx]
                    cell_text = ""
                    if elem_id in element_cell_texts and cell_idx < len(element_cell_texts[elem_id]):
                        cell_text = element_cell_texts[elem_id][cell_idx]
                    
                    matched_text = " ".join(w["text"] for w in words)
                    if not cell_text or not cell_text.strip():
                        cell_text = matched_text
                    
                    x0 = min(w["bbox"]["x0"] for w in words)
                    y0 = min(w["bbox"]["y0"] for w in words)
                    x1 = max(w["bbox"]["x1"] for w in words)
                    y1 = max(w["bbox"]["y1"] for w in words)
                    
                    merged_segments = merge_bboxes_token_level(words, is_formula=False)
                    
                    # Extract content items
                    content_items = []
                    content_bboxes = []
                    elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                    if elem and elem.delemen_json_tree:
                        # Get cell structure from original JSON
                        json_tree = elem.delemen_json_tree
                        if isinstance(json_tree, dict) and "content" in json_tree:
                            json_tree = json_tree["content"]
                        
                        if isinstance(json_tree, dict) and "rows" in json_tree:
                            # Find the cell in the table structure
                            cell_count = 0
                            found_cell = None
                            for row in json_tree.get("rows", []):
                                if isinstance(row, dict):
                                    for cell in row.get("cells", []):
                                        if cell_count == cell_idx:
                                            found_cell = cell
                                            break
                                        cell_count += 1
                                    if found_cell:
                                        break
                            
                            # Extract content items from the cell
                            if found_cell:
                                if isinstance(found_cell, str):
                                    # Simple string cell
                                    content_items = [found_cell]
                                    content_bboxes = [{"x0": x0, "y0": y0, "x1": x1, "y1": y1}]
                                elif isinstance(found_cell, list):
                                    # Cell with multiple items (shapes, text, images, etc)
                                    for item in found_cell:
                                        if isinstance(item, dict):
                                            if item.get("type") == "image":
                                                content_items.append("[IMAGE]")
                                            elif item.get("type") in ["text", "list"] and "value" in item:
                                                content_items.append(item["value"])
                                            elif item.get("type") == "shape" and "content" in item:
                                                # Extract from shape
                                                shape_items = extract_shape_content_items(item)
                                                content_items.extend(shape_items)
                                            else:
                                                # Other types (nested table, etc)
                                                item_text = extract_text_from_json_tree(item, return_cells=False)
                                                if item_text and item_text.strip():
                                                    content_items.append(item_text)
                                elif isinstance(found_cell, dict):
                                    # Cell is a dict (shape or other structure)
                                    if found_cell.get("type") == "shape" and "content" in found_cell:
                                        content_items = extract_shape_content_items(found_cell)
                                    else:
                                        cell_text_extracted = extract_text_from_json_tree(found_cell, return_cells=False)
                                        if cell_text_extracted:
                                            content_items = [cell_text_extracted]
                            
                            # Generate bboxes for content items
                            if content_items:
                                token_idx = 0
                                for content_text in content_items:
                                    if content_text == "[IMAGE]":
                                        # Image - add dummy bbox
                                        content_bboxes.append({"x0": 0, "y0": 0, "x1": 0, "y1": 0})
                                    else:
                                        content_tokens = tokenize(content_text)
                                        item_bboxes = []
                                        for _ in content_tokens:
                                            if token_idx < len(words):
                                                item_bboxes.append(words[token_idx]["bbox"])
                                                token_idx += 1
                                        if item_bboxes:
                                            item_x0 = min(b["x0"] for b in item_bboxes)
                                            item_y0 = min(b["y0"] for b in item_bboxes)
                                            item_x1 = max(b["x1"] for b in item_bboxes)
                                            item_y1 = max(b["y1"] for b in item_bboxes)
                                            content_bboxes.append({"x0": item_x0, "y0": item_y0, "x1": item_x1, "y1": item_y1})
                                        else:
                                            content_bboxes.append({"x0": 0, "y0": 0, "x1": 0, "y1": 0})
                        
                        # Fallback if no content items found
                        if not content_items:
                            content_items = [cell_text]
                            content_bboxes = [{"x0": x0, "y0": y0, "x1": x1, "y1": y1}]
                    
                    cell_id = f"{elem_id}_cell_{cell_idx}"
                    result = {
                        "text": cell_text,
                        "matched_text": matched_text,
                        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bboxes": merged_segments,
                        "page": words[0]["page"],
                        "element_id": cell_id,
                        "parent_element_id": container_id,
                        "confidence": 1.0,
                        "depth": depth + 1,
                        "before_align_bboxes": [w["bbox"] for w in words],
                    }
                    if content_items:
                        result["content_items"] = content_items
                        result["content_bboxes"] = content_bboxes
                    
                    final_aligned.append(result)
        else:
            # Non-table element
            words = cell_groups.get(-1, [])
            if words:
                page_groups = defaultdict(list)
                for w in words:
                    page_groups[w['page']].append(w)
                
                for page_num in sorted(page_groups.keys()):
                    page_words = page_groups[page_num]
                    x0 = min(w["bbox"]["x0"] for w in page_words)
                    y0 = min(w["bbox"]["y0"] for w in page_words)
                    x1 = max(w["bbox"]["x1"] for w in page_words)
                    y1 = max(w["bbox"]["y1"] for w in page_words)
                    
                    elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                    is_formula = elem and elem.delemen_type == 'math'
                    merged_segments = merge_bboxes_token_level(page_words, is_formula=is_formula)
                    
                    final_elem_id = elem_id if len(page_groups) == 1 else f"{elem_id}_page_{page_num}"
                    elem_text = element_texts.get(elem_id, "")
                    matched_text = " ".join(w["text"] for w in page_words)
                    
                    if not elem_text or not elem_text.strip():
                        elem_text = matched_text
                    
                    final_aligned.append({
                        "text": elem_text,
                        "matched_text": matched_text,
                        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bboxes": merged_segments,
                        "page": page_num,
                        "element_id": final_elem_id,
                        "parent_element_id": parent_id,
                        "confidence": 1.0,
                        "depth": depth,
                        "before_align_bboxes": [w["bbox"] for w in page_words],
                    })
    
    for elem_id, cell_groups in element_groups.items():
        if cell_groups:
            add_hierarchy_recursive(elem_id, cell_groups)
        else:
            # Non-table: bbox biasa
            words = cell_groups.get(-1, [])  # cell_idx = -1 untuk non-table
            if not words:
                continue
            
            # Group by page untuk paragraf yang terpotong
            page_groups = defaultdict(list)
            for w in words:
                page_groups[w['page']].append(w)
            
            # Buat elemen terpisah per page (jika terpotong)
            for page_num in sorted(page_groups.keys()):
                page_words = page_groups[page_num]
                
                x0 = min(w["bbox"]["x0"] for w in page_words)
                y0 = min(w["bbox"]["y0"] for w in page_words)
                x1 = max(w["bbox"]["x1"] for w in page_words)
                y1 = max(w["bbox"]["y1"] for w in page_words)

                # Cek apakah element ini formula
                elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                is_formula = elem and elem.delemen_type == 'math'
                
                merged_segments = merge_bboxes_token_level(page_words, is_formula=is_formula)

                # Gunakan element_id unik per page jika terpotong
                final_elem_id = elem_id if len(page_groups) == 1 else f"{elem_id}_page_{page_num}"
                
                # Get text, fallback to matched_text if empty
                elem_text = element_texts.get(elem_id, "")
                matched_text = " ".join(w["text"] for w in page_words)
                
                # DEBUG: Log jika text kosong
                if (not elem_text or not elem_text.strip()) and log_file:
                    log_file.write(f"Element {elem_id}: empty text, using matched_text: {matched_text[:50]}\n")
                
                if not elem_text or not elem_text.strip():
                    elem_text = matched_text
                
                final_aligned.append({
                    "text": elem_text,
                    "matched_text": matched_text,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "bboxes": merged_segments,
                    "page": page_num,
                    "element_id": final_elem_id,
                    "confidence": 1.0,
                    "before_align_bboxes": [w["bbox"] for w in page_words],
                })
    
    # Track which PDF tokens have been used
    used_pdf_tokens = set()
    for elem_id, cell_groups in element_groups.items():
        for cell_idx, words in cell_groups.items():
            for w in words:
                used_pdf_tokens.add(w['pdf_index'])
    
    if log_file:
        log_file.write(f"\nUsed PDF tokens: {len(used_pdf_tokens)}/{len(pdf_tokens)}\n")
    
    # FALLBACK: Try to match unmapped table cells using exact sequence search
    if unmapped_table_cells:
        if log_file:
            log_file.write(f"\nFallback: Found {len(unmapped_table_cells)} unmapped table cells\n")
        
        for elem_id, cell_idx, cell_text in unmapped_table_cells:
            if log_file:
                log_file.write(f"  Trying to match cell {elem_id}_cell_{cell_idx}: {cell_text[:50]}...\n")
            
            cell_tokens = tokenize(cell_text)
            if not cell_tokens or len(cell_tokens) == 0:
                if log_file:
                    log_file.write(f"    Skipped: no tokens\n")
                continue
            
            # Get page from other cells in same table, or search all pages
            target_pages = []
            if elem_id in element_groups:
                for other_cell_idx, words in element_groups[elem_id].items():
                    if words:
                        page = words[0]['page']
                        if page not in target_pages:
                            target_pages.append(page)
            
            # If no pages found, search all pages
            if not target_pages:
                target_pages = list(range(len(pdf_images)))
            
            # Search for matching sequence in PDF on target pages (allow skipping tokens)
            best_match = None
            best_match_indices = []
            best_match_len = 0
            
            for target_page in target_pages:
                for j in range(len(pdf_tokens)):
                    if pdf_pages[j] != target_page:
                        continue
                    
                    # Skip if already used
                    if j in used_pdf_tokens:
                        continue
                    
                    # Try to match as many tokens as possible from this position
                    # Allow skipping up to 2 tokens (for numbering like "1.", "2.")
                    matched_indices = []
                    pdf_idx = j
                    max_skip = 2
                    
                    for k in range(len(cell_tokens)):
                        # Try to find next matching token within skip window
                        found = False
                        for skip in range(max_skip + 1):
                            check_idx = pdf_idx + skip
                            if check_idx < len(pdf_tokens) and pdf_pages[check_idx] == target_page:
                                # Skip if already used
                                if check_idx in used_pdf_tokens:
                                    continue
                                if pdf_tokens[check_idx] == cell_tokens[k]:
                                    matched_indices.append(check_idx)
                                    pdf_idx = check_idx + 1
                                    found = True
                                    break
                        if not found:
                            break
                    
                    # Keep best match (longest sequence)
                    if len(matched_indices) > best_match_len:
                        best_match_len = len(matched_indices)
                        best_match = j
                        best_match_indices = matched_indices
            
            # Accept match if we got at least 1 token (for single-token cells like "A", "START")
            if best_match is not None and best_match_len > 0:
                matched_words = []
                for pdf_idx in best_match_indices:
                    bbox = pdf_bboxes[pdf_idx]
                    matched_words.append({
                        "text": pdf_tokens[pdf_idx],
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "page": pdf_pages[pdf_idx],
                    })
                
                if matched_words:
                    x0 = min(w["bbox"]["x0"] for w in matched_words)
                    y0 = min(w["bbox"]["y0"] for w in matched_words)
                    x1 = max(w["bbox"]["x1"] for w in matched_words)
                    y1 = max(w["bbox"]["y1"] for w in matched_words)
                    
                    merged_segments = merge_bboxes_token_level(matched_words, is_formula=False)
                    
                    # Confidence based on match completeness
                    confidence = best_match_len / len(cell_tokens)
                    
                    # Group tokens into content items (split by newline/paragraph in original text)
                    content_items = []
                    content_bboxes = []  # Bbox per content item
                    
                    # Get original shape structure from element
                    elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                    if elem and elem.delemen_json_tree:
                        all_shapes = extract_all_shapes_recursive(elem.delemen_json_tree)
                        if cell_idx < len(all_shapes):
                            shape = all_shapes[cell_idx]
                            content_items = extract_shape_content_items(shape)
                            
                            # Match content items to tokens and create bboxes
                            if content_items:
                                token_idx = 0
                                for content_text in content_items:
                                    content_tokens = tokenize(content_text)
                                    item_bboxes = []
                                    for _ in content_tokens:
                                        if token_idx < len(matched_words):
                                            item_bboxes.append(matched_words[token_idx]["bbox"])
                                            token_idx += 1
                                    # Always add bbox even if empty (to maintain index alignment)
                                    if item_bboxes:
                                        item_x0 = min(b["x0"] for b in item_bboxes)
                                        item_y0 = min(b["y0"] for b in item_bboxes)
                                        item_x1 = max(b["x1"] for b in item_bboxes)
                                        item_y1 = max(b["y1"] for b in item_bboxes)
                                        content_bboxes.append({"x0": item_x0, "y0": item_y0, "x1": item_x1, "y1": item_y1})
                                    else:
                                        # No tokens matched for this content item, add dummy bbox
                                        content_bboxes.append({"x0": 0, "y0": 0, "x1": 0, "y1": 0})
                    else:
                        # Fallback: use Y-gap heuristic
                        current_item_tokens = []
                        current_item_bboxes = []
                        
                        for i, w in enumerate(matched_words):
                            current_item_tokens.append(w["text"])
                            current_item_bboxes.append(w["bbox"])
                            
                            is_end = False
                            if i < len(matched_words) - 1:
                                next_w = matched_words[i + 1]
                                y_gap = abs(next_w["bbox"]["y0"] - w["bbox"]["y1"])
                                if y_gap > 5:
                                    is_end = True
                            else:
                                is_end = True
                            
                            if is_end and current_item_tokens:
                                content_items.append(" ".join(current_item_tokens))
                                item_x0 = min(b["x0"] for b in current_item_bboxes)
                                item_y0 = min(b["y0"] for b in current_item_bboxes)
                                item_x1 = max(b["x1"] for b in current_item_bboxes)
                                item_y1 = max(b["y1"] for b in current_item_bboxes)
                                content_bboxes.append({"x0": item_x0, "y0": item_y0, "x1": item_x1, "y1": item_y1})
                                current_item_tokens = []
                                current_item_bboxes = []
                    
                    if log_file:
                        log_file.write(f"    Matched {best_match_len}/{len(cell_tokens)} tokens (conf={confidence:.2f}), {len(content_items)} content items\n")
                    
                    final_aligned.append({
                        "text": cell_text,
                        "matched_text": " ".join(w["text"] for w in matched_words),
                        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bboxes": merged_segments,
                        "page": matched_words[0]["page"],
                        "element_id": f"{elem_id}_cell_{cell_idx}",
                        "parent_element_id": elem_id,
                        "confidence": confidence,
                        "before_align_bboxes": [w["bbox"] for w in matched_words],
                        "content_items": content_items,  # Text per content item
                        "content_bboxes": content_bboxes,  # Bbox per content item
                    })
                elif log_file:
                    log_file.write(f"    No match found\n")
    
    # FALLBACK: Fuzzy matching for formulas with low coverage
    for elem in elements:
        if elem.delemen_type != 'math':
            continue
        
        # Check if already aligned
        existing = next((r for r in final_aligned if r.get('element_id') == elem.delemen_id), None)
        if not existing:
            continue
        
        # Check coverage
        elem_text = extract_text_from_json_tree(elem.delemen_json_tree)
        elem_tokens = tokenize(elem_text)
        matched_tokens = existing.get('matched_text', '').split()
        
        if not elem_tokens:
            continue
        
        coverage = len(matched_tokens) / len(elem_tokens)
        
        # If coverage < 90%, try fuzzy matching for missing tokens
        if coverage < 0.9:
            # Get page and bbox from existing alignment
            page = existing['page']
            bbox = existing['bbox']
            
            # Find missing tokens
            matched_set = set(matched_tokens)
            missing_tokens = [t for t in elem_tokens if t not in matched_set]
            
            if not missing_tokens:
                continue
            
            # Search for missing tokens near existing bbox
            additional_words = []
            for missing_tok in missing_tokens[:5]:  # Limit to 5 missing tokens
                for j in range(len(pdf_tokens)):
                    if pdf_pages[j] != page:
                        continue
                    if pdf_tokens[j] != missing_tok:
                        continue
                    
                    # Check if near existing bbox (within 100px)
                    tok_bbox = pdf_bboxes[j]
                    x_dist = min(abs(tok_bbox[0] - bbox['x1']), abs(tok_bbox[2] - bbox['x0']))
                    y_dist = min(abs(tok_bbox[1] - bbox['y1']), abs(tok_bbox[3] - bbox['y0']))
                    
                    if x_dist < 100 and y_dist < 50:
                        additional_words.append({
                            "text": pdf_tokens[j],
                            "bbox": {"x0": tok_bbox[0], "y0": tok_bbox[1], "x1": tok_bbox[2], "y1": tok_bbox[3]},
                            "page": page,
                        })
                        break
            
            # Merge with existing
            if additional_words:
                all_words = existing.get('before_align_bboxes', [])
                for w in additional_words:
                    all_words.append(w['bbox'])
                
                # Recalculate bbox
                all_bboxes = [existing['bbox']] + [w['bbox'] for w in additional_words]
                x0 = min(b['x0'] for b in all_bboxes)
                y0 = min(b['y0'] for b in all_bboxes)
                x1 = max(b['x1'] for b in all_bboxes)
                y1 = max(b['y1'] for b in all_bboxes)
                
                existing['bbox'] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
                existing['matched_text'] += ' ' + ' '.join(w['text'] for w in additional_words)
                existing['before_align_bboxes'] = all_words
                existing['confidence'] = 0.85  # Lower confidence for fuzzy match
    
    # FALLBACK: Add elements that were not aligned at all
    if log_file:
        log_file.write(f"\nChecking for unaligned elements...\n")
    
    for elem in elements:
        elem_id = elem.delemen_id
        if elem_id in aligned_elem_ids or elem_id in aligned_parent_ids:
            continue
        
        # Element tidak ter-align sama sekali
        elem_text = element_texts.get(elem_id, "")
        if not elem_text or not elem_text.strip():
            continue
        
        if log_file:
            log_file.write(f"Element {elem_id} not aligned, text: {elem_text[:50]}...\n")
        
        # Tambahkan sebagai unaligned element dengan bbox dummy
        # Ini akan muncul di list tapi tidak punya bbox di PDF
        final_aligned.append({
            "text": elem_text,
            "matched_text": "",
            "bbox": {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
            "bboxes": [],
            "page": 0,
            "element_id": elem_id,
            "confidence": 0.0,
            "before_align_bboxes": [],
            "unaligned": True,
        })
    
    # Align images: ekstrak semua image dari content array
    image_items = []
    for elem in elements:
        json_tree = elem.delemen_json_tree
        if not json_tree:
            continue
        
        images = extract_all_images_recursive(json_tree)
        for idx, img in enumerate(images):
            rId = img.get('rId')
            if rId:
                image_items.append((elem.delemen_id, rId, elem.delemen_sequence, idx))
    
    if log_file:
        log_file.write(f"\nFound {len(image_items)} images in content arrays\n")
        log_file.flush()
    
    for elem_id, rId, sequence, content_idx in image_items:
        already_aligned = any(a['element_id'] == elem_id and a.get('rId') == rId and a.get('content_idx') == content_idx for a in final_aligned)
        if already_aligned:
            continue
        
        target_page = 0
        for prev_elem in elements:
            if prev_elem.delemen_sequence >= sequence:
                break
            for aligned in final_aligned:
                if aligned['element_id'] == prev_elem.delemen_id:
                    target_page = aligned['page']
        
        found_image = None
        search_pages = list(range(target_page, len(pdf_images))) + list(range(0, target_page))
        
        for page_idx in search_pages:
            if page_idx in pdf_images:
                for img in pdf_images[page_idx]:
                    img_key = (page_idx, img['xref'], img['img_index'], img['rect_index'])
                    if img_key not in used_images:
                        found_image = (page_idx, img)
                        used_images.add(img_key)
                        break
            if found_image:
                break
        
        if found_image:
            page_idx, img = found_image
            bbox = img['bbox']
            final_aligned.append({
                "text": "[IMAGE]",
                "matched_text": "[IMAGE]",
                "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                "bboxes": [{"page": page_idx, "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}}],
                "page": page_idx,
                "element_id": elem_id,
                "rId": rId,
                "content_idx": content_idx,
                "confidence": 0.8,
                "before_align_bboxes": [{"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}],
                "is_image": True,
            })

    if log_file:
        log_file.write(f"\nAligned elements: {len(final_aligned)}\n")
        log_file.write("=== END TRACE LOG ===\n\n")
        log_file.flush()
    sys.stderr.write(f"\nAligned elements: {len(final_aligned)}\n")
    sys.stderr.write("=== END TRACE LOG ===\n\n")
    sys.stderr.flush()

    return {
        "aligned_words": final_aligned,
        "stats": {
            "total_words": len(final_aligned),
            "assigned_words": len(final_aligned),
            "coverage": 1.0,
        },
    }
