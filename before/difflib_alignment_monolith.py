
"""difflib_alignment.py - Global difflib-based alignment between OpenXML and PDF"""

import difflib
import sys
import os
import json
import re
import unicodedata
from collections import defaultdict
import fitz  # PyMuPDF

# Fix encoding error on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Might be python < 3.7 or different env

# Token regex (dipakai untuk tokenisasi, tapi juga butuh posisi via finditer)
# PENTING: Pisahkan footnote (huruf besar + angka 1-2 digit di akhir), tapi gabung subscript (huruf kecil + angka)
# PENTING: Include underscore (_) di tengah identifier
TOKEN_RE = re.compile(r"\d*[A-Z\u00c0-\u00df\u0391-\u03a9][A-Za-z\u00c0-\u00ff\u0370-\u03ff_]*(?=\d{1,2}(?!\d))|\d*[A-Za-z\u00c0-\u00ff\u0370-\u03ff_]+\d*|\d+(?:\.\d+)*|[^\w\s]", flags=re.UNICODE)

# CONFIGURATION FOR FORMULA ALIGNMENT
FORMULA_EXPAND_VERT_TOP_RATIO = 0.5    # Expand top by 50% of avg height
FORMULA_EXPAND_VERT_BOTTOM_RATIO = 0.3 # Expand bottom by 30% of avg height
FORMULA_EXPAND_HORZ_RIGHT_RATIO = 0.2  # Expand right by 20% of width
FORMULA_EXPAND_HORZ_RIGHT_MAX_PX = 100 # Max expand right in pixels
FORMULA_EXPAND_HORZ_LEFT_RATIO = 0.15  # Expand left by 15% of width
FORMULA_EXPAND_HORZ_LEFT_MAX_PX = 60   # Max expand left in pixels
FORMULA_PAGE_MARGIN_LEFT = 50          # Safe left margin (approx for A4)
FORMULA_PAGE_MARGIN_RIGHT = 520        # Safe right margin (approx for A4)
FORMULA_MERGE_Y_THRESHOLD = 15         # Max vertical deviation from median line (px)
TABLE_MERGE_Y_MAX_GAP = 150            # Max vertical gap to consider cells part of same table cluster

# CONFIGURATION FOR BBOX MERGING
DEFAULT_MERGE_X_GAP = 2.0              # Max horizontal gap to merge tokens
DEFAULT_MERGE_Y_OVERLAP = 0.5          # Min vertical overlap ratio to merge tokens


def extract_pdf_table_cells(pdf_doc, page_idx):
    """Extract table cell bboxes from a PDF page using PyMuPDF.
    
    Args:
        pdf_doc: fitz.Document object
        page_idx: Page index (0-based)
    
    Returns:
        List of tables, each table is a dict with:
        - 'bbox': overall table bbox
        - 'cells': list of cell bboxes [(x0, y0, x1, y1), ...]
        - 'cell_texts': list of cell texts
        - 'rows': row count
        - 'cols': column count
    """
    if page_idx >= len(pdf_doc):
        return []
    
    page = pdf_doc[page_idx]
    tables_result = []
    
    try:
        tables = page.find_tables()
        
        for table in tables.tables:
            cells = table.cells
            cell_texts = []
            cell_bboxes = []
            
            for cell in cells:
                if cell:
                    cell_bboxes.append(cell)  # (x0, y0, x1, y1)
                    # Extract text from cell
                    cell_rect = fitz.Rect(cell)
                    cell_text = page.get_text("text", clip=cell_rect).strip()
                    cell_texts.append(cell_text)
                else:
                    cell_bboxes.append(None)
                    cell_texts.append("")
            
            tables_result.append({
                'bbox': table.bbox,
                'cells': cell_bboxes,
                'cell_texts': cell_texts,
                'rows': table.row_count,
                'cols': table.col_count
            })
    except Exception as e:
        # If table detection fails, return empty list
        pass
    
    return tables_result


def match_docx_cell_to_pdf_cell(docx_text, pdf_table_cells, pdf_cell_texts, exclude_indices=None):
    """Match a DOCX cell text to the best matching PDF cell.
    
    Args:
        docx_text: Text from DOCX cell
        pdf_table_cells: List of PDF cell bboxes
        pdf_cell_texts: List of PDF cell texts
        exclude_indices: Set of indices to ignore (already matched)
    
    Returns:
        Index of best matching PDF cell, or None if no good match
    """
    if not docx_text or not pdf_cell_texts:
        return None
    
    if exclude_indices is None:
        exclude_indices = set()
    
    import re
    
    def normalize(s):
        """Normalize text: lowercase, collapse whitespace, remove punctuation differences."""
        s = s.strip().lower()
        s = re.sub(r'\s+', ' ', s)  # Collapse all whitespace including newlines
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation for comparison
        return s
    
    docx_norm = normalize(docx_text)
    best_idx = None
    best_score = 0
    
    for i, pdf_text in enumerate(pdf_cell_texts):
        if i in exclude_indices:
            continue
            
        if not pdf_text:
            continue

        # HEIGHT HEURISTIC: Reject giant merged cells (e.g. height 600px for "Tabel 5.11")
        # Access bbox from parallel list
        cell_bbox = pdf_table_cells[i]
        cell_h = cell_bbox[3] - cell_bbox[1]
        
        # Estimate text lines (conservative: 50 chars/line)
        est_lines = max(1, len(pdf_text) / 50)
        est_h = est_lines * 30  # Generous 30px per line
        
        # If cell is tall (>200px) AND much taller than text metric (>5x), reject it
        if cell_h > 200 and cell_h > (est_h * 5):
            continue

        pdf_norm = normalize(pdf_text)
        
        # Exact match after normalization
        if docx_norm == pdf_norm:
            return i
        
        # Substring match (either direction)
        if docx_norm in pdf_norm or pdf_norm in docx_norm:
            score = min(len(docx_norm), len(pdf_norm)) / max(len(docx_norm), len(pdf_norm), 1)
            if score > best_score:
                best_score = score
                best_idx = i
                continue
        
        # Token overlap matching (for partial matches)
        docx_tokens = set(docx_norm.split())
        pdf_tokens = set(pdf_norm.split())
        
        if docx_tokens and pdf_tokens:
            overlap = len(docx_tokens & pdf_tokens)
            total = len(docx_tokens | pdf_tokens)
            token_score = overlap / total if total > 0 else 0
            
            # Require at least 60% token overlap
            if token_score > 0.6 and token_score > best_score:
                best_score = token_score
                best_idx = i
    
    # Return match only if score is good enough
    if best_score > 0.4:
        return best_idx
    
    return None


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


def merge_bboxes_token_level(items, x_gap: float = DEFAULT_MERGE_X_GAP, y_overlap_min: float = DEFAULT_MERGE_Y_OVERLAP, is_formula: bool = False):
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
        y_threshold = FORMULA_MERGE_Y_THRESHOLD
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


def normalize_text(s: str, preserve_whitespace: bool = False, is_formula: bool = False) -> str:
    """Normalisasi teks supaya ekstraksi DOCX dan PDF lebih mudah di-align.
    
    Args:
        s: Text to normalize
        preserve_whitespace: If True, don't collapse multiple spaces (for bbox mapping)
        is_formula: If True, apply formula-specific normalization
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
    
    # Apply formula-specific normalization if needed
    if is_formula:
        s = normalize_formula_text(s)
    
    if not preserve_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_formula_text(s: str) -> str:
    """Normalisasi khusus untuk teks formula agar lebih mudah di-match dengan PDF.
    
    Masalah: OpenXML SDK mengekstrak formula tanpa spasi yang proper:
        DOCX: "μLO0.04, 9.12,0.13=e-120.04-0.139.122 =1"
        PDF:  "μLO (0.04, 9.12,0.13) = e-1"
    
    Solusi: Tambahkan spasi untuk memisahkan bagian-bagian formula
    """
    if not s:
        return s
    
    # 1. Tambah spasi setelah Greek letter + subscript (μLO, μMD, μHI, dll)
    # μLO0.04 -> μLO 0.04
    s = re.sub(r'([μαβγδε][A-Z]{1,3})(\d)', r'\1 \2', s)
    
    # 2. Tambah spasi di sekitar operator = (tapi hindari <=, >=, !=)
    # Jangan double spasi jika sudah ada
    s = re.sub(r'(?<![<>!=\s])=(?![=\s])', r' = ', s)
    
    # 3. Tambah spasi setelah koma jika tidak ada
    s = re.sub(r',(?!\s)', r', ', s)
    
    # 4. Handle fraction/exponent content yang tidak ada di PDF
    # Pattern: e-12... -> e - 1 2 (memisahkan "1" dan "2" yang merupakan numerator dan denominator)
    # Gaussian formula: e^(-1/2 * ...) di PDF cuma muncul sebagai "e - 1" karena fraction konten terpisah
    # Step 4a: Pisahkan "e-12" atau "e-1 2" menjadi "e - 1 2"
    s = re.sub(r'([eE])\s*-\s*12', r'\1 - 1 2', s)
    s = re.sub(r'([eE])\s*-\s*1\s+2', r'\1 - 1 2', s)
    
    # 4b: Setelah "e - 1 2", kemungkinan ada content fraction seperti "(x-μ)/σ"
    # Yang di-extract jadi angka gabungan seperti "0.04-0.13" atau "0.04-0.139.122"
    # Pisahkan dengan spasi
    s = re.sub(r'(\d)(\d+\.\d+)-', r'\1 \2 - ', s)  # "20.04-" -> "2 0.04 - "
    
    # 4c: Split Latin character followed immediately by Greek character
    # alpha n z n alpha 1 -> ...n alpha...
    # Case: "znα1" -> "zn α1"
    s = re.sub(r'([a-zA-Z0-9])([μαβγδεΑ-Ω])', r'\1 \2', s)
    
    # 4e: Split Greek character from following lowercase Latin (handling subscripts like αn)
    # But preserve Uppercase (variable names like μLO)
    # αn -> α n
    # μLO -> μLO (unchanged)
    s = re.sub(r'([μαβγδεΑ-Ω])([a-z])', r'\1 \2', s)
    
    # 4d: Normalize glued uppercase variables followed by equals
    # Pattern: WA= -> WA =, atau F= -> F =, atau GPM= -> GPM =
    # Menangani variabel rumus yang menempel dengan operator
    s = re.sub(r'([A-Z]{1,10})(=)', r'\1 \2', s)
    
    # 5. Pisahkan angka digit tunggal yang menempel
    # Pattern: "2 0.04" sudah OK, "20.04" harus jadi "2 0.04" jika setelah "e - 1 2"
    # Tapi jangan affect angka normal seperti "20" atau "120"
    
    # 6. Handle subscript/superscript yang digabung
    # Pattern: 0.139.122 (denominator + exponent 2) -> pisahkan ke "0.13 9.12 2"
    # Ini tricky karena bisa jadi "0.13" dan "9.12" adalah angka terpisah dengan "2" sebagai exponent
    s = re.sub(r'(\d+\.\d{2})(\d\.\d+)', r'\1 \2', s)  # 0.139.12 -> 0.13 9.12
    s = re.sub(r'(\d+\.\d+)(\d{3})$', r'\1 \2', s)  # ...122 -> ... 122 (jika di akhir)
    s = re.sub(r'(\d+\.\d+)(\d{3})\s', r'\1 \2 ', s)  # ...122 -> ... 122 (jika ada spasi setelah)
    
    # 7. Normalize multiple spaces
    s = re.sub(r'\s+', ' ', s)
    
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


def tokenize(s: str, is_formula: bool = False):
    """Tokenizer untuk alignment.
    
    Args:
        s: Text to tokenize
        is_formula: If True, apply formula-specific normalization
    """
    s = normalize_text(s, is_formula=is_formula)
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


def extract_text_from_json_tree(json_tree, return_cells=False, return_shapes=False, return_image_only_cells=False):
    """Ekstrak teks dari dokumen_elemen_json_tree (OpenXML).
    
    Args:
        json_tree: JSON tree dari OpenXML
        return_cells: Jika True dan ini table, return list of cell texts
        return_shapes: Jika True dan content adalah array of shapes, return list of shape texts
        return_image_only_cells: Jika True, return tuple (cells, image_only_cells) dimana
                                 image_only_cells adalah list of (cell_index, [rIds]) untuk
                                 cells yang HANYA berisi image tanpa text
    
    Returns:
        String teks gabungan, atau list of cell texts jika return_cells=True dan ada table,
        atau list of shape texts jika return_shapes=True dan ada shapes,
        atau tuple (cells, image_only_cells) jika return_image_only_cells=True
    """
    if not json_tree:
        if return_image_only_cells:
            return ([], [])
        return "" if not (return_cells or return_shapes) else []

    # Unwrap content wrapper
    if isinstance(json_tree, dict) and "content" in json_tree and isinstance(json_tree["content"], dict):
        json_tree = json_tree["content"]
    
    # Cek apakah ini content array dengan multiple shapes (bukan table)
    if return_shapes and isinstance(json_tree, dict) and "content" in json_tree:
        content = json_tree["content"]
        if isinstance(content, list) and len(content) > 0:
            # Check if all items are shapes
            has_shapes = any(isinstance(item, dict) and item.get('type') == 'shape' for item in content)
            if has_shapes:
                shapes = []
                shape_index = 0
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'shape':
                        # GRANULAR SPLIT: Check if shape has content list (text boxes, etc)
                        # Instead of merging all text, yield them separately for better alignment
                        if "content" in item and isinstance(item["content"], list):
                            sub_idx = 0
                            has_sub_splits = False
                            for sub_item in item["content"]:
                                if isinstance(sub_item, dict):
                                    # Extract text from this specific sub-item
                                    # Recursive call handles wrapped content
                                    sub_text = extract_text_from_json_tree(sub_item, return_cells=False, return_shapes=False)
                                    if sub_text and sub_text.strip():
                                        # Use hierarchical index: "{main_idx}_{sub_idx}"
                                        # Pass 'item' (parent shape) as the 3rd arg for context if needed
                                        shapes.append((f"{shape_index}_{sub_idx}", sub_text, item))
                                        sub_idx += 1
                                        has_sub_splits = True
                            
                            if not has_sub_splits:
                                # Fallback: No granular content found, treat as whole
                                shape_text = extract_text_from_json_tree(item, return_cells=False, return_shapes=False)
                                if shape_text and shape_text.strip():
                                    shapes.append((shape_index, shape_text, item))
                        else:
                            # Standard shape (no content list or other type)
                            shape_text = extract_text_from_json_tree(item, return_cells=False, return_shapes=False)
                            if shape_text and shape_text.strip():
                                shapes.append((shape_index, shape_text, item))
                        
                        shape_index += 1
                return shapes
    
    # Cek apakah ini table
    if isinstance(json_tree, dict) and "rows" in json_tree:
        cells = []
        image_only_cells = []  # Track cells yang HANYA berisi image
        cell_index = 0
        for row in json_tree.get("rows", []):
            if isinstance(row, dict):
                for cell in row.get("cells", []):
                    if isinstance(cell, str):
                        cells.append((cell_index, cell))
                        cell_index += 1
                    elif isinstance(cell, list):
                        # Empty cell
                        if len(cell) == 0:
                            cell_index += 1
                            continue
                        
                        # Cell is array of content items
                        cell_texts = []
                        has_shape = False
                        has_image = False
                        image_rIds = []  # Track rIds of images in this cell
                        for item in cell:
                            if isinstance(item, dict):
                                if item.get("type") == "shape":
                                    has_shape = True
                                    if "content" in item:
                                        item_text = extract_text_from_json_tree(item, return_cells=False, return_shapes=False)
                                        if item_text and item_text.strip():
                                            cells.append((cell_index, item_text))
                                    cell_index += 1
                                elif item.get("type") == "image":
                                    has_image = True
                                    rId = item.get("rId")
                                    if rId:
                                        image_rIds.append(rId)
                                    # TIDAK lagi append "[IMAGE]" ke cell_texts
                                elif item.get("type") == "text" and "value" in item:
                                    cell_texts.append(item["value"])
                        
                        if not has_shape:
                            # Regular cell with text content
                            if cell_texts:
                                # Cell dengan text (mungkin juga ada image)
                                cells.append((cell_index, " ".join(cell_texts)))
                            elif has_image:
                                # Cell HANYA berisi image tanpa text
                                # Track secara terpisah untuk direct image alignment
                                image_only_cells.append((cell_index, image_rIds))
                            # Note: empty cells are not added to cells list
                            cell_index += 1
                    elif isinstance(cell, dict):
                        cell_text = extract_text_from_json_tree(cell, return_cells=False, return_shapes=False)
                        if cell_text:
                            cells.append((cell_index, cell_text))
                        cell_index += 1
        
        if return_image_only_cells:
            return (cells, image_only_cells)
        if return_cells:
            return cells
        return " ".join(text for _, text in cells)

    texts = []

    def rec(node):
        if isinstance(node, dict):
            if "rows" in node:
                for row in node.get("rows", []):
                    if isinstance(row, dict):
                        for cell in row.get("cells", []):
                            if isinstance(cell, str):
                                texts.append(cell)
                            elif isinstance(cell, dict):
                                rec(cell)
                return

            if node.get("type") == "text" and "value" in node:
                texts.append(node["value"])
            elif node.get("type") == "math" and "text" in node:
                # Formula: normalize untuk match dengan PDF
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
    return result


def get_pdf_images(pdf_doc):
    """
    Ekstrak semantic images dan vector drawings dari PDF.
    
    Structure:
    {
        page_index: [
            {'bbox': [x0, y0, x1, y1], 'xref': xref_or_dummy_id, 'type': 'image'|'vector'}
        ]
    }
    """
    pdf_images = {}
    
    for page_idx in range(pdf_doc.page_count):
        page = pdf_doc[page_idx]
        images_on_page = []
        
        # 1. Standard Images (Raster)
        try:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                # Get rects where this image is drawn
                rects = page.get_image_rects(xref)
                for r in rects:
                    images_on_page.append({
                        'bbox': [r.x0, r.y0, r.x1, r.y1],
                        'xref': xref,
                        'type': 'image'
                    })
        except Exception as e:
            print(f"Error getting images on page {page_idx}: {e}")
            
        # 2. Vector Drawings (clustered)
        # Only use this if the page has significant vector content
        try:
            drawings = page.get_drawings()
            if drawings and len(drawings) > 20: 
                # Clustering algorithm
                # 1. Filter valid drawing rects
                rects = []
                for d in drawings:
                    r = d['rect']
                    w, h = r.width, r.height
                    if w < 2 and h < 2: continue # Ignore dots
                    if w > 500 and h > 800: continue # Ignore page borders
                    
                    # Ignore likely table borders (long thin lines)
                    # Horizontal lines: Large Width, Small Height
                    if w > 50 and h < 5: continue
                    # Vertical lines: Small Width, Large Height
                    if h > 50 and w < 5: continue
                    
                    rects.append([r.x0, r.y0, r.x1, r.y1])
                
                # 2. Iterative box merging
                # Simple greedy approach: 
                # - Take a box, find all overlapping/close boxes in remaining set. 
                # - Merge them. Repeat until no more overlapping.
                # - Move to next unmerged box.
                
                clusters = []
                THRESHOLD = 15.0 # pixels gap allowed to merge
                
                while rects:
                    # Pop the first rect to start a cluster
                    current_cluster = rects.pop(0)
                    changed = True
                    
                    while changed:
                        changed = False
                        # Find all rects that overlap/close to current_cluster
                        i = 0
                        while i < len(rects):
                            r = rects[i]
                            # Check overlap/proximity with current_cluster
                            # Expand current_cluster by threshold for checking
                            c_x0, c_y0, c_x1, c_y1 = current_cluster
                            r_x0, r_y0, r_x1, r_y1 = r
                            
                            # Intersection check with sensitivity
                            # gap x
                            gap_x = max(0, r_x0 - c_x1, c_x0 - r_x1)
                            # gap y
                            gap_y = max(0, r_y0 - c_y1, c_y0 - r_y1)
                            
                            if gap_x <= THRESHOLD and gap_y <= THRESHOLD:
                                # Merge r into current_cluster
                                current_cluster[0] = min(c_x0, r_x0)
                                current_cluster[1] = min(c_y0, r_y0)
                                current_cluster[2] = max(c_x1, r_x1)
                                current_cluster[3] = max(c_y1, r_y1)
                                
                                # Remove r from rects (it's merged)
                                rects.pop(i)
                                changed = True
                                # Don't increment i, as we popped current index
                            else:
                                i += 1
                    
                    # Validate cluster size
                    c_w = current_cluster[2] - current_cluster[0]
                    c_h = current_cluster[3] - current_cluster[1]
                    
                    # Only accept clusters of significant size (e.g. icon/chart size)
                    if c_w > 20 and c_h > 20:
                        clusters.append(current_cluster)
                
                # 3. Add clusters as vector images
                for i, c in enumerate(clusters):
                     # Unique XREF for each cluster on the page
                     # Use (page_idx * 1000 + i + 5000) * -1 to avoid collision
                     dummy_xref = -(page_idx * 10000 + i + 5000)
                     images_on_page.append({
                        'bbox': c,
                        'xref': dummy_xref,
                        'type': 'vector'
                     })
                     
        except Exception as e:
            print(f"Error getting drawings on page {page_idx}: {e}")
            import traceback
            traceback.print_exc()

        if images_on_page:
            pdf_images[page_idx] = images_on_page
            
    return pdf_images


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


    # Extract images using helper
    with fitz.open(pdf_path) as pdf:
        pdf_images = get_pdf_images(pdf)
    docx_tokens = []
    docx_owner = []
    docx_cell_index = []  # Track cell index untuk table
    docx_is_formula = []  # Track apakah token dari formula
    
    # Track image-only cells untuk direct image alignment
    # Format: {elem_id: [(cell_index, [rIds]), ...]}
    image_only_cells_map = {}
    
    # Track used PDF cells to prevent duplicate assignment across tables
    # Set of (page_num, table_idx, cell_idx)
    used_pdf_cells = set()

    for elem in elements:
        elem_text = extract_text_from_json_tree(elem.delemen_json_tree)
        
        # Cek apakah ini table - gunakan return_image_only_cells untuk track image-only cells
        cells_result = extract_text_from_json_tree(elem.delemen_json_tree, return_cells=True, return_image_only_cells=True)
        if isinstance(cells_result, tuple):
            cells, image_only_cells = cells_result
        else:
            cells = cells_result
            image_only_cells = []
        
        is_table = isinstance(cells, list) and len(cells) > 0
        
        # Simpan image-only cells untuk diproses nanti
        if image_only_cells:
            image_only_cells_map[elem.delemen_id] = image_only_cells
        
        # Cek apakah ini content array dengan shapes
        shapes = extract_text_from_json_tree(elem.delemen_json_tree, return_shapes=True)
        is_shape_array = isinstance(shapes, list) and len(shapes) > 0
        
        # Cek apakah mengandung formula
        has_formula = has_formula_in_tree(elem.delemen_json_tree)
        
        if is_table:
            # Table: tokenize per cell
            for cell_idx, cell_text in cells:
                toks = tokenize(cell_text, is_formula=has_formula)
                docx_tokens.extend(toks)
                docx_owner.extend([elem.delemen_id] * len(toks))
                docx_cell_index.extend([cell_idx] * len(toks))
                docx_is_formula.extend([has_formula] * len(toks))
        elif is_shape_array:
            # Shape array: tokenize per shape
            for shape_idx, shape_text, _ in shapes:
                toks = tokenize(shape_text, is_formula=has_formula)
                docx_tokens.extend(toks)
                docx_owner.extend([elem.delemen_id] * len(toks))
                docx_cell_index.extend([shape_idx] * len(toks))  # Use shape_idx like cell_idx
                docx_is_formula.extend([has_formula] * len(toks))
        else:
            # Non-table: tokenize biasa
            toks = tokenize(elem_text, is_formula=has_formula)
            docx_tokens.extend(toks)
            docx_owner.extend([elem.delemen_id] * len(toks))
            docx_cell_index.extend([-1] * len(toks))  # -1 = bukan table/shape array
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

    # Global alignment dengan constraint monotonic
    sm = difflib.SequenceMatcher(a=docx_tokens, b=pdf_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    docx_to_pdf = [None] * len(docx_tokens)
    docx_to_pdf_multi = {}  # Track 1 DOCX token -> multiple PDF tokens
    last_pdf_idx = -1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                pdf_idx = j1 + k
                if pdf_idx > last_pdf_idx:
                    docx_to_pdf[i1 + k] = pdf_idx
                    last_pdf_idx = pdf_idx
        elif tag == "replace" and i2 - i1 == 1 and j2 - j1 > 1:
            # WORD WRAP FIX: 1 DOCX token match ke multiple PDF tokens
            docx_token = docx_tokens[i1]
            pdf_segment = pdf_tokens[j1:j2]
            
            # Cek apakah PDF tokens bisa digabung jadi DOCX token
            combined = "".join(pdf_segment)
            if combined == docx_token or combined.replace("-", "") == docx_token:
                # Match! Assign semua PDF tokens ke DOCX token ini
                pdf_indices = list(range(j1, j2))
                if all(idx > last_pdf_idx for idx in pdf_indices):
                    docx_to_pdf[i1] = j1  # Primary assignment
                    docx_to_pdf_multi[i1] = pdf_indices  # Track all PDF tokens
                    last_pdf_idx = j2 - 1

    # PRE-COMPUTE: Group PDF tokens by page for fast spatial lookup
    pdf_on_page = defaultdict(list)
    for idx, (bbox, page_idx) in enumerate(zip(pdf_bboxes, pdf_pages)):
        pdf_on_page[page_idx].append((idx, bbox))
    
    # Build aligned_words
    element_groups = defaultdict(lambda: defaultdict(list))  # elem_id -> cell_idx -> words
    element_texts = {}  # Store RAW text without normalization
    element_is_table = {}
    element_is_shape_array = {}
    element_cell_texts = {}  # elem_id -> list of cell texts (RAW, tidak dinormalisasi)
    element_shape_data = {}  # elem_id -> {shape_idx: (text, shape_item)}

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
        element_is_table[elem.delemen_id] = isinstance(cells, list) and len(cells) > 0
        if element_is_table[elem.delemen_id]:
            # Store RAW cell texts as dict {cell_idx: text}
            element_cell_texts[elem.delemen_id] = {idx: text for idx, text in cells}
        
        # Check for shape array
        shapes = extract_text_from_json_tree(elem.delemen_json_tree, return_shapes=True)
        element_is_shape_array[elem.delemen_id] = isinstance(shapes, list) and len(shapes) > 0
        if element_is_shape_array[elem.delemen_id]:
            # Store shape data as dict {shape_idx: (text, shape_item)}
            element_shape_data[elem.delemen_id] = {idx: (text, shape_item) for idx, text, shape_item in shapes}

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

    
    # -------------------------------------------------------
    # PHASE 2b: INDEPENDENT ALIGNMENT FOR SHAPE ARRAYS (Fix for Page 11)
    # -------------------------------------------------------
    # Collect currently used PDF indices
    used_pdf_indices = set()
    for elem_id in element_groups:
        for cell_idx in element_groups[elem_id]:
            for w in element_groups[elem_id][cell_idx]:
                if 'pdf_index' in w:
                    used_pdf_indices.add(w['pdf_index'])
    
    # helper for localized matching
    def find_best_sequence_match(target_tokens, candidate_indices, pdf_tokens):
        if not target_tokens: 
            return None, 0
        
        best_match = None
        best_score = 0.0
        
        # Simple sliding window if target is small, or SequenceMatcher if larger
        # Using SequenceMatcher for robustness against OCR errors
        target_str = " ".join(target_tokens)
        
        # Optimize: distinct candidates might be scattered. 
        # But commonly shapes are contiguous.
        # We try to find contiguous blocks in candidates that match.
        
        # Group candidates into contiguous segments (allowing small gaps)
        segments = []
        if not candidate_indices:
            return None, 0
            
        current_seg = [candidate_indices[0]]
        for i in range(1, len(candidate_indices)):
            if candidate_indices[i] == candidate_indices[i-1] + 1:
                current_seg.append(candidate_indices[i])
            else:
                if len(current_seg) >= len(target_tokens) * 0.5: # Only consider segments of reasonable size
                    segments.append(current_seg)
                current_seg = [candidate_indices[i]]
        segments.append(current_seg)
        
        for seg in segments:
            seg_tokens = [pdf_tokens[i] for i in seg]
            sm_local = difflib.SequenceMatcher(None, target_tokens, seg_tokens, autojunk=False)
            match = sm_local.find_longest_match(0, len(target_tokens), 0, len(seg_tokens))
            
            if match.size > 0:
                # Calculate coverage
                score = match.size / len(target_tokens)
                if score > best_score and score > 0.6: # Threshold 0.6
                    best_score = score
                    # Map back to global indices
                    matched_indices = seg[match.b : match.b + match.size]
                    best_match = matched_indices
        
        return best_match, best_score

    # Check all shape arrays
    for elem_id, is_shape_array in element_is_shape_array.items():
        if not is_shape_array:
            continue
            
        shape_data = element_shape_data.get(elem_id, {})
        for shape_idx, (shape_text, shape_item) in shape_data.items():
            # Check if this shape is already aligned
            if element_groups[elem_id][shape_idx]:
                continue
            
            # Not aligned - try separate alignment
            shape_tokens = tokenize(shape_text)
            if not shape_tokens:
                continue
                
            # Get candidate PDF tokens (unused)
            # Optimization: Filter by page? 
            # We don't know the page, but we can guess from other shapes in the same array
            target_pages = set()
            for other_idx, words in element_groups[elem_id].items():
                if words:
                    target_pages.add(words[0]['page'])
            
            # If no other shapes aligned, fallback to previous element's page
            if not target_pages:
                 # Logic to find prev element page... simplified: scan all pages or just existing heuristic
                 # For now, scan ALL unused tokens (slow but safe) or restrict to +/- 1 page of "current" context
                 # Let's scan ALL for robustness on small docs like 227
                 pass
            
            # Collect candidate indices (global) that are NOT used
            # Filtering by page if we have hints might be better, but let's try global first
            candidate_indices = [i for i in range(len(pdf_tokens)) if i not in used_pdf_indices]
            
            # Refine candidates: only on pages where we expect the content?
            # If we have target_pages, prioritize/restrict to them
            if target_pages:
                # Add +/- 1 page
                expanded_pages = set()
                for p in target_pages:
                    expanded_pages.add(p)
                    expanded_pages.add(p-1)
                    expanded_pages.add(p+1)
                candidate_indices = [i for i in candidate_indices if pdf_pages[i] in expanded_pages]
            
            if not candidate_indices:
                continue

            matched_indices, score = find_best_sequence_match(shape_tokens, candidate_indices, pdf_tokens)
            
            if matched_indices:
                if log_file:
                    log_file.write(f"Independent Align Shape {elem_id}_{shape_idx}: '{shape_text}' matched {len(matched_indices)} tokens (score {score:.2f})\n")
                
                # Add to element_groups
                for pdf_idx in matched_indices:
                    bbox = pdf_bboxes[pdf_idx]
                    element_groups[elem_id][shape_idx].append({
                        "text": pdf_tokens[pdf_idx],
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "page": pdf_pages[pdf_idx],
                        "is_formula": False,
                        "pdf_index": pdf_idx,
                        "docx_token_index": -1, # Independent
                    })
                    used_pdf_indices.add(pdf_idx)
    for elem_id in element_groups:
        for cell_idx in element_groups[elem_id]:
            words = element_groups[elem_id][cell_idx]
            if len(words) <= 1:
                continue
            
            # Cek apakah ini formula
            is_formula = any(w.get('is_formula', False) for w in words)
            if not is_formula:
                continue
            
            # Hitung Y midpoint untuk setiap token dan sort
            word_y_pairs = [(w, (w['bbox']['y0'] + w['bbox']['y1']) / 2) for w in words]
            word_y_pairs.sort(key=lambda x: x[1])  # Sort by Y
            
            # Hitung tinggi font rata-rata
            heights = [w['bbox']['y1'] - w['bbox']['y0'] for w in words]
            avg_height = sum(heights) / len(heights) if heights else 15
            
            # Cari gap yang signifikan (> 2x avg_height = kemungkinan formula berbeda)
            # Gap 2x karena formula dengan pecahan bisa punya gap 1.5x untuk fraction line
            gap_threshold = avg_height * 2.0
            
            cutoff_y = None
            for i in range(len(word_y_pairs) - 1):
                current_y = word_y_pairs[i][1]
                next_y = word_y_pairs[i + 1][1]
                gap = next_y - current_y
                
                if gap > gap_threshold:
                    # Gap besar ditemukan - token setelah ini milik elemen lain
                    cutoff_y = current_y + gap_threshold / 2  # Cutoff di tengah gap
                    break
            
            if cutoff_y is not None:
                # Filter token yang berada sebelum cutoff
                filtered_words = [w for w, y in word_y_pairs if y <= cutoff_y]
                
                if len(filtered_words) < len(words):
                    element_groups[elem_id][cell_idx] = filtered_words

    # List of fully aligned elements
    final_aligned = []
    
    # Also track completely unaligned tables to populate unmapped_table_cells
    aligned_elem_ids = set()
    unmapped_table_cells = []
    
    # Track which PDF images have been used
    used_images = set()

    for elem_id, cell_groups in element_groups.items():
        aligned_elem_ids.add(elem_id)
        
        is_table = element_is_table.get(elem_id, False)
        is_shape_array = element_is_shape_array.get(elem_id, False)
        
        if is_shape_array:
            # Shape array: buat elemen terpisah per shape
            shape_order = []
            for shape_idx, words in cell_groups.items():
                if words:
                    first_pdf_idx = min(w.get('pdf_index', float('inf')) for w in words)
                    shape_order.append((first_pdf_idx, shape_idx))
            shape_order.sort()
            sorted_shape_indices = [shape_idx for _, shape_idx in shape_order]
            
            # Add parent container first
            if sorted_shape_indices:
                first_shape_words = cell_groups[sorted_shape_indices[0]]
                all_shape_words = []
                for shape_idx in sorted_shape_indices:
                    all_shape_words.extend(cell_groups[shape_idx])
                
                if all_shape_words:
                    container_x0 = min(w["bbox"]["x0"] for w in all_shape_words)
                    container_y0 = min(w["bbox"]["y0"] for w in all_shape_words)
                    container_x1 = max(w["bbox"]["x1"] for w in all_shape_words)
                    container_y1 = max(w["bbox"]["y1"] for w in all_shape_words)
                    
                    final_aligned.append({
                        "text": element_texts.get(elem_id, ""),
                        "matched_text": "",
                        "bbox": {"x0": container_x0, "y0": container_y0, "x1": container_x1, "y1": container_y1},
                        "bboxes": [],
                        "page": first_shape_words[0]["page"],
                        "element_id": elem_id,
                        "confidence": 1.0,
                        "before_align_bboxes": [],
                        "is_shape_container": True,
                        "children": [],
                    })
            
            for shape_idx in sorted_shape_indices:
                words = cell_groups[shape_idx]
                if not words:
                    continue
                
                # Get original shape text from DOCX
                shape_text = ""
                shape_item = None
                if elem_id in element_shape_data and shape_idx in element_shape_data[elem_id]:
                    shape_text, shape_item = element_shape_data[elem_id][shape_idx]
                
                # Fallback: jika shape_text kosong, gunakan matched_text
                matched_text = " ".join(w["text"] for w in words)
                if not shape_text or not shape_text.strip():
                    shape_text = matched_text
                
                x0 = min(w["bbox"]["x0"] for w in words)
                y0 = min(w["bbox"]["y0"] for w in words)
                x1 = max(w["bbox"]["x1"] for w in words)
                y1 = max(w["bbox"]["y1"] for w in words)
                
                merged_segments = merge_bboxes_token_level(words, is_formula=False)
                
                # Extract nested content from shape
                content_items = []
                content_bboxes = []
                if shape_item and 'content' in shape_item and isinstance(shape_item['content'], list):
                    for item in shape_item['content']:
                        if isinstance(item, dict):
                            if item.get('type') == 'text' and 'value' in item:
                                content_items.append({'type': 'text', 'value': item['value']})
                                if merged_segments:
                                    content_bboxes.append(merged_segments[0]['bbox'])
                            elif item.get('type') == 'image':
                                content_items.append({'type': 'image', 'rId': item.get('rId')})
                                content_bboxes.append(None)
                
                shape_elem = {
                    "text": shape_text,
                    "matched_text": matched_text,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "bboxes": merged_segments,
                    "page": words[0]["page"],
                    "element_id": f"{elem_id}_shape_{shape_idx}",
                    "parent_element_id": elem_id,
                    "confidence": 1.0,
                    "before_align_bboxes": [w["bbox"] for w in words],
                    "content_items": content_items if content_items else None,
                    "content_bboxes": content_bboxes if content_bboxes else None,
                }
                final_aligned.append(shape_elem)
                
                # Add to parent's children list
                for parent in final_aligned:
                    if parent.get('element_id') == elem_id and parent.get('is_shape_container'):
                        parent['children'].append(f"{elem_id}_shape_{shape_idx}")
                        break
        elif is_table:
            # FIX: Sort cells by PDF reading order
            cell_order = []
            for cell_idx, words in cell_groups.items():
                if words:
                    first_pdf_idx = min(w.get('pdf_index', float('inf')) for w in words)
                    cell_order.append((first_pdf_idx, cell_idx))
            cell_order.sort()
            sorted_cell_indices = [cell_idx for _, cell_idx in cell_order]
            
            # Add table container first - calculate bbox from cells on SAME PAGE only
            if sorted_cell_indices:
                # Group cells by page
                cells_by_page = defaultdict(list)
                for cell_idx in sorted_cell_indices:
                    words = cell_groups[cell_idx]
                    if words:
                        page = words[0]['page']
                        cells_by_page[page].append(cell_idx)
                
                # Use the page with most cells
                if cells_by_page:
                    main_page = max(cells_by_page.keys(), key=lambda p: len(cells_by_page[p]))
                    main_page_cells = cells_by_page[main_page]
                    
                    # Calculate table bbox from cells on main page only
                    cell_bboxes = []
                    for cell_idx in main_page_cells:
                        words = cell_groups[cell_idx]
                        if words:
                            x0 = min(w["bbox"]["x0"] for w in words)
                            y0 = min(w["bbox"]["y0"] for w in words)
                            x1 = max(w["bbox"]["x1"] for w in words)
                            y1 = max(w["bbox"]["y1"] for w in words)
                            cell_bboxes.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
                    
                    if cell_bboxes:
                        # FILTER OUTLIERS: Table content should be roughly contiguous vertically
                        # If we have a massive vertical gap, it's likely a misaligned cell (e.g. at page top/bottom)
                        
                        # Sort by Y0 center
                        sorted_bboxes = sorted(cell_bboxes, key=lambda b: (b['y0'] + b['y1']) / 2)
                        
                        if len(sorted_bboxes) > 1:
                            clusters = []
                            current_cluster = [sorted_bboxes[0]]
                            
                            for i in range(1, len(sorted_bboxes)):
                                prev = sorted_bboxes[i-1]
                                curr = sorted_bboxes[i]
                                
                                # Calculate vertical distance between current top and previous bottom
                                gap = curr['y0'] - prev['y1']
                                
                                if gap > TABLE_MERGE_Y_MAX_GAP:
                                    clusters.append(current_cluster)
                                    current_cluster = []
                                
                                current_cluster.append(curr)
                            
                            clusters.append(current_cluster)
                            
                            # Use the largest cluster (most cells)
                            if len(clusters) > 1:
                                main_cluster = max(clusters, key=len)
                                cell_bboxes = main_cluster
                                # Log if needed
                                # if log_file: log_file.write(f"Table {elem_id}: Filtered {len(sorted_bboxes)-len(cell_bboxes)} outlier cells from container bbox\n")

                        table_x0 = min(b['x0'] for b in cell_bboxes)
                        table_y0 = min(b['y0'] for b in cell_bboxes)
                        table_x1 = max(b['x1'] for b in cell_bboxes)
                        table_y1 = max(b['y1'] for b in cell_bboxes)
                        
                        # first_cell_words = cell_groups[main_page_cells[0]] # This line was removed as it's not used
                        
                        # POST-PROCESSING: Grid Alignment
                        # Align X coordinates of cells in the same column to fix "jaggy" borders
                        # caused by mix of PyMuPDF bboxes and token-based fallback bboxes.
                        def align_table_columns_x(table_cells, table_id):
                            if not table_cells:
                                return
                            # Debug log for columns
                            debug_log = open('alignment_columns_debug.log', 'a', encoding='utf-8')
                            debug_log.write(f"Processing table {table_id} with {len(table_cells)} cells\n")
                                
                            # 1. Group cells into potential columns based on x-center
                            # Simple clustering by x-center proximity
                            cols = []
                            # Sort by x-center
                            sorted_cells = sorted(table_cells, key=lambda c: (c['bbox']['x0'] + c['bbox']['x1'])/2)
                            
                            current_col = []
                            if sorted_cells:
                                current_col = [sorted_cells[0]]
                                last_cnt = (sorted_cells[0]['bbox']['x0'] + sorted_cells[0]['bbox']['x1'])/2
                                
                                for i in range(1, len(sorted_cells)):
                                    curr = sorted_cells[i]
                                    curr_cnt = (curr['bbox']['x0'] + curr['bbox']['x1'])/2
                                    
                                    debug_log.write(f"  Cell {i} '{curr['text'][:10]}' Cnt={curr_cnt:.1f} Last={last_cnt:.1f} Diff={abs(curr_cnt-last_cnt):.1f}\n")
                                    
                                    # INCREASED THRESHOLD to 40px to catch alignment drifts
                                    if abs(curr_cnt - last_cnt) < 40:
                                        current_col.append(curr)
                                        # Update centroid (simplistic moving average)
                                        last_cnt = (last_cnt * (len(current_col)-1) + curr_cnt) / len(current_col)
                                    else:
                                        cols.append(current_col)
                                        current_col = [curr]
                                        last_cnt = curr_cnt
                                cols.append(current_col)
                            
                            debug_log.write(f"Found {len(cols)} columns\n")
                            
                            # 2. For each column, calculate unified Width (min x0, max x1)
                            # But ignore outliers (e.g. colspan cells, merged headers)
                            for col_idx, col in enumerate(cols):
                                if not col: continue
                                
                                # Calculate widths
                                widths = [c['bbox']['x1'] - c['bbox']['x0'] for c in col]
                                if not widths: continue
                                
                                # Median width
                                sorted_widths = sorted(widths)
                                median_width = sorted_widths[len(widths)//2]
                                
                                # Identify outliers (cells much wider than median, e.g. > 1.5x)
                                # Or cells much narrower (though less critical)
                                normal_cells = []
                                outliers = []
                                
                                for c in col:
                                    w = c['bbox']['x1'] - c['bbox']['x0']
                                    # Tolerance: 1.5x for merged cells.
                                    # Also merged cells usually have diff > 40px from median?
                                    if w > median_width * 1.5:
                                        outliers.append(c)
                                    else:
                                        normal_cells.append(c)
                                
                                if not normal_cells:
                                    # If all are outliers (unlikely) or single cell, just keep them as is?
                                    # Or if single cell, it is the column.
                                    if len(col) == 1:
                                        continue # Don't touch single cell columns? Or alignment is irrelevant.
                                    # If mixed widths but all 'wide'?
                                    # Fallback: align everything if no normal cells found (very rare)
                                    normal_cells = col

                                # Gather coords ONLY from normal cells
                                x0s = [c['bbox']['x0'] for c in normal_cells]
                                x1s = [c['bbox']['x1'] for c in normal_cells]
                                
                                # Use min X0 and max X1 (expand to widest NORMAL cell)
                                unified_x0 = min(x0s)
                                unified_x1 = max(x1s)
                                
                                debug_log.write(f"  Col {col_idx}: {len(col)} cells ({len(normal_cells)} normal). Median W={median_width:.1f}. Unifying to x={unified_x0:.0f}-{unified_x1:.0f}\n")
                                
                                # Apply to NORMAL cells only
                                # Outliers (merged cells) keep their original PyMuPDF bbox
                                for c in normal_cells:
                                    c['bbox']['x0'] = unified_x0
                                    c['bbox']['x1'] = unified_x1
                                
                                for c in outliers:
                                    debug_log.write(f"    Skipping outlier cell '{c['text'][:10]}' W={c['bbox']['x1']-c['bbox']['x0']:.1f}\n")
                            
                            # 3. GLOBAL HEADER FIX
                            # If we have detected table bounds from columns, force "Tabel ..." headers 
                            # to span the FULL table width.
                            # Get global min_x and max_x from ALIGNED columns (unified coords)
                            table_min_x = None
                            table_max_x = None
                            
                            # Re-scan cells to find aligned bounds
                            # (We prioritize cells that were part of 'normal' columns)
                            for c in table_cells:
                                # Skip outlier-only cells for bounds calculation to avoid noise
                                # But we updated bboxes in place, so we can just use the final values 
                                # of aligned cells.
                                pass 
                            
                            # Better: Collecting during col processing or just simple min/max of all cells ?
                            # Simple min/max of all cells might be affected by outliers?
                            # Use min/max of column limits we calculated.
                            
                            all_x0 = [c['bbox']['x0'] for c in table_cells]
                            all_x1 = [c['bbox']['x1'] for c in table_cells]
                            
                            if all_x0 and all_x1:
                                table_min_x = min(all_x0)
                                table_max_x = max(all_x1)
                                
                                # Apply expansion to Header Cells
                                for c in table_cells:
                                    text_lower = c['text'].lower()
                                    # Heuristic: Starts with 'tabel' or is very likely a main header
                                    if text_lower.startswith('tabel') or text_lower.startswith('table'):
                                        old_w = c['bbox']['x1'] - c['bbox']['x0']
                                        new_w = table_max_x - table_min_x
                                        c['bbox']['x0'] = table_min_x
                                        c['bbox']['x1'] = table_max_x
                                        debug_log.write(f"  Refining Header '{c['text'][:15]}' to full width {new_w:.1f} (was {old_w:.1f})\n")

                            debug_log.close()

                        # Collect all cells for this table to post-process
                        current_table_final_cells = [] 
                        
                        # We need to temporarily collect cells, process them, then append to final_aligned
                        # But final_aligned structure is flat.
                        # Wait, we are inside the 'is_table' block. 
                        # We append cells to final_aligned ONE BY ONE later in the loop.
                        # We should collect them first, align, then append.
                        
                        # Existing loop `for cell_idx in sorted_cell_indices:` appends directly to final_aligned?
                        # No, the loop calculates PDF match.
                        # We need to modify the flow.
                        
                        table_cells_buffer = []  # Buffer to store processed cells
                        
                        # --- START MODIFIED TABLE CELL PROCESSING LOOP ---
                        # Instead of appending immediately, we store in buffer
                        # ...
                        
                        final_aligned.append({
                            "text": f"Table {elem_id}",
                            "matched_text": "",
                            "bbox": {"x0": table_x0, "y0": table_y0, "x1": table_x1, "y1": table_y1},
                            "bboxes": [],
                            "page": main_page,
                            "element_id": elem_id,
                            "confidence": 1.0,
                            "before_align_bboxes": [],
                            "is_table_container": True,
                        })
            
            # Table: buat elemen terpisah per cell
            # Try to get PyMuPDF table cells for more accurate bboxes (especially for image cells)
            pymupdf_table_cache = {}  # page -> list of tables
            pymupdf_pdf_doc = fitz.open(pdf_path)  # Open PDF for table extraction
            
            table_cells_buffer = []

            # Union of cell indices from tokens and text content
            all_cell_indices = set(cell_groups.keys())
            if elem_id in element_cell_texts:
                all_cell_indices.update(element_cell_texts[elem_id].keys())
            sorted_cell_indices = sorted([k for k in all_cell_indices if isinstance(k, int) and k != -1])

            for cell_idx in sorted_cell_indices:
                words = cell_groups.get(cell_idx, [])
                
                # Get original cell text from DOCX
                cell_text = ""
                if elem_id in element_cell_texts and cell_idx in element_cell_texts[elem_id]:
                    cell_text = element_cell_texts[elem_id][cell_idx]
                
                # Fallback: jika cell_text kosong, gunakan matched_text
                matched_text = " ".join(w["text"] for w in words)
                if not cell_text or not cell_text.strip():
                    cell_text = matched_text

                # Initialize bbox
                x0, y0, x1, y1 = 0, 0, 0, 0
                page_num = 0
                
                if words:
                    # Default: calculate bbox from token positions
                    x0 = min(w["bbox"]["x0"] for w in words)
                    y0 = min(w["bbox"]["y0"] for w in words)
                    x1 = max(w["bbox"]["x1"] for w in words)
                    y1 = max(w["bbox"]["y1"] for w in words)
                    page_num = words[0]["page"]
                else:
                    # No tokens, but we have text. 
                    # We need a page number to look up PDF table.
                    # Heuristic: use page of adjacent cells or table container info?
                    # For now, if we can't determine page, we might match wrong page.
                    # Try to use previous cell's page if available
                    if table_cells_buffer:
                        page_num = table_cells_buffer[-1]['page']
                    else:
                        # Fallback to main_page (usually accurate for single page tables)
                        page_num = main_page

                # Try PyMuPDF table enhancement
                # Cache PyMuPDF tables per page
                if page_num not in pymupdf_table_cache:
                    pymupdf_table_cache[page_num] = extract_pdf_table_cells(pymupdf_pdf_doc, page_num)
                
                pdf_tables = pymupdf_table_cache[page_num]
                
                pdf_match_found = False
                
                # Try to find matching PDF cell by text
                # ... existing match logic ...
                for t_idx, pdf_table in enumerate(pdf_tables):
                     # Get used indices for this SPECIFIC table in this page
                    table_used_indices = {c for (p, t, c) in used_pdf_cells if p == page_num and t == t_idx}
                    
                    pdf_cell_idx = match_docx_cell_to_pdf_cell(
                        cell_text, 
                        pdf_table['cells'], 
                        pdf_table['cell_texts'],
                        exclude_indices=table_used_indices
                    )
                    
                    if pdf_cell_idx is not None:
                        # PHASE 5: Vertical Gap Detection (Split Table Fix)
                        # Prevents a single DOCX table from claiming cells across a large vertical gap (e.g. merged PDF tables)
                        # Check against Y-bounds of ALL cells matched so far on this page
                        try:
                            candidate_bbox = pdf_table['cells'][pdf_cell_idx]
                            if candidate_bbox and table_cells_buffer:
                                # Get all cells on same page
                                same_page_cells = [c for c in table_cells_buffer if c['page'] == page_num]
                                if same_page_cells:
                                    # Calculate Y bounds of existing cluster
                                    cluster_min_y = min(c['bbox']['y0'] for c in same_page_cells)
                                    cluster_max_y = max(c['bbox']['y1'] for c in same_page_cells)
                                    
                                    curr_y0 = candidate_bbox[1]
                                    curr_y1 = candidate_bbox[3]
                                    
                                    # Calculate dynamic gap threshold based on average cell height
                                    # Gap threshold = avg_cell_height * GAP_MULTIPLIER
                                    # GAP_MULTIPLIER = 5 means we allow up to 5 empty rows between cells
                                    cell_heights = [c['bbox']['y1'] - c['bbox']['y0'] for c in same_page_cells]
                                    avg_cell_height = sum(cell_heights) / len(cell_heights) if cell_heights else 30.0
                                    GAP_MULTIPLIER = 5  # Number of "empty rows" that indicates a table split
                                    gap_threshold = max(avg_cell_height * GAP_MULTIPLIER, 50.0)  # Minimum 50px
                                    
                                    # Check if candidate is FAR BELOW the existing cluster
                                    gap_below = curr_y0 - cluster_max_y
                                    # Check if candidate is FAR ABOVE the existing cluster  
                                    gap_above = cluster_min_y - curr_y1
                                    
                                    if gap_below > gap_threshold:
                                        print(f"  [Phase 5] Rejecting '{cell_text[:10]}' -> PDF Cell {pdf_cell_idx}: {gap_below:.0f}px BELOW cluster (threshold={gap_threshold:.0f}px)")
                                        if log_file:
                                            log_file.write(f"  [Phase 5] Rejecting '{cell_text[:10]}': {gap_below:.0f}px below cluster (threshold={gap_threshold:.0f})\n")
                                        continue
                                    # NOTE: We do NOT reject cells ABOVE the cluster
                                    # This allows the cluster to expand upward as more cells are matched
                        except Exception as e:
                            pass # Fallback to allow match if check fails

                        # Mark as used logic here
                        used_pdf_cells.add((page_num, t_idx, pdf_cell_idx))
                        
                        pdf_cell_bbox = pdf_table['cells'][pdf_cell_idx]
                        if pdf_cell_bbox:
                            # Use PyMuPDF bbox (more accurate for image cells)
                            x0, y0, x1, y1 = pdf_cell_bbox
                            pdf_match_found = True
                            
                            # If words was empty, this rescues the cell!
                            break
                            
                if not words and not pdf_match_found:
                    # If no tokens AND no PDF match, we can't place this cell. Skip.
                    continue
                
                merged_segments = merge_bboxes_token_level(words, is_formula=False)

                # PHASE 5b: Check Gap for Token Fallback
                # If we didn't find a PDF match (or rejected it), ensuring tokens aren't also jumping the gap
                if not pdf_match_found and merged_segments and table_cells_buffer:
                     try:
                        same_page_cells = [c for c in table_cells_buffer if c['page'] == page_num]
                        if same_page_cells:
                            cluster_min_y = min(c['bbox']['y0'] for c in same_page_cells)
                            cluster_max_y = max(c['bbox']['y1'] for c in same_page_cells)
                            
                            # Calculate dynamic gap threshold (same as Phase 5)
                            cell_heights = [c['bbox']['y1'] - c['bbox']['y0'] for c in same_page_cells]
                            avg_cell_height = sum(cell_heights) / len(cell_heights) if cell_heights else 30.0
                            gap_threshold = max(avg_cell_height * 5, 50.0)
                            
                            curr_y0 = merged_segments[0]['bbox']['y0']
                            curr_y1 = merged_segments[-1]['bbox']['y1'] if len(merged_segments) > 1 else merged_segments[0]['bbox']['y1']
                            
                            gap_below = curr_y0 - cluster_max_y
                            
                            # Only reject cells FAR BELOW the cluster (not above)
                            if gap_below > gap_threshold:
                                if log_file:
                                    log_file.write(f"  [Phase 5b] Rejecting Token Match '{cell_text[:10]}': {gap_below:.0f}px below cluster\n")
                                continue
                     except Exception:
                        pass
                
                # Extract content items for cells with mixed content
                content_items = []
                content_bboxes = []
                if elem_id in element_cell_texts and cell_idx in element_cell_texts[elem_id]:
                    # Check if cell has image or shape
                    elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                    if elem and isinstance(elem.delemen_json_tree.get('content'), dict):
                        rows = elem.delemen_json_tree['content'].get('rows', [])
                        current_cell_idx = 0
                        for row in rows:
                            if isinstance(row, dict):
                                for cell in row.get('cells', []):
                                    if current_cell_idx == cell_idx:
                                        if isinstance(cell, list):
                                            for item in cell:
                                                if isinstance(item, dict):
                                                    if item.get('type') == 'text' and 'value' in item:
                                                        content_items.append({'type': 'text', 'value': item['value']})
                                                        if merged_segments:
                                                            content_bboxes.append(merged_segments[0]['bbox'])
                                                    elif item.get('type') == 'image':
                                                        content_items.append({'type': 'image', 'rId': item.get('rId')})
                                                        content_bboxes.append(None)
                                                    elif item.get('type') == 'drawing':
                                                        rId = item.get('value')
                                                        if rId and page_num in pdf_images:
                                                            # Try to find matching image
                                                            for img in pdf_images[page_num]:
                                                                # Simple heuristic: if cell bbox contains image center
                                                                img_cx = (img['bbox']['x0'] + img['bbox']['x1']) / 2
                                                                img_cy = (img['bbox']['y0'] + img['bbox']['y1']) / 2
                                                                if x0 <= img_cx <= x1 and y0 <= img_cy <= y1:
                                                                    content_items.append({'type': 'image', 'bbox': img['bbox'], 'id': rId})
                                                                    content_bboxes.append(img['bbox'])
                                                    elif item.get('type') == 'shape':
                                                        shape_text = extract_text_from_json_tree(item, return_cells=False)
                                                        # Extract shape content (nested items)
                                                        shape_content = []
                                                        if 'content' in item and isinstance(item['content'], list):
                                                            for shape_item in item['content']:
                                                                if isinstance(shape_item, dict):
                                                                    if shape_item.get('type') == 'text' and 'value' in shape_item:
                                                                        shape_content.append({'type': 'text', 'value': shape_item['value']})
                                                                    elif shape_item.get('type') == 'image':
                                                                        shape_content.append({'type': 'image', 'rId': shape_item.get('rId')})
                                                        content_items.append({'type': 'shape', 'value': shape_text, 'content': shape_content})
                                                        if merged_segments:
                                                            content_bboxes.append(merged_segments[0]['bbox'])
                                        break
                                    current_cell_idx += 1
                            if content_items:
                                break
                
                # Add cell to buffer
                table_cells_buffer.append({
                    "text": cell_text,
                    "matched_text": matched_text,
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "bboxes": merged_segments,
                    "page": page_num,
                    "element_id": f"{elem_id}_cell_{cell_idx}",
                    "parent_element_id": elem_id,
                    "confidence": 1.0,
                    "before_align_bboxes": [w["bbox"] for w in words],
                    "content_items": content_items if content_items else None,
                    "content_bboxes": content_bboxes if content_bboxes else None,
                    "is_table_cell": True
                })
            
            # Perform Column Alignment on buffer
            # This updates bboxes in-place in table_cells_buffer
            align_table_columns_x(table_cells_buffer, elem_id)
            
            # PHASE 5 POST-PROCESS: Split table if large internal Y-gap detected
            # Sort cells by Y and check for gaps
            if table_cells_buffer and len(table_cells_buffer) > 1:
                # Group by page
                cells_by_page = defaultdict(list)
                for c in table_cells_buffer:
                    cells_by_page[c['page']].append(c)
                
                for page_num, page_cells in cells_by_page.items():
                    if len(page_cells) <= 1:
                        continue
                    
                    # Sort by Y
                    page_cells.sort(key=lambda c: c['bbox']['y0'])
                    
                    # Calculate dynamic gap threshold
                    cell_heights = [c['bbox']['y1'] - c['bbox']['y0'] for c in page_cells]
                    avg_cell_height = sum(cell_heights) / len(cell_heights) if cell_heights else 30.0
                    gap_threshold = avg_cell_height * 5
                    
                    # Find largest internal gap
                    max_gap = 0
                    split_idx = -1
                    for i in range(len(page_cells) - 1):
                        curr_y1 = page_cells[i]['bbox']['y1']
                        next_y0 = page_cells[i + 1]['bbox']['y0']
                        gap = next_y0 - curr_y1
                        if gap > max_gap:
                            max_gap = gap
                            split_idx = i + 1
                    
                    # If gap exceeds threshold, remove cells AFTER the gap
                    if max_gap > gap_threshold and split_idx > 0:
                        cells_to_remove = page_cells[split_idx:]
                        for c in cells_to_remove:
                            if c in table_cells_buffer:
                                table_cells_buffer.remove(c)
                        if log_file:
                            log_file.write(f"  [Phase 5 Post] Removed {len(cells_to_remove)} cells after {max_gap:.0f}px gap (threshold={gap_threshold:.0f})\n")
                        print(f"  [Phase 5 Post] Removed {len(cells_to_remove)} cells after {max_gap:.0f}px gap (Table {elem_id})")
            
            # Recalculate container bbox from actual cell bboxes
            container_cells = [c for c in table_cells_buffer if c['page'] == main_page]
            if container_cells:
                 min_x = min(c['bbox']['x0'] for c in container_cells)
                 min_y = min(c['bbox']['y0'] for c in container_cells)
                 max_x = max(c['bbox']['x1'] for c in container_cells)
                 max_y = max(c['bbox']['y1'] for c in container_cells)
                 final_aligned[-1]['bbox'] = {"x0": min_x, "y0": min_y, "x1": max_x, "y1": max_y}

            # Add processed cells to final_aligned
            final_aligned.extend(table_cells_buffer)
            
            # Close PDF document
            pymupdf_pdf_doc.close()
            continue
        
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

                # Cek apakah element ini mengandung formula
                elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                is_formula = elem and (
                    elem.delemen_type == 'math' or 
                    has_formula_in_tree(elem.delemen_json_tree)
                )
                
                # FORMULA BBOX EXPANSION: 
                # PDF extraction sering tidak menangkap teks di superscript/subscript/fraction
                # Expand bbox vertikal untuk mencakup seluruh formula
                # FORMULA BBOX EXPANSION: 
                # PDF extraction sering tidak menangkap teks di superscript/subscript/fraction
                # Expand bbox vertikal untuk mencakup seluruh formula
                if is_formula and len(page_words) > 1:
                    # Calculate average line height
                    heights = [w["bbox"]["y1"] - w["bbox"]["y0"] for w in page_words]
                    avg_height = sum(heights) / len(heights)
                    
                    # Expand vertically by ~50% of average height (for superscript/subscript)
                    # Vertical expansion is usually safe/needed for superscript/subscript
                    y0 = y0 - avg_height * FORMULA_EXPAND_VERT_TOP_RATIO
                    y1 = y1 + avg_height * FORMULA_EXPAND_VERT_BOTTOM_RATIO
                    
                    # CONDITIONAL HORIZONTAL EXPANSION:
                    # Only expand explicitly if we find unaligned PDF tokens in that area
                    
                    # Ensure pdf_on_page exists (fallback if not created outside)
                    if 'pdf_on_page' not in locals():
                        pdf_on_page = defaultdict(list)
                        for idx, (bbox, page_idx) in enumerate(zip(pdf_bboxes, pdf_pages)):
                            pdf_on_page[page_idx].append((idx, bbox))
                    
                    current_width = x1 - x0
                    aligned_indices = set(w['pdf_index'] for w in page_words if 'pdf_index' in w)

                    # 1. Right Expansion Search
                    max_expand_right = min(current_width * FORMULA_EXPAND_HORZ_RIGHT_RATIO, FORMULA_EXPAND_HORZ_RIGHT_MAX_PX)
                    proposed_x1 = min(x1 + max_expand_right, FORMULA_PAGE_MARGIN_RIGHT)
                    
                    found_extra_right_x = x1
                    
                    if proposed_x1 > x1:
                        # Check PDF tokens on this page
                        for p_idx, p_bbox in pdf_on_page[page_num]:
                            if p_idx in aligned_indices: continue
                            
                            # Check geometric intersection with expansion zone
                            # Vertical overlap with the formula row
                            p_y_mid = (p_bbox[1] + p_bbox[3]) / 2
                            if y0 <= p_y_mid <= y1:
                                # Horizontal overlap: starts after current x1 and ends before proposed limit
                                if x1 < p_bbox[0] < proposed_x1:
                                    found_extra_right_x = max(found_extra_right_x, p_bbox[2])
                    
                    # Update x1 only if we found something
                    if found_extra_right_x > x1:
                        x1 = found_extra_right_x + 2 # slight padding
                    
                    # 2. Left Expansion Search
                    max_expand_left = min(current_width * FORMULA_EXPAND_HORZ_LEFT_RATIO, FORMULA_EXPAND_HORZ_LEFT_MAX_PX)
                    proposed_x0 = max(x0 - max_expand_left, FORMULA_PAGE_MARGIN_LEFT)
                    
                    found_extra_left_x = x0
                    
                    if proposed_x0 < x0:
                        for p_idx, p_bbox in pdf_on_page[page_num]:
                            if p_idx in aligned_indices: continue
                            
                            p_y_mid = (p_bbox[1] + p_bbox[3]) / 2
                            if y0 <= p_y_mid <= y1:
                                # Horizontal overlap: ends before current x0 and starts after proposed limit
                                if proposed_x0 < p_bbox[2] < x0:
                                    found_extra_left_x = min(found_extra_left_x, p_bbox[0])
                    
                    # Update x0 only if we found something
                    if found_extra_left_x < x0:
                        x0 = found_extra_left_x - 2
                
                merged_segments = merge_bboxes_token_level(page_words, is_formula=is_formula)
                
                # For formulas, also expand bboxes in merged_segments
                if is_formula and merged_segments:
                    for seg in merged_segments:
                        if 'bbox' in seg:
                            seg_height = seg['bbox']['y1'] - seg['bbox']['y0']
                            seg['bbox']['y0'] -= seg_height * FORMULA_EXPAND_VERT_TOP_RATIO
                            seg['bbox']['y1'] += seg_height * FORMULA_EXPAND_VERT_BOTTOM_RATIO

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
                    "is_formula": is_formula,  # Add flag for debugging
                })
    
    # Track used PDF tokens to prevent reuse
    used_pdf_tokens = set()
    for elem_id, cell_groups in element_groups.items():
        for cell_idx, words in cell_groups.items():
            for w in words:
                if 'pdf_index' in w:
                    used_pdf_tokens.add(w['pdf_index'])
    
    # Check for completely unaligned tables and populate unmapped_table_cells
    for elem in elements:
        elem_id = elem.delemen_id
        cells = extract_text_from_json_tree(elem.delemen_json_tree, return_cells=True)
        if isinstance(cells, list) and len(cells) > 0:
            # This is a table
            if elem_id in aligned_elem_ids:
                # Partially aligned, check for unmapped cells
                for cell_idx, cell_text in cells:
                    if cell_idx not in element_groups[elem_id] or not element_groups[elem_id][cell_idx]:
                        unmapped_table_cells.append((elem_id, cell_idx, cell_text))
            else:
                # Completely unaligned table, add all cells
                for cell_idx, cell_text in cells:
                    unmapped_table_cells.append((elem_id, cell_idx, cell_text))
    
    # FALLBACK: Try to match unmapped table cells using flexible sequence matching
    if unmapped_table_cells:
        for elem_id, cell_idx, cell_text in unmapped_table_cells:
            cell_tokens = tokenize(cell_text)
            if not cell_tokens:
                continue
            
            # Get page and expected Y range from other cells in same table
            target_page = None
            expected_y_min = None
            expected_y_max = None
            if elem_id in element_groups:
                for other_cell_idx, words in element_groups[elem_id].items():
                    if words:
                        target_page = words[0]['page']
                        # Calculate Y range from existing cells
                        for w in words:
                            y_mid = (w['bbox']['y0'] + w['bbox']['y1']) / 2
                            if expected_y_min is None or y_mid < expected_y_min:
                                expected_y_min = y_mid
                            if expected_y_max is None or y_mid > expected_y_max:
                                expected_y_max = y_mid
            
            # Expand Y range with tolerance (allow 100px above/below)
            if expected_y_min is not None:
                expected_y_min -= 100
                expected_y_max += 100
            
            # Fallback: use page from previous element
            if target_page is None:
                elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
                if elem:
                    for prev_elem in reversed(elements[:elements.index(elem)]):
                        for aligned in final_aligned:
                            if aligned.get('element_id') == prev_elem.delemen_id or aligned.get('parent_element_id') == prev_elem.delemen_id:
                                target_page = aligned['page']
                                break
                        if target_page is not None:
                            break
            
            if target_page is None:
                target_page = 0  # Last resort: start from page 0
            
            # Flexible sequence matching: allow skipping up to 2 tokens
            best_match = None
            best_score = 0
            
            for start_j in range(len(pdf_tokens)):
                if pdf_pages[start_j] != target_page:
                    continue
                if start_j in used_pdf_tokens:
                    continue
                
                # VALIDATION: Check if token Y position is within expected range
                if expected_y_min is not None:
                    token_y = (pdf_bboxes[start_j][1] + pdf_bboxes[start_j][3]) / 2
                    if token_y < expected_y_min or token_y > expected_y_max:
                        continue
                
                # Try to match cell_tokens starting from start_j
                matched_indices = []
                cell_tok_idx = 0
                pdf_tok_idx = start_j
                skip_count = 0
                
                while cell_tok_idx < len(cell_tokens) and pdf_tok_idx < len(pdf_tokens):
                    if pdf_pages[pdf_tok_idx] != target_page:
                        break
                    
                    if pdf_tok_idx in used_pdf_tokens:
                        pdf_tok_idx += 1
                        continue
                    
                    if pdf_tokens[pdf_tok_idx] == cell_tokens[cell_tok_idx]:
                        # SPATIAL CHECK: Ensure consecutive matches are not too far apart
                        if matched_indices:
                            prev_idx = matched_indices[-1]
                            prev_bbox = pdf_bboxes[prev_idx]
                            curr_bbox = pdf_bboxes[pdf_tok_idx]
                            
                            # Calculate vertical distance (gap between bottom of prev and top of curr, or vice-versa)
                            gap_y = max(0, curr_bbox[1] - prev_bbox[3], prev_bbox[1] - curr_bbox[3])
                            
                            # Also check horizontal distance for same-line items
                            # If lines are different (gap_y > 0), horizontal distance doesn't matter much (wrapping)
                            # But if gap_y is huge, it's a problem.
                            
                            # Limit gap to 150px (approx 10 lines)
                            if gap_y > 150:
                                # Start gap is too large, stop matching strictly for this sequence
                                break
                        
                        matched_indices.append(pdf_tok_idx)
                        cell_tok_idx += 1
                        pdf_tok_idx += 1
                        skip_count = 0
                    else:
                        # Allow skipping up to 2 PDF tokens
                        if skip_count < 2:
                            pdf_tok_idx += 1
                            skip_count += 1
                        else:
                            break
                
                # Score: matched tokens / total tokens
                if matched_indices:
                    score = len(matched_indices) / len(cell_tokens)
                    if score > best_score:
                        best_score = score
                        best_match = matched_indices
            
            # Accept match if score >= 50%
            if best_match and best_score >= 0.5:
                matched_words = []
                for pdf_idx in best_match:
                    bbox = pdf_bboxes[pdf_idx]
                    matched_words.append({
                        "text": pdf_tokens[pdf_idx],
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "page": pdf_pages[pdf_idx],
                    })
                    used_pdf_tokens.add(pdf_idx)
                
                if matched_words:
                    x0 = min(w["bbox"]["x0"] for w in matched_words)
                    y0 = min(w["bbox"]["y0"] for w in matched_words)
                    x1 = max(w["bbox"]["x1"] for w in matched_words)
                    y1 = max(w["bbox"]["y1"] for w in matched_words)
                    
                    merged_segments = merge_bboxes_token_level(matched_words, is_formula=False)
                    
                    final_aligned.append({
                        "text": cell_text,
                        "matched_text": " ".join(w["text"] for w in matched_words),
                        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                        "bboxes": merged_segments,
                        "page": matched_words[0]["page"],
                        "element_id": f"{elem_id}_cell_{cell_idx}",
                        "parent_element_id": elem_id,
                        "confidence": best_score,
                        "before_align_bboxes": [w["bbox"] for w in matched_words],
                    })
    
    # Add table containers for fallback-aligned tables
    fallback_table_ids = set()
    for aligned in final_aligned:
        if aligned.get('parent_element_id') and not aligned.get('is_table_container'):
            fallback_table_ids.add(aligned['parent_element_id'])
    
    for table_id in fallback_table_ids:
        # Check if container already exists
        if any(a.get('element_id') == table_id and a.get('is_table_container') for a in final_aligned):
            continue
        
        # Get all cells for this table
        table_cells = [a for a in final_aligned if a.get('parent_element_id') == table_id]
        if not table_cells:
            continue
        
        # Calculate table bbox from all cells
        table_x0 = min(c['bbox']['x0'] for c in table_cells)
        table_y0 = min(c['bbox']['y0'] for c in table_cells)
        table_x1 = max(c['bbox']['x1'] for c in table_cells)
        table_y1 = max(c['bbox']['y1'] for c in table_cells)
        
        # Insert container before first cell
        first_cell_idx = next(i for i, a in enumerate(final_aligned) if a.get('parent_element_id') == table_id)
        final_aligned.insert(first_cell_idx, {
            "text": f"Table {table_id}",
            "matched_text": "",
            "bbox": {"x0": table_x0, "y0": table_y0, "x1": table_x1, "y1": table_y1},
            "bboxes": [],
            "page": table_cells[0]['page'],
            "element_id": table_id,
            "confidence": min(c.get('confidence', 1.0) for c in table_cells),
            "before_align_bboxes": [],
            "is_table_container": True,
        })
    
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
    
    # Deduplicate final_aligned based on element_id
    # Keep the FIRST occurrence (assuming earlier update is correct buffer from table loop)
    # This addresses the ghost cell appearing after correct table processing
    seen_ids = set()
    deduped_aligned = []
    for item in final_aligned:
        eid = item.get('element_id')
        if eid and eid in seen_ids:
            continue
        if eid:
            seen_ids.add(eid)
        deduped_aligned.append(item)
    final_aligned = deduped_aligned

    # Track which elements have been aligned (including cells)
    aligned_parent_ids = set()
    for aligned in final_aligned:
        elem_id = aligned.get('element_id')
        parent_id = aligned.get('parent_element_id')
        if parent_id:
            aligned_parent_ids.add(parent_id)
        elif isinstance(elem_id, int):
            aligned_parent_ids.add(elem_id)
    
    for elem in elements:
        elem_id = elem.delemen_id
        if elem_id in aligned_parent_ids:
            continue
        
        # Element tidak ter-align sama sekali
        elem_text = element_texts.get(elem_id, "")
        if not elem_text or not elem_text.strip():
            continue
        
        if log_file:
            log_file.write(f"Element {elem_id} not aligned, text: {elem_text[:50]}...\n")
        
        # Tambahkan sebagai unaligned element dengan bbox dummy
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
    image_items = []  # (elem_id, rId, sequence, image_index_in_element)
    table_cell_images = []  # (elem_id, cell_idx, rId)
    for elem in elements:
        json_tree = elem.delemen_json_tree
        if not json_tree or not isinstance(json_tree, dict):
            continue
        
        # Check for images in content array
        content = json_tree.get('content', [])
        if isinstance(content, list):
            image_index = 0
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'image':
                        rId = item.get('rId')
                        if rId:
                            image_items.append((elem.delemen_id, rId, elem.delemen_sequence, image_index))
                            image_index += 1
                    # Check for images in shapes
                    elif item.get('type') == 'shape' and 'content' in item:
                        for shape_item in item['content']:
                            if isinstance(shape_item, dict) and shape_item.get('type') == 'image':
                                rId = shape_item.get('rId')
                                if rId:
                                    image_items.append((elem.delemen_id, rId, elem.delemen_sequence, image_index))
                                    image_index += 1
        
        # Check for images in table cells
        if isinstance(json_tree.get('content'), dict) and 'rows' in json_tree.get('content'):
            rows = json_tree['content']['rows']
            cell_index = 0
            for row in rows:
                if isinstance(row, dict):
                    for cell in row.get('cells', []):
                        has_image = False
                        if isinstance(cell, list):
                            for item in cell:
                                if isinstance(item, dict):
                                    if item.get('type') == 'image':
                                        rId = item.get('rId')
                                        if rId:
                                            table_cell_images.append((elem.delemen_id, cell_index, rId))
                                            has_image = True
                                    # Check for images in shapes within cells
                                    elif item.get('type') == 'shape' and 'content' in item:
                                        for shape_item in item['content']:
                                            if isinstance(shape_item, dict) and shape_item.get('type') == 'image':
                                                rId = shape_item.get('rId')
                                                if rId:
                                                    table_cell_images.append((elem.delemen_id, cell_index, rId))
                                                    has_image = True
                        cell_index += 1
    
    if log_file:
        log_file.write(f"\nFound {len(image_items)} images in content arrays\n")
        log_file.write(f"Found {len(table_cell_images)} images in table cells\n")
        log_file.write(f"Total images to align: {len(image_items) + len(table_cell_images)}\n")
        log_file.write(f"Found {len(image_only_cells_map)} tables with image-only cells\n")
        log_file.flush()
    sys.stderr.write(f"\nFound {len(image_items)} images in content arrays\n")
    sys.stderr.write(f"Found {len(table_cell_images)} images in table cells\n")
    sys.stderr.write(f"Found {len(image_only_cells_map)} tables with image-only cells\n")
    sys.stderr.flush()
    
    # PRE-PROCESS: Create entries in final_aligned for image-only cells
    # Ini penting agar image alignment bisa menemukan cell_entry
    for elem_id, image_only_cells in image_only_cells_map.items():
        if not image_only_cells:
            continue
        
        # Cari page target dari cells lain di table yang sama
        target_page = None
        table_bbox = None
        
        # Cek apakah ada cells dari table ini yang sudah aligned
        for aligned in final_aligned:
            aligned_elem_id = aligned.get('element_id')
            parent_id = aligned.get('parent_element_id')
            
            # Match jika parent_element_id == elem_id (cell dari table ini)
            # atau element_id == elem_id (table container)
            if parent_id == elem_id:
                target_page = aligned.get('page')
                # Kumpulkan bbox untuk estimasi posisi
                if table_bbox is None:
                    table_bbox = dict(aligned.get('bbox', {}))
                else:
                    cb = aligned.get('bbox', {})
                    if cb:
                        table_bbox['x0'] = min(table_bbox.get('x0', 9999), cb.get('x0', 9999))
                        table_bbox['y0'] = min(table_bbox.get('y0', 9999), cb.get('y0', 9999))
                        table_bbox['x1'] = max(table_bbox.get('x1', 0), cb.get('x1', 0))
                        table_bbox['y1'] = max(table_bbox.get('y1', 0), cb.get('y1', 0))
            elif aligned_elem_id == elem_id and aligned.get('is_table_container'):
                target_page = aligned.get('page')
                table_bbox = dict(aligned.get('bbox', {}))
        
        # Fallback: cari dari aligned elements berdasarkan sequence
        if target_page is None:
            elem = next((e for e in elements if e.dokumen_elemen_id == elem_id), None)
            if elem:
                # Cari element sebelumnya yang sudah aligned
                for prev_elem in reversed(elements[:elements.index(elem)]):
                    for aligned in final_aligned:
                        if aligned.get('element_id') == prev_elem.delemen_id or \
                           str(aligned.get('element_id', '')).startswith(f"{prev_elem.delemen_id}_"):
                            target_page = aligned.get('page')
                            break
                    if target_page is not None:
                        break
        
        if target_page is None:
            target_page = 0  # Last resort
        
        # Buat entry untuk setiap image-only cell
        for cell_idx, rIds in image_only_cells:
            cell_elem_id = f"{elem_id}_cell_{cell_idx}"
            
            # Skip jika sudah ada
            if any(a.get('element_id') == cell_elem_id for a in final_aligned):
                continue
            
            if log_file:
                log_file.write(f"Creating placeholder for image-only cell {cell_elem_id} on page {target_page}\n")
            
            # Buat entry dengan bbox placeholder (akan di-update oleh image alignment)
            placeholder_bbox = {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0}
            if table_bbox:
                # Gunakan bbox table sebagai hint untuk image search
                placeholder_bbox = dict(table_bbox)
            
            final_aligned.append({
                "text": "",  # Tidak ada text, hanya image
                "matched_text": "",
                "bbox": placeholder_bbox,
                "bboxes": [],
                "page": target_page,
                "element_id": cell_elem_id,
                "parent_element_id": elem_id,
                "confidence": 0.0,  # Akan di-update setelah image aligned
                "before_align_bboxes": [],
                "is_image_only_cell": True,  # Flag untuk identifikasi
                "image_rIds": rIds,  # rIds yang perlu di-align
                "content_items": [{'type': 'image', 'rId': rId} for rId in rIds],
                "content_bboxes": [None] * len(rIds),
            })
    
    # Align images in table cells FIRST (more specific)
    aligned_cell_images_count = 0
    for elem_id, cell_idx, rId in table_cell_images:
        cell_elem_id = f"{elem_id}_cell_{cell_idx}"
        already_aligned = any(a.get('element_id') == cell_elem_id and a.get('has_image_bbox') for a in final_aligned)
        if already_aligned:
            continue
        
        # Find the cell in final_aligned
        cell_entry = next((a for a in final_aligned if a.get('element_id') == cell_elem_id), None)
        if not cell_entry:
            if log_file:
                log_file.write(f"WARNING: Cell {cell_elem_id} not found in final_aligned for image rId={rId}\n")
            continue
        
        target_page = cell_entry['page']
        
        # Find best matching unused image on same page or nearby pages
        found_image = None
        best_score = 0
        
        # Search target page first, then nearby pages
        search_pages = [target_page] + [p for p in range(len(pdf_images)) if abs(p - target_page) <= 2 and p != target_page]
        
        for page_idx in search_pages:
            if page_idx not in pdf_images:
                continue
            
            for img in pdf_images[page_idx]:
                img_key = (page_idx, img['xref'])
                if img_key in used_images:
                    continue
                
                # Score based on page match and proximity to cell
                score = 0
                if page_idx == target_page:
                    score += 100
                else:
                    score += max(0, 50 - abs(page_idx - target_page) * 10)
                
                # Check proximity to cell bbox
                img_bbox = img['bbox']
                cell_bbox = cell_entry['bbox']
                
                # Calculate overlap or distance
                x_overlap = max(0, min(cell_bbox['x1'], img_bbox[2]) - max(cell_bbox['x0'], img_bbox[0]))
                y_overlap = max(0, min(cell_bbox['y1'], img_bbox[3]) - max(cell_bbox['y0'], img_bbox[1]))
                
                if x_overlap > 0 and y_overlap > 0:
                    # Image overlaps with cell - very good
                    score += 200
                else:
                    # Calculate distance
                    cell_center_x = (cell_bbox['x0'] + cell_bbox['x1']) / 2
                    cell_center_y = (cell_bbox['y0'] + cell_bbox['y1']) / 2
                    img_center_x = (img_bbox[0] + img_bbox[2]) / 2
                    img_center_y = (img_bbox[1] + img_bbox[3]) / 2
                    distance = ((cell_center_x - img_center_x)**2 + (cell_center_y - img_center_y)**2)**0.5
                    score += max(0, 100 - distance / 5)
                
                if score > best_score:
                    best_score = score
                    found_image = (page_idx, img)
        
        if found_image:
            page_idx, img = found_image
            img_bbox = img['bbox']
            used_images.add((page_idx, img['xref']))
            aligned_cell_images_count += 1
            
            # Add image bbox to cell's bboxes
            cell_entry['bboxes'].append({
                'page': page_idx,
                'bbox': {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
            })
            
            # Update cell bbox
            # Untuk image-only cells, bbox awalnya adalah placeholder - langsung set ke image bbox
            # Gunakan confidence == 0 sebagai indikator bahwa bbox belum di-set oleh image
            if cell_entry.get('is_image_only_cell') and cell_entry.get('confidence', 1.0) == 0.0:
                # Placeholder bbox - langsung set ke image bbox
                cell_entry['bbox'] = {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
                cell_entry['page'] = page_idx  # Update page juga
                cell_entry['confidence'] = 1.0  # Image found, update confidence
            else:
                # Expand existing bbox to include image
                cell_entry['bbox']['x0'] = min(cell_entry['bbox']['x0'], img_bbox[0])
                cell_entry['bbox']['y0'] = min(cell_entry['bbox']['y0'], img_bbox[1])
                cell_entry['bbox']['x1'] = max(cell_entry['bbox']['x1'], img_bbox[2])
                cell_entry['bbox']['y1'] = max(cell_entry['bbox']['y1'], img_bbox[3])
            
            cell_entry['has_image_bbox'] = True
            
            # Update parent table bbox recursively
            parent_id = cell_entry.get('parent_element_id')
            if parent_id:
                # Find parent on same page (for split tables)
                cell_page = cell_entry['page']
                parent_entry = next((a for a in final_aligned 
                                   if (a.get('element_id') == parent_id or a.get('element_id') == f"{parent_id}_page_{cell_page}") 
                                   and a.get('is_table_container') 
                                   and a.get('page') == cell_page), None)
                if parent_entry:
                    parent_entry['bbox']['x0'] = min(parent_entry['bbox']['x0'], img_bbox[0])
                    parent_entry['bbox']['y0'] = min(parent_entry['bbox']['y0'], img_bbox[1])
                    parent_entry['bbox']['x1'] = max(parent_entry['bbox']['x1'], img_bbox[2])
                    parent_entry['bbox']['y1'] = max(parent_entry['bbox']['y1'], img_bbox[3])
            
            # Update content_bboxes if exists
            if cell_entry.get('content_items'):
                for i, item in enumerate(cell_entry['content_items']):
                    if item.get('type') == 'image' and cell_entry['content_bboxes'][i] is None:
                        cell_entry['content_bboxes'][i] = {
                            'page': page_idx,
                            'bbox': {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
                        }
                        break
        else:
            if log_file:
                log_file.write(f"WARNING: No PDF image found for cell {cell_elem_id} rId={rId} page={target_page}\n")
    
    if log_file:
        log_file.write(f"\nAligned {aligned_cell_images_count} table cell images out of {len(table_cell_images)}\n")
    
    # Then align standalone images
    aligned_images_count = 0
    
    # Collect all unused images by page
    unused_images_by_page = {}
    for page_idx in pdf_images:
        unused_images_by_page[page_idx] = []
        for img in pdf_images[page_idx]:
            img_key = (page_idx, img['xref'])
            if img_key not in used_images:
                unused_images_by_page[page_idx].append((page_idx, img))
    
    for elem_id, rId, sequence, img_idx in image_items:
        already_aligned = any(a['element_id'] == elem_id and a.get('rId') == rId and a.get('image_index') == img_idx for a in final_aligned)
        if already_aligned:
            continue
        
        # Strategy: Use prev/next element pages to determine image location
        best_image = None
        best_score = 0
        
        # Find prev and next elements
        elem_idx = next((i for i, e in enumerate(elements) if e.dokumen_elemen_id == elem_id), None)
        prev_page = None
        next_page = None
        
        if elem_idx is not None:
            # Get prev element page
            if elem_idx > 0:
                prev_elem_id = elements[elem_idx - 1].dokumen_elemen_id
                for aligned in final_aligned:
                    if aligned.get('element_id') == prev_elem_id or str(aligned.get('element_id')).startswith(f"{prev_elem_id}_"):
                        prev_page = aligned['page']
                        break
            
            # Get next element page
            if elem_idx < len(elements) - 1:
                next_elem_id = elements[elem_idx + 1].dokumen_elemen_id
                for aligned in final_aligned:
                    if aligned.get('element_id') == next_elem_id or str(aligned.get('element_id')).startswith(f"{next_elem_id}_"):
                        next_page = aligned['page']
                        break
        
        # Determine target page from context
        target_page = None
        if next_page is not None:
            target_page = next_page  # Prefer next element (caption)
        elif prev_page is not None:
            target_page = prev_page
        
        # Fallback: use element's own alignment
        elem_bbox = None
        if target_page is None:
            for aligned in final_aligned:
                if aligned['element_id'] == elem_id:
                    target_page = aligned['page']
                    elem_bbox = aligned['bbox']
                    break
        
        # Search for image on target page or nearby
        if target_page is not None:
            # Search target page first, then prev page (if image between prev and next)
            search_pages = [target_page]
            if prev_page is not None and prev_page != target_page:
                search_pages.insert(0, prev_page)  # Check prev page first
            
            for page_idx in search_pages:
                if page_idx not in unused_images_by_page:
                    continue
                
                for page_idx2, img in unused_images_by_page[page_idx]:
                    img_key = (page_idx2, img['xref'])
                    if img_key in used_images:
                        continue
                    
                    score = 0
                    page_diff = abs(page_idx2 - target_page)
                    
                    if page_diff == 0:
                        score += 100
                    else:
                        score += max(0, 50 - page_diff * 3)
                    
                    if elem_bbox and elem_bbox['x0'] > 0:
                        img_bbox = img['bbox']
                        elem_center_x = (elem_bbox['x0'] + elem_bbox['x1']) / 2
                        elem_center_y = (elem_bbox['y0'] + elem_bbox['y1']) / 2
                        img_center_x = (img_bbox[0] + img_bbox[2]) / 2
                        img_center_y = (img_bbox[1] + img_bbox[3]) / 2
                        distance = ((elem_center_x - img_center_x)**2 + (elem_center_y - img_center_y)**2)**0.5
                        score += max(0, 100 - distance / 10)
                    
                    if score > best_score:
                        best_score = score
                        best_image = (page_idx2, img)
        
        # Approach 3: If no good match yet, just take next unused image in sequence
        if best_image is None or best_score < 20:
            # Find first unused image
            for page_idx in sorted(unused_images_by_page.keys()):
                if unused_images_by_page[page_idx]:
                    for page_idx2, img in unused_images_by_page[page_idx]:
                        img_key = (page_idx2, img['xref'])
                        if img_key not in used_images:
                            best_image = (page_idx2, img)
                            best_score = 10
                            if log_file:
                                log_file.write(f"Image elem_id={elem_id} using fallback: first unused image on page {page_idx2}\n")
                            break
                if best_image:
                    break
        
        if best_image:
            page_idx, img = best_image
            bbox = img['bbox']
            used_images.add((page_idx, img['xref']))
            aligned_images_count += 1
            
            final_aligned.append({
                "text": "[IMAGE]",
                "matched_text": "[IMAGE]",
                "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                "bboxes": [{"page": page_idx, "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}}],
                "page": page_idx,
                "element_id": elem_id,
                "rId": rId,
                "image_index": img_idx,
                "confidence": 0.8,
                "before_align_bboxes": [{"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}],
                "is_image": True,
            })
        else:
            if log_file:
                log_file.write(f"WARNING: No PDF image found for elem_id={elem_id} rId={rId} target_page={target_page}\n")
    
    if log_file:
        log_file.write(f"\nAligned {aligned_images_count} standalone images out of {len(image_items)}\n")

    if log_file:
        log_file.write(f"\nAligned elements: {len(final_aligned)}\n")
        log_file.write("=== END TRACE LOG ===\n\n")
        log_file.flush()
    sys.stderr.write(f"\nAligned elements: {len(final_aligned)}\n")
    sys.stderr.write("=== END TRACE LOG ===\n\n")
    sys.stderr.flush()

    # Separate unaligned elements
    unaligned_elements = [e for e in final_aligned if e.get('unaligned')]
    aligned_only = [e for e in final_aligned if not e.get('unaligned')]
    
    # Find unaligned PDF tokens
    used_pdf_indices = set()
    for elem_id, cell_groups in element_groups.items():
        for cell_idx, words in cell_groups.items():
            for w in words:
                if 'pdf_index' in w:
                    used_pdf_indices.add(w['pdf_index'])
    
    unaligned_tokens = []
    for i, token in enumerate(pdf_tokens):
        if i not in used_pdf_indices:
            unaligned_tokens.append({
                'text': token,
                'bbox': {'x0': pdf_bboxes[i][0], 'y0': pdf_bboxes[i][1], 'x1': pdf_bboxes[i][2], 'y1': pdf_bboxes[i][3]},
                'page': pdf_pages[i],
                'pdf_index': i
            })
    
    if log_file:
        log_file.write(f"\nUnaligned tokens before reassignment: {len(unaligned_tokens)}\n")
        if unaligned_tokens:
            log_file.write(f"Sample unaligned (first 10): {[t['text'] for t in unaligned_tokens[:10]]}\n")
        log_file.flush()

    # PHASE 4: GLOBAL CONTAINER CLEANUP (Force Strict BBox)
    print("\n--- PHASE 4: GLOBAL CONTAINER CLEANUP ---")
    
    # Map table_id -> page -> [child_bboxes]
    table_children_bboxes = {}
    for item in final_aligned:
        pid = item.get('parent_element_id')
        page = item.get('page')
        if pid and item.get('bbox') and page is not None:
            # Handle str/int mismatch
            pid_str = str(pid)
            if pid_str not in table_children_bboxes:
                table_children_bboxes[pid_str] = {}
            
            if page not in table_children_bboxes[pid_str]:
                table_children_bboxes[pid_str][page] = []
                
            table_children_bboxes[pid_str][page].append(item['bbox'])
            
    # Update Containers
    for item in final_aligned:
        if item.get('is_table_container') or (item.get('text', '').startswith('Table ') and not item.get('parent_element_id')):
            eid_str = str(item.get('element_id'))
            page = item.get('page')
            
            # Skip if page is None or invalid
            if page is None:
                continue
                
            print(f"Checking Container {eid_str} on page {page} (IsTable={item.get('is_table_container')})...")
            
            if eid_str in table_children_bboxes and page in table_children_bboxes[eid_str]:
                children_bboxes = table_children_bboxes[eid_str][page]
                if children_bboxes:
                    new_x0 = min(b['x0'] for b in children_bboxes)
                    new_y0 = min(b['y0'] for b in children_bboxes)
                    new_x1 = max(b['x1'] for b in children_bboxes)
                    new_y1 = max(b['y1'] for b in children_bboxes)
                    
                    old_h = item['bbox']['y1'] - item['bbox']['y0']
                    new_h = new_y1 - new_y0
                    
                    print(f"  Refining {eid_str}: H {old_h:.1f} -> {new_h:.1f}")
                    
                    # Update bbox IF it changes significantly or if we want strict fit
                    # For cross-page fix, we ALWAYS update to the page-specific union
                    if log_file:
                        log_file.write(f"Refining Container {item['element_id']} on page {page}: H {old_h:.1f} -> {new_h:.1f}\n")
                    item['bbox'] = {"x0": new_x0, "y0": new_y0, "x1": new_x1, "y1": new_y1}
            else:
                print(f"  No children found for {eid_str} on page {page}")
    
    return {
        "aligned_words": aligned_only,
        "unaligned_elements": unaligned_elements,
        "unaligned_tokens": unaligned_tokens,
        "stats": {
            "total_words": len(aligned_only),
            "assigned_words": len(aligned_only),
            "unaligned_count": len(unaligned_elements),
            "unaligned_tokens_count": len(unaligned_tokens),

            "coverage": 1.0,
        },
    }
