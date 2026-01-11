"""Formula-specific utilities"""

import re


def normalize_formula_text(s: str) -> str:
    """Normalisasi khusus untuk teks formula"""
    if not s:
        return s
    
    # Tambah spasi setelah Greek letter + subscript
    s = re.sub(r'([μαβγδε][A-Z]{1,3})(\d)', r'\1 \2', s)
    
    # Tambah spasi di sekitar operator =
    s = re.sub(r'(?<![<>!=\s])=(?![=\s])', r' = ', s)
    
    # Tambah spasi setelah koma
    s = re.sub(r',(?!\s)', r', ', s)
    
    # Handle fraction/exponent
    s = re.sub(r'([eE])\s*-\s*12', r'\1 - 1 2', s)
    s = re.sub(r'([eE])\s*-\s*1\s+2', r'\1 - 1 2', s)
    s = re.sub(r'(\d)(\d+\.\d+)-', r'\1 \2 - ', s)
    
    # Split Latin/Greek characters
    s = re.sub(r'([a-zA-Z0-9])([μαβγδεΑ-Ω])', r'\1 \2', s)
    s = re.sub(r'([μαβγδεΑ-Ω])([a-z])', r'\1 \2', s)
    
    # Normalize glued variables
    s = re.sub(r'([A-Z]{1,10})(=)', r'\1 \2', s)
    
    # Pisahkan angka digit
    s = re.sub(r'(\d+\.\d{2})(\d\.\d+)', r'\1 \2', s)
    s = re.sub(r'(\d+\.\d+)(\d{3})$', r'\1 \2', s)
    s = re.sub(r'(\d+\.\d+)(\d{3})\s', r'\1 \2 ', s)
    
    # Normalize multiple spaces
    s = re.sub(r'\s+', ' ', s)
    
    return s


def has_formula_in_tree(json_tree):
    """Cek apakah json_tree mengandung formula"""
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
