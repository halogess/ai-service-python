"""Text normalization and tokenization utilities"""

import re
import unicodedata
from .config import TOKEN_RE


def normalize_text(s: str, preserve_whitespace: bool = False, is_formula: bool = False) -> str:
    """Normalisasi teks untuk alignment"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("×", "*").replace("÷", "/")
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("≠", "!=").replace("≈", "~=")
    
    # Normalisasi mathematical symbols
    s = s.replace('\U0001D707', 'μ').replace('\U0001D6CD', 'μ')
    
    # Italic/Bold uppercase A-Z -> normal
    for i in range(26):
        s = s.replace(chr(0x1D434 + i), chr(0x41 + i))
        s = s.replace(chr(0x1D400 + i), chr(0x41 + i))
        s = s.replace(chr(0x1D468 + i), chr(0x41 + i))
    
    # Italic/Bold lowercase a-z -> normal
    for i in range(26):
        if i != 7:
            s = s.replace(chr(0x1D44E + i), chr(0x61 + i))
        s = s.replace(chr(0x1D41A + i), chr(0x61 + i))
        s = s.replace(chr(0x1D482 + i), chr(0x61 + i))
    s = s.replace('ℎ', 'h')
    
    # Greek letters normalization
    for i in range(25):
        s = s.replace(chr(0x1D6FC + i), chr(0x03B1 + i))
        s = s.replace(chr(0x1D6E2 + i), chr(0x0391 + i))
    
    # Gabungkan Greek letter + subscript
    s = re.sub(r'([α-ωΑ-Ω])\s+([A-Z]{1,3})(?=\s|$|[^A-Za-z])', r'\1\2', s)
    
    if is_formula:
        from .formula_utils import normalize_formula_text
        s = normalize_formula_text(s)
    
    if not preserve_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str, is_formula: bool = False):
    """Tokenizer untuk alignment"""
    s = normalize_text(s, is_formula=is_formula)
    if not s:
        return []
    
    # Normalisasi: hapus spasi di antara huruf dan 2 digit
    s = re.sub(r'([a-zα-ω]{2,})\s+(\d)\s+(\d)(?=\s|$)', r'\1\2\3', s)
    s = re.sub(r'(?<=\s)(\d)\s+(\d)(?=\s|$)', r'\1\2', s)
    s = re.sub(r'([\-=e])\s+(\d)\s+(\d)', r'\1 \2\3', s)
    s = re.sub(r'([a-zA-Zα-ωΑ-Ω]+)-\s+([a-zA-Zα-ωΑ-Ω]+)', r'\1\2', s)
    
    tokens = re.findall(
        r'\d*[A-Z\u00c0-\u00df\u0391-\u03a9][A-Za-z\u00c0-\u00ff\u0370-\u03ff\u2080-\u2089\u2070-\u2079_]*(?=\d{1,2}(?!\d))|'
        r'\d*[A-Za-z\u00c0-\u00ff\u0370-\u03ff\u2080-\u2089\u2070-\u2079_]+\d*|\d+(?:\.\d+)*|[^\w\s]',
        s, flags=re.UNICODE
    )
    if tokens and tokens[-1] in ('.', ':'):
        tokens = tokens[:-1]
    return tokens
