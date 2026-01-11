"""openxml_alignment.py

OpenXML ↔ PDF word alignment (token stream alignment + bbox assignment)

Target constraints:
- Semua word dari PDF harus ter-assign (tidak ada yang hilang).
- Word tidak boleh "nyangkut" ke elemen tetangga (monotonic alignment).
- Token ekstra di PDF (page number, numbering/list marker, punctuation) dipertahankan
  dan di-attach via classifier berbasis bbox.

Core idea:
- Flatten OpenXML menjadi stream token + element_id per token.
- Flatten PDF words menjadi stream token (word-level) + bbox.
- Buat token stream PDF versi "repaired" untuk alignment (merge acronym/decimal split)
  sambil menyimpan mapping repaired_token -> span original_word_idx.
- Jalankan Levenshtein.opcodes pada (docx_tokens, pdf_repaired_tokens).
- Assign hasil opcodes ke ORIGINAL pdf_words via mapping span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein
from shapely.geometry import box as Box
from shapely.ops import unary_union


EMPTY_TOKEN = "<EMPTY>"


@dataclass
class PDFWord:
    text: str
    normalized: str
    bbox: Tuple[float, float, float, float]
    page: int
    index: int
    assigned_element_id: Optional[int] = None
    confidence: float = 1.0


class TextNormalizer:
    """Normalisasi text untuk matching yang lebih stabil."""

    @staticmethod
    def normalize_mathematical_chars(text: str) -> str:
        """Normalize Mathematical Alphanumeric Symbols to plain ASCII/Greek"""
        mapping = {
            # Greek letters
            '\U0001D707': 'μ', '\U0001D6FC': 'α', '\U0001D6FD': 'β', '\U0001D6FE': 'γ',
            # Mathematical Italic Capital Letters
            '\U0001D434': 'A', '\U0001D435': 'B', '\U0001D436': 'C', '\U0001D437': 'D',
            '\U0001D438': 'E', '\U0001D439': 'F', '\U0001D43A': 'G', '\U0001D43B': 'H',
            '\U0001D43C': 'I', '\U0001D43D': 'J', '\U0001D43E': 'K', '\U0001D43F': 'L',
            '\U0001D440': 'M', '\U0001D441': 'N', '\U0001D442': 'O', '\U0001D443': 'P',
            '\U0001D444': 'Q', '\U0001D445': 'R', '\U0001D446': 'S', '\U0001D447': 'T',
            '\U0001D448': 'U', '\U0001D449': 'V', '\U0001D44A': 'W', '\U0001D44B': 'X',
            '\U0001D44C': 'Y', '\U0001D44D': 'Z',
            # Mathematical Italic Small Letters
            '\U0001D44E': 'a', '\U0001D44F': 'b', '\U0001D450': 'c', '\U0001D451': 'd',
            '\U0001D452': 'e', '\U0001D453': 'f', '\U0001D454': 'g', '\U0001D456': 'i',
            '\U0001D457': 'j', '\U0001D458': 'k', '\U0001D459': 'l', '\U0001D45A': 'm',
            '\U0001D45B': 'n', '\U0001D45C': 'o', '\U0001D45D': 'p', '\U0001D45E': 'q',
            '\U0001D45F': 'r', '\U0001D460': 's', '\U0001D461': 't', '\U0001D462': 'u',
            '\U0001D463': 'v', '\U0001D464': 'w', '\U0001D465': 'x', '\U0001D466': 'y',
            '\U0001D467': 'z',
            # Mathematical Bold Capital Letters
            '\U0001D400': 'A', '\U0001D401': 'B', '\U0001D402': 'C', '\U0001D403': 'D',
            '\U0001D404': 'E', '\U0001D405': 'F', '\U0001D406': 'G', '\U0001D407': 'H',
            '\U0001D408': 'I', '\U0001D409': 'J', '\U0001D40A': 'K', '\U0001D40B': 'L',
            '\U0001D40C': 'M', '\U0001D40D': 'N', '\U0001D40E': 'O', '\U0001D40F': 'P',
            '\U0001D410': 'Q', '\U0001D411': 'R', '\U0001D412': 'S', '\U0001D413': 'T',
            '\U0001D414': 'U', '\U0001D415': 'V', '\U0001D416': 'W', '\U0001D417': 'X',
            '\U0001D418': 'Y', '\U0001D419': 'Z',
            # Operators
            '−': '-', '–': '-', '—': '-', '×': '*', '÷': '/',
        }
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""

        # Normalize mathematical characters first
        text = TextNormalizer.normalize_mathematical_chars(text)

        # Tab & newline → spasi
        text = re.sub(r"[\t\n\r]+", " ", text)

        # Gabung run huruf kapital berspasi: "G P M I S L O" → "GPMISLO"
        text = re.sub(
            r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
            lambda m: m.group(0).replace(" ", ""),
            text,
        )

        # Koma desimal dengan spasi: "-0, 44" → "-0.44" (hanya di antara digit)
        text = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ".", text)

        # Spasi sebelum dot: "0 .84" → "0.84"
        text = re.sub(r"(?<=\d)\s+\.(?=\d)", ".", text)

        # Spasi di angka desimal: "0.1 3" → "0.13"
        text = re.sub(r"(\d+\.\d*)\s+(\d+)", r"\1\2", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def deduplicate_math(content_items: List[Dict]) -> List[Dict]:
        """Deduplikasi math chunks yang identik bersebelahan (kasus umum output OpenXML)."""
        if not content_items:
            return []

        result: List[Dict] = []
        prev_math: Optional[str] = None

        for item in content_items:
            if item.get("type") == "math":
                math_text = item.get("text", "")
                if math_text != prev_math:
                    result.append(item)
                    prev_math = math_text
            else:
                result.append(item)
                prev_math = None

        return result


class OpenXMLFlattener:
    """Flatten OpenXML elements menjadi token stream (word-level)."""

    @staticmethod
    def flatten_element(element: Dict) -> Tuple[List[str], List[int]]:
        element_id = element["dokumen_elemen_id"]
        element_type = element["dokumen_elemen_type"]
        json_tree = element.get("dokumen_elemen_json_tree", {})

        if element_type == "table":
            return OpenXMLFlattener._flatten_table(json_tree, element_id)
        if element_type in ("image", "sectionBreak"):
            return [], []
        return OpenXMLFlattener._flatten_paragraph(json_tree, element_id)

    @staticmethod
    def _flatten_paragraph(json_tree: Dict, element_id: int) -> Tuple[List[str], List[int]]:
        content = json_tree.get("content", [])
        if not content:
            return [], []

        content = TextNormalizer.deduplicate_math(content)

        parts: List[str] = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("value", ""))
            elif item.get("type") == "math":
                parts.append(item.get("text", ""))

        normalized = TextNormalizer.normalize(" ".join(parts))
        tokens = normalized.split() if normalized else []
        return tokens, [element_id] * len(tokens)

    @staticmethod
    def _flatten_table(json_tree: Dict, element_id: int) -> Tuple[List[str], List[int]]:
        rows = json_tree.get("rows", [])
        all_text: List[str] = []

        for row in rows:
            cells = row.get("cells", [])
            for cell in cells:
                cell_parts: List[str] = []

                # CASE 1: cell adalah string (sesuai contoh JSON kamu)
                if isinstance(cell, str):
                    if cell.strip():
                        cell_parts.append(cell)

                # CASE 2: cell adalah list of dict (text/math items)
                elif isinstance(cell, list):
                    for item in cell:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") == "text":
                            cell_parts.append(item.get("value", ""))
                        elif item.get("type") == "math":
                            cell_parts.append(item.get("text", ""))

                if cell_parts:
                    all_text.append(" ".join(cell_parts))

        normalized = TextNormalizer.normalize(" ".join(all_text))
        tokens = normalized.split() if normalized else []
        return tokens, [element_id] * len(tokens)

    @staticmethod
    def flatten_all(elements: List[Dict]) -> Tuple[List[str], List[int], Dict[int, Dict]]:
        all_tokens: List[str] = []
        all_element_ids: List[int] = []
        element_map: Dict[int, Dict] = {}

        for element in elements:
            eid = element["dokumen_elemen_id"]
            element_map[eid] = element

            tokens, ids = OpenXMLFlattener.flatten_element(element)
            all_tokens.extend(tokens)
            all_element_ids.extend(ids)

        return all_tokens, all_element_ids, element_map


class PDFFlattener:
    """Flatten PDF words (PyMuPDF) menjadi list PDFWord + token stream repaired untuk alignment."""

    @staticmethod
    def flatten_words(pdf_words: List[Dict]) -> List[PDFWord]:
        out: List[PDFWord] = []
        for idx, w in enumerate(pdf_words):
            text = w.get("text", "")
            bbox = tuple(w.get("bbox", [0, 0, 0, 0]))
            page = int(w.get("page", 0))

            normalized = TextNormalizer.normalize(text)
            if not normalized:
                normalized = EMPTY_TOKEN

            out.append(
                PDFWord(
                    text=text,
                    normalized=normalized,
                    bbox=bbox,
                    page=page,
                    index=idx,
                )
            )
        return out

    @staticmethod
    def repair_token_stream(
        pdf_words: List[PDFWord],
        *,
        x_gap_threshold: float = 3.0,
        y_tol: float = 2.0,
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Buat token stream PDF yang lebih cocok untuk alignment.

        Meng-merge token yang sering pecah di PDF:
        - acronym: "G" "P" "M" → "GPM" (hanya jika satu line / gap kecil)
        - decimal split: "0.1" "3" → "0.13" (hanya jika dekat)

        Return:
        - repaired_tokens: List[str]
        - spans: List[(orig_start, orig_end)] dengan panjang sama (index di pdf_words)
        """

        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []

        def same_line(a: PDFWord, b: PDFWord) -> bool:
            if a.page != b.page:
                return False
            # vertical overlap / closeness
            return abs(a.bbox[1] - b.bbox[1]) <= y_tol and abs(a.bbox[3] - b.bbox[3]) <= y_tol

        def close_x(a: PDFWord, b: PDFWord) -> bool:
            if a.page != b.page:
                return False
            gap = b.bbox[0] - a.bbox[2]
            return gap >= -0.5 and gap <= x_gap_threshold

        i = 0
        n = len(pdf_words)
        while i < n:
            w = pdf_words[i]

            # Keep EMPTY as-is (tidak di-merge)
            if w.normalized == EMPTY_TOKEN:
                tokens.append(EMPTY_TOKEN)
                spans.append((i, i + 1))
                i += 1
                continue

            # Merge acronym run: single capital letters, same line, gap kecil
            if re.fullmatch(r"[A-Z]", w.normalized):
                merged = w.normalized
                j = i + 1
                while j < n:
                    wj = pdf_words[j]
                    if not re.fullmatch(r"[A-Z]", wj.normalized):
                        break
                    if not same_line(pdf_words[j - 1], wj) or not close_x(pdf_words[j - 1], wj):
                        break
                    merged += wj.normalized
                    j += 1
                if j > i + 1:
                    tokens.append(merged)
                    spans.append((i, j))
                    i = j
                    continue

            # Merge decimal split: "0.1" + "3" → "0.13" (same line, close)
            if re.fullmatch(r"\d+\.\d*", w.normalized) and i + 1 < n:
                nxt = pdf_words[i + 1]
                if re.fullmatch(r"\d+", nxt.normalized) and same_line(w, nxt) and close_x(w, nxt):
                    tokens.append(w.normalized + nxt.normalized)
                    spans.append((i, i + 2))
                    i += 2
                    continue

            tokens.append(w.normalized)
            spans.append((i, i + 1))
            i += 1

        return tokens, spans


class TokenAligner:
    """Alignment menggunakan RapidFuzz Levenshtein opcodes."""

    @staticmethod
    def align(docx_tokens: List[str], pdf_tokens: List[str]) -> List[Tuple]:
        return Levenshtein.opcodes(docx_tokens, pdf_tokens)

    @staticmethod
    def assign_from_opcodes_repaired(
        opcodes: List[Tuple],
        docx_element_ids: List[int],
        pdf_words: List[PDFWord],
        element_map: Dict[int, Dict],
        repaired_spans: List[Tuple[int, int]],
    ) -> None:
        """Assign ke ORIGINAL pdf_words, tapi opcodes memakai index repaired tokens."""

        def orig_indices_for_repaired_range(j1: int, j2: int) -> List[int]:
            idxs: List[int] = []
            for r in range(j1, j2):
                if r < 0 or r >= len(repaired_spans):
                    continue
                a, b = repaired_spans[r]
                idxs.extend(range(a, b))
            return idxs

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                # 1:1 mapping pada repaired token. Bisa 1->many pada original word span.
                for docx_idx, rep_idx in zip(range(i1, i2), range(j1, j2)):
                    if docx_idx >= len(docx_element_ids) or rep_idx >= len(repaired_spans):
                        break
                    eid = docx_element_ids[docx_idx]
                    a, b = repaired_spans[rep_idx]
                    for pdf_idx in range(a, b):
                        if 0 <= pdf_idx < len(pdf_words):
                            pdf_words[pdf_idx].assigned_element_id = eid

            elif tag == "replace":
                if i1 >= i2 or j1 >= j2:
                    continue
                if i1 >= len(docx_element_ids):
                    continue

                # Segment docx by element boundaries
                groups: List[Tuple[int, int, int]] = []
                start = i1
                cur = docx_element_ids[i1]
                for k in range(i1 + 1, min(i2, len(docx_element_ids))):
                    if docx_element_ids[k] != cur:
                        groups.append((start, k, cur))
                        start = k
                        cur = docx_element_ids[k]
                groups.append((start, min(i2, len(docx_element_ids)), cur))

                # Convert repaired range to list of ORIGINAL pdf indices
                pdf_orig_idxs = orig_indices_for_repaired_range(j1, j2)
                if not pdf_orig_idxs:
                    continue

                pdf_len = len(pdf_orig_idxs)
                docx_len = max(1, (i2 - i1))
                weights = [max(1, (b - a)) for (a, b, _eid) in groups]

                # Allocation strategy: if enough tokens, guarantee each group >=1
                g = len(groups)
                counts = [0] * g
                if pdf_len >= g:
                    counts = [1] * g
                    remaining = pdf_len - g
                else:
                    remaining = pdf_len

                # Proportional distribute remaining
                wsum = float(sum(weights))
                if remaining > 0:
                    raw = [remaining * (w / wsum) for w in weights]
                    base = [int(x) for x in raw]
                    frac = [x - int(x) for x in raw]
                    add = remaining - sum(base)
                    # distribute by largest fractional part
                    order = sorted(range(g), key=lambda idx: frac[idx], reverse=True)
                    for idx in order[:add]:
                        base[idx] += 1
                    counts = [c + b for c, b in zip(counts, base)]

                # Assign sequentially
                cursor = 0
                for (count, (_a, _b, eid)) in zip(counts, groups):
                    element = element_map.get(eid, {})
                    has_math = TokenAligner._has_math_content(element)
                    for _ in range(count):
                        if cursor >= pdf_len:
                            break
                        pdf_idx = pdf_orig_idxs[cursor]
                        if 0 <= pdf_idx < len(pdf_words):
                            pdf_words[pdf_idx].assigned_element_id = eid
                            if has_math:
                                pdf_words[pdf_idx].confidence = min(pdf_words[pdf_idx].confidence, 0.8)
                        cursor += 1

            elif tag == "insert":
                # token PDF ekstra (akan di-handle classifier)
                continue
            elif tag == "delete":
                continue

    @staticmethod
    def _has_math_content(element: Dict) -> bool:
        json_tree = element.get("dokumen_elemen_json_tree", {})
        content = json_tree.get("content", [])
        return any(isinstance(it, dict) and it.get("type") == "math" for it in content)


class InsertTokenClassifier:
    """Assign token PDF yang tidak punya pasangan di stream DOCX (insert tokens)."""

    def __init__(self, page_heights: Optional[Dict[int, float]] = None):
        # page_heights: {page_num (0-based): height}
        self.page_heights = page_heights or {}

    def classify_and_assign(self, pdf_words: List[PDFWord], element_map: Dict[int, Dict]) -> None:
        unassigned = [w for w in pdf_words if w.assigned_element_id is None]
        for word in unassigned:
            if self._is_page_number(word):
                word.assigned_element_id = -100 - word.page
                word.confidence = 0.95
                continue

            if self._is_list_marker(word):
                nearest = self._find_nearest_list_item(word, pdf_words, element_map)
                if nearest is not None:
                    word.assigned_element_id = nearest
                    word.confidence = 0.9
                    continue

            # fallback: nearest assigned word center on same page
            nearest = self._find_nearest_element_by_bbox(word, pdf_words)
            if nearest is not None:
                word.assigned_element_id = nearest
                word.confidence = 0.75

    def _is_page_number(self, word: PDFWord) -> bool:
        if not word.normalized or word.normalized == EMPTY_TOKEN:
            return False
        if not word.normalized.isdigit():
            return False
        page_height = float(self.page_heights.get(word.page, 842.0))
        return word.bbox[3] > page_height * 0.9

    def _is_list_marker(self, word: PDFWord) -> bool:
        if not word.normalized or word.normalized == EMPTY_TOKEN:
            return False
        text = word.normalized
        patterns = [
            r"^\d+\.$",  # 1.
            r"^\d+\)$",  # 1)
            r"^\([a-z]\)$",  # (a)
            r"^[a-z]\.$",  # a.
            r"^[•\-\*]$",  # bullet
            r"^[ivxlcdm]+\.$",  # roman numeral
        ]
        return any(re.match(p, text, re.IGNORECASE) for p in patterns)

    def _find_nearest_list_item(
        self, word: PDFWord, all_words: List[PDFWord], element_map: Dict[int, Dict]
    ) -> Optional[int]:
        # Kumpulkan candidate list-item pada page yang sama
        candidates: List[Tuple[float, int]] = []
        for w in all_words:
            if w.page != word.page:
                continue
            if w.assigned_element_id is None or w.assigned_element_id <= 0:
                continue
            element = element_map.get(w.assigned_element_id)
            if not element:
                continue
            if "list-item" not in element.get("dokumen_elemen_type", ""):
                continue
            y_dist = abs(word.bbox[1] - w.bbox[1])
            candidates.append((y_dist, w.assigned_element_id))

        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        # threshold y-dist (pixel) – bisa dituning
        return candidates[0][1] if candidates[0][0] < 25 else None

    def _find_nearest_element_by_bbox(self, word: PDFWord, all_words: List[PDFWord]) -> Optional[int]:
        assigned = [
            w
            for w in all_words
            if w.page == word.page and w.assigned_element_id is not None and w.assigned_element_id > 0
        ]
        if not assigned:
            return None

        wc = ((word.bbox[0] + word.bbox[2]) / 2.0, (word.bbox[1] + word.bbox[3]) / 2.0)
        min_dist = float("inf")
        nearest: Optional[int] = None
        for w in assigned:
            cc = ((w.bbox[0] + w.bbox[2]) / 2.0, (w.bbox[1] + w.bbox[3]) / 2.0)
            dist = float(np.hypot(wc[0] - cc[0], wc[1] - cc[1]))
            if dist < min_dist:
                min_dist = dist
                nearest = w.assigned_element_id
        return nearest


class OpenXMLAligner:
    """Main aligner."""

    def __init__(
        self,
        openxml_elements: List[Dict],
        pdf_words: List[Dict],
        page_heights: Optional[Dict[int, float]] = None,
    ):
        self.openxml_elements = openxml_elements
        self.pdf_words_raw = pdf_words
        self.page_heights = page_heights or {}

        # Flatten DOCX/OpenXML
        self.docx_tokens, self.docx_element_ids, self.element_map = OpenXMLFlattener.flatten_all(openxml_elements)

        # Flatten PDF
        self.pdf_words = PDFFlattener.flatten_words(pdf_words)
        self.pdf_tokens = [w.normalized for w in self.pdf_words]

        # Repair stream for alignment + spans mapping
        self.pdf_tokens_repaired, self.repaired_spans = PDFFlattener.repair_token_stream(self.pdf_words)

    def align(self) -> List[PDFWord]:
        # 1) Align using repaired token stream
        opcodes = TokenAligner.align(self.docx_tokens, self.pdf_tokens_repaired)

        # 2) Assign from opcodes to ORIGINAL pdf_words via spans mapping
        TokenAligner.assign_from_opcodes_repaired(
            opcodes,
            self.docx_element_ids,
            self.pdf_words,
            self.element_map,
            self.repaired_spans,
        )

        # 3) Assign insert tokens (page number, numbering, punctuation)
        InsertTokenClassifier(self.page_heights).classify_and_assign(self.pdf_words, self.element_map)

        return self.pdf_words

    def get_coverage_stats(self) -> Dict:
        total = len(self.pdf_words)
        assigned = sum(1 for w in self.pdf_words if w.assigned_element_id is not None)

        confidence_dist: Dict[str, int] = {}
        for w in self.pdf_words:
            if w.assigned_element_id is None:
                continue
            bucket = f"{int(w.confidence * 10) * 10}%"
            confidence_dist[bucket] = confidence_dist.get(bucket, 0) + 1

        return {
            "total_words": total,
            "assigned_words": assigned,
            "coverage": assigned / total if total else 0.0,
            "confidence_distribution": confidence_dist,
        }

    def get_element_words(self, element_id: int) -> List[PDFWord]:
        return [w for w in self.pdf_words if w.assigned_element_id == element_id]

    def get_element_bbox(self, element_id: int) -> Optional[Tuple[float, float, float, float]]:
        words = self.get_element_words(element_id)
        if not words:
            return None
        try:
            boxes = [Box(w.bbox[0], w.bbox[1], w.bbox[2], w.bbox[3]) for w in words]
            u = unary_union(boxes)
            return (u.bounds[0], u.bounds[1], u.bounds[2], u.bounds[3])
        except Exception:
            x0 = min(w.bbox[0] for w in words)
            y0 = min(w.bbox[1] for w in words)
            x1 = max(w.bbox[2] for w in words)
            y1 = max(w.bbox[3] for w in words)
            return (x0, y0, x1, y1)
