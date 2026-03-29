import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DoclingFusionHeuristicsMixin:
    @staticmethod
    def _canonicalize_label(label: str) -> str:
        normalized = str(label or '').strip().lower()
        if normalized == 'paragraph':
            return 'text'
        return normalized

    def fallback_label(self, item: Dict) -> str:
        """
        Generate fallback label from alignment element_type when no Docling match.
        
        Args:
            item: Aligned item with element_type, source, zone, etc.
            
        Returns:
            Appropriate label string
        """
        if not item:
            return 'text'
        
        # Header/footer source
        if item.get('source') == 'header_footer' and item.get('zone'):
            return 'page_header' if item['zone'] == 'header' else 'page_footer'
        
        element_type = str(item.get('element_type', '')).lower()
        
        if 'table' in element_type:
            return 'table'
        if 'list' in element_type:
            return 'list_item'
        if 'caption' in element_type:
            return 'caption'
        if 'title' in element_type:
            return 'title'
        if 'header' in element_type:
            return 'section_header'
        if 'footer' in element_type:
            return 'page_footer'
        if 'formula' in element_type:
            return 'formula'
        if 'code' in element_type:
            return 'code'
        if 'paragraph' in element_type:
            return 'text'

        return 'text'

    def is_picture_area(self, item: Dict) -> bool:
        """
        Check if item contains image or shape content.
        
        Args:
            item: Aligned item to check
            
        Returns:
            True if item has image/shape content
        """
        if not item:
            return False
        return bool(
            item.get('is_image_part') or 
            item.get('has_pdf_image') or 
            item.get('has_shape_units') or
            item.get('is_openxml_chart') or
            item.get('is_openxml_visual_slot')
        )

    def _is_text_only_item(self, item: Dict) -> bool:
        if not item:
            return False
        if item.get('source') != 'alignment':
            return False
        element_type = str(item.get('element_type', '')).lower()
        if 'table' in element_type or item.get('has_table_units'):
            return False
        return not (
            item.get('is_image_part') or
            item.get('has_pdf_image') or
            item.get('has_shape_units') or
            item.get('is_openxml_chart') or
            item.get('is_openxml_visual_slot')
        )

    def _is_caption_candidate(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return bool(self.CAPTION_TEXT_REGEX.match(text.strip()))

    def _extract_figure_key(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        match = re.search(
            r'\b(?:gambar|figure|fig\.?|tabel|table)\s*(\d+(?:\.\d+)*)',
            str(text),
            re.IGNORECASE
        )
        if not match:
            return None
        prefix = str(match.group(0)).split()[0].lower().rstrip('.')
        return f"{prefix}:{match.group(1)}"

    @staticmethod
    def _narrative_word_count(text: Optional[str]) -> int:
        if not text:
            return 0
        return len([part for part in re.split(r'\s+', str(text).strip()) if part])

    def _is_narrative_text(
        self,
        text: Optional[str],
        min_chars: int = 80,
        min_words: int = 8
    ) -> bool:
        if not text:
            return False
        normalized = re.sub(r'\s+', ' ', str(text)).strip()
        if len(normalized) < min_chars:
            return False
        return self._narrative_word_count(normalized) >= min_words

    def _canonicalize_table_matches(self, matching_items: List[Dict]) -> List[Dict]:
        """
        Resolve table split where one Docling table overlaps multiple OpenXML table elements.
        Keep per-cell output, but rewrite small/header-only fragments to dominant element.
        """
        if not matching_items:
            return matching_items

        table_like_count = sum(
            1 for matched in matching_items
            if self._is_table_element_item((matched or {}).get('item'))
        )
        if table_like_count != len(matching_items):
            if table_like_count > 0:
                logger.debug(
                    "Skip table canonicalization for mixed candidates: table_like=%s non_table_like=%s",
                    table_like_count,
                    len(matching_items) - table_like_count
                )
            return matching_items

        groups: Dict[Any, List[Dict]] = {}
        for matched in matching_items or []:
            item = matched.get('item') or {}
            elem_id = item.get('element_id')
            if elem_id is None or not self._is_table_element_item(item):
                continue
            groups.setdefault(elem_id, []).append(matched)

        if len(groups) <= 1:
            return matching_items

        group_stats: Dict[Any, Dict[str, float]] = {}
        for elem_id, matches in groups.items():
            rows = [m.get('item', {}).get('row') for m in matches]
            row_ints = [r for r in rows if isinstance(r, int)]
            header_cells = sum(1 for r in rows if r == 0)
            non_header_cells = sum(1 for r in row_ints if r > 0)
            count = len(matches)
            overlap_sum = sum(float(m.get('overlap') or 0.0) for m in matches)
            area_sum = sum(self._bbox_area((m.get('item') or {}).get('bbox')) for m in matches)
            group_stats[elem_id] = {
                'count': count,
                'header_cells': header_cells,
                'non_header_cells': non_header_cells,
                'overlap_sum': overlap_sum,
                'area_sum': area_sum
            }

        canonical_elem_id = max(
            group_stats.items(),
            key=lambda kv: (
                kv[1]['non_header_cells'],
                kv[1]['count'],
                kv[1]['overlap_sum'],
                kv[1]['area_sum']
            )
        )[0]
        canonical_matches = groups.get(canonical_elem_id) or []
        if not canonical_matches:
            return matching_items

        canonical_item = None
        for matched in canonical_matches:
            item = matched.get('item') or {}
            row = item.get('row')
            if isinstance(row, int) and row > 0:
                canonical_item = item
                break
        if canonical_item is None:
            canonical_item = canonical_matches[0].get('item')
        if not canonical_item:
            return matching_items

        canonical_count = group_stats[canonical_elem_id]['count']
        canonical_non_header = group_stats[canonical_elem_id]['non_header_cells']

        for elem_id, matches in groups.items():
            if elem_id == canonical_elem_id:
                continue
            stats = group_stats.get(elem_id) or {}
            count = int(stats.get('count') or 0)
            non_header = int(stats.get('non_header_cells') or 0)
            ratio = (canonical_count / count) if count > 0 else float('inf')

            is_header_only_fragment = (
                non_header == 0 and
                count <= self.TABLE_HEADER_FRAGMENT_MAX_CELLS
            )
            is_small_dominated_fragment = (
                count <= self.TABLE_FRAGMENT_MAX_CELLS and
                canonical_non_header > 0 and
                ratio >= self.TABLE_DOMINANCE_MIN_RATIO
            )
            if not (is_header_only_fragment or is_small_dominated_fragment):
                continue

            for matched in matches:
                item = matched.get('item')
                if not isinstance(item, dict):
                    continue
                if item.get('element_id') == canonical_item.get('element_id'):
                    continue
                item['table_canonical_from_element_id'] = item.get('element_id')
                item['table_canonical_from_sequence'] = item.get('element_sequence')
                item['element_id'] = canonical_item.get('element_id')
                item['element_sequence'] = canonical_item.get('element_sequence')
                item['element_type'] = canonical_item.get('element_type')
                item['openxml_idx'] = canonical_item.get('openxml_idx')

        return matching_items

    def _has_code_font_hint(self, item: Dict) -> bool:
        if not item:
            return False
        if item.get('is_code_like_openxml') or item.get('is_code_font') or item.get('is_code_style'):
            return True

        for font_name in item.get('font_families') or []:
            font_norm = str(font_name).strip().lower()
            if any(marker in font_norm for marker in self.CODE_FONT_MARKERS):
                return True

        for style_id in item.get('style_ids') or []:
            style_norm = str(style_id).strip().lower().replace(' ', '')
            if any(marker in style_norm for marker in self.CODE_STYLE_MARKERS):
                return True

        return False

    def _looks_like_code_text(self, text: Optional[str]) -> bool:
        if not text:
            return False
        text = str(text).strip()
        if len(text) < 4:
            return False
        if self.CODE_KEYWORD_REGEX.search(text):
            return True
        if '\t' in text and any(ch in text for ch in '{}[]();='):
            return True

        symbol_count = sum(1 for ch in text if ch in '{}[]();=<>:+-*/%#@\\')
        symbol_ratio = symbol_count / max(1, len(text))
        if symbol_count >= 3 and symbol_ratio >= 0.08 and any(ch in text for ch in '{}[]();='):
            return True

        lowered = text.lower()
        if text.endswith(':') and re.match(r'^(if|for|while|def|class)\b', lowered):
            return True

        return False

    def _should_relabel_table_to_code(self, matching_items: List[Dict]) -> bool:
        """
        Detect Docling false-positive table labels over code blocks.
        Trigger only when matched OpenXML items are non-table, and code-style hints exist.
        """
        if not matching_items:
            return False

        items = [m.get('item') or {} for m in matching_items]

        # Never override real table alignments.
        for item in items:
            element_type = str(item.get('element_type', '')).lower()
            if item.get('source') == 'cell':
                return False
            if item.get('has_table_units'):
                return False
            if 'table' in element_type:
                return False

        code_font_hits = sum(1 for item in items if self._has_code_font_hint(item))
        if code_font_hits > 0:
            return True

        # Fallback when font metadata is unavailable but text is strongly code-like.
        code_text_hits = sum(1 for item in items if self._looks_like_code_text(item.get('text')))
        return code_text_hits == len(items) and len(items) <= 3

    @staticmethod
    def _normalized_element_type(item: Optional[Dict]) -> str:
        if not item:
            return ''
        return str(item.get('element_type') or '').strip().lower()

    def _is_table_element_item(self, item: Optional[Dict]) -> bool:
        if not item:
            return False
        if item.get('source') == 'cell' or item.get('has_table_units'):
            return True
        element_type = self._normalized_element_type(item)
        return 'table' in element_type

    def _is_picture_element_item(self, item: Optional[Dict]) -> bool:
        if not item:
            return False
        if item.get('is_image_part') or item.get('has_pdf_image'):
            return True
        if item.get('has_shape_units') and (
            item.get('is_openxml_chart') or item.get('is_openxml_visual_slot')
        ):
            return True
        if item.get('is_openxml_visual_slot'):
            return True
        element_type = self._normalized_element_type(item)
        return any(marker in element_type for marker in ('image', 'picture', 'figure', 'gambar'))

    def _is_list_element_item(self, item: Optional[Dict]) -> bool:
        element_type = self._normalized_element_type(item)
        return 'list' in element_type

    def _resolve_table_prediction_label(self, matching_items: List[Dict]) -> str:
        """
        Resolve Docling `table` prediction using dokumen_elemen metadata.

        Priority:
        1) real table element -> table
        2) image/picture element -> picture
        3) list element -> list_item
        4) otherwise -> text
        """
        items = []
        for matched in (matching_items or []):
            if isinstance(matched, dict) and 'item' in matched:
                items.append(matched.get('item') or {})
            elif isinstance(matched, dict):
                items.append(matched)

        if not items:
            return 'text'

        if any(self._is_table_element_item(item) for item in items):
            return 'table'
        if any(self._is_picture_element_item(item) for item in items):
            return 'picture'
        if any(self._is_list_element_item(item) for item in items):
            return 'list_item'
        return 'text'
