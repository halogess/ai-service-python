import os
import re


class AlignmentPreprocessMixin:
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
                            collected.append({
                                'item_idx': item_idx,
                                'item_type': itype,
                                'text': ctext,
                                'bbox': cell.get('bbox'),
                                'is_cell': True,
                                'row': r_idx,
                                'col': c_idx,
                                'table_bbox': ibbox
                            })
            elif itype == 'hline_table':
                # Handle hline_table: prefer cells if available
                cells = idata.get('cells', [])
                rows = idata.get('rows', [])
                if rows:
                    all_text = []
                    for r_idx, row in enumerate(rows):
                        for c_idx, cell in enumerate(row.get('cells', [])):
                            ctext = self._extract_cell_content_text(cell)
                            if ctext.strip():
                                all_text.append(ctext.strip())
                    if all_text:
                        collected.append({
                            'item_idx': item_idx,
                            'item_type': itype,
                            'text': ' '.join(all_text),
                            'bbox': ibbox,
                            'is_cell': False,
                            'is_hline_table_unit': True
                        })
            elif itype == 'shape':
                text = idata.get('text', '')
                image_bbox = idata.get('image_bbox')
                if text.strip() or image_bbox:
                    # If it's an image-shape with no text, use [IMG] placeholder
                    if not text.strip() and image_bbox:
                        text = '[IMG]'
                    collected.append({'item_idx': item_idx, 'item_type': itype, 'text': text, 'bbox': ibbox, 'is_cell': False})
            elif itype == 'image':
                collected.append({'item_idx': item_idx, 'item_type': itype, 'text': None, 'bbox': ibbox, 'is_cell': False, 'is_image': True})

        collected = self._merge_list_markers_with_following_text(collected)
        collected = self._merge_code_like_lines(collected)

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
                    'unit_id': f'pdf_{unit_counter}',
                    'item_idx': item['item_idx'],
                    'item_type': item['item_type'],
                    'text': ph,
                    'text_normalized': ph.lower(),
                    'bbox': item['bbox'],
                    'is_cell': False
                })
            else:
                txt = item['text']
                pdf_units.append({
                    'unit_id': f'pdf_{unit_counter}',
                    'item_idx': item['item_idx'],
                    'item_type': item['item_type'],
                    'text': txt,
                    'text_normalized': self._normalize_text(txt),
                    'bbox': item['bbox'],
                    'is_cell': item.get('is_cell', False),
                    'row': item.get('row'),
                    'col': item.get('col'),
                    'is_hline_table_unit': item.get('is_hline_table_unit', False)
                })
            unit_counter += 1
        return pdf_units

    def _merge_list_markers_with_following_text(self, items):
        if not items:
            return items
        enabled = os.getenv("ALIGNMENT_MERGE_LIST_MARKERS", "").lower() in ("1", "true", "yes", "on")
        if not enabled:
            return items

        marker_re = re.compile(
            r'^\s*(?:\d+(?:\.\d+)*[.)]?|[•\u2022\u2023\u25e6\u2043\u2219\u00b7\u2024\u25aa\u25cf\-–—\*])\s*$'
        )

        def y_overlap(a, b):
            if not a or not b or len(a) < 4 or len(b) < 4:
                return 0.0
            y0 = max(a[1], b[1])
            y1 = min(a[3], b[3])
            overlap = max(0.0, y1 - y0)
            ha = a[3] - a[1]
            hb = b[3] - b[1]
            denom = min(ha, hb) if min(ha, hb) > 0 else 1.0
            return overlap / denom

        merged = []
        idx = 0
        while idx < len(items):
            cur = items[idx]
            text = (cur.get('text') or '').strip()
            if text and marker_re.match(text) and idx + 1 < len(items):
                nxt = items[idx + 1]
                nxt_text = (nxt.get('text') or '').strip()
                if nxt_text and nxt.get('bbox') and cur.get('bbox'):
                    if y_overlap(cur.get('bbox'), nxt.get('bbox')) >= 0.3:
                        merged_text = f"{text} {nxt_text}".strip()
                        merged_bbox = self._merge_bboxes([cur.get('bbox'), nxt.get('bbox')])
                        merged.append({
                            'item_idx': cur.get('item_idx', nxt.get('item_idx')),
                            'item_type': nxt.get('item_type', cur.get('item_type')),
                            'text': merged_text,
                            'bbox': merged_bbox,
                            'is_cell': nxt.get('is_cell', False),
                            'row': nxt.get('row'),
                            'col': nxt.get('col'),
                            'table_bbox': nxt.get('table_bbox'),
                            'is_hline_table_unit': nxt.get('is_hline_table_unit', False)
                        })
                        idx += 2
                        continue
            merged.append(cur)
            idx += 1

        return merged

    def _merge_code_like_lines(self, items):
        if not items:
            return items
        enabled = os.getenv("ALIGNMENT_GROUP_CODE_LINES", "").lower() in ("1", "true", "yes", "on")
        if not enabled:
            return items

        code_kw = re.compile(r'\b(def|class|return|if|else|for|while|import|from|yield|try|except|with)\b')
        code_symbols = set("{}();=<>[]_/.")

        def is_code_like(text):
            if not text:
                return False
            t = text.strip()
            if code_kw.search(t):
                return True
            if t in {"=", "==", "!=", "<=", ">=", "->", ":="}:
                return True
            sym_count = sum(1 for ch in t if ch in code_symbols)
            return sym_count >= 2

        def y_overlap(a, b):
            if not a or not b or len(a) < 4 or len(b) < 4:
                return 0.0
            y0 = max(a[1], b[1])
            y1 = min(a[3], b[3])
            overlap = max(0.0, y1 - y0)
            ha = a[3] - a[1]
            hb = b[3] - b[1]
            denom = min(ha, hb) if min(ha, hb) > 0 else 1.0
            return overlap / denom

        def x_gap(a, b):
            if not a or not b or len(a) < 4 or len(b) < 4:
                return 0.0
            return b[0] - a[2]

        merged = []
        idx = 0
        while idx < len(items):
            cur = items[idx]
            if cur.get('item_type') in {'image', 'shape', 'table', 'hline_table'} or not cur.get('bbox'):
                merged.append(cur)
                idx += 1
                continue

            group = [cur]
            j = idx + 1
            while j < len(items):
                nxt = items[j]
                if nxt.get('item_type') in {'image', 'shape', 'table', 'hline_table'} or not nxt.get('bbox'):
                    break
                if y_overlap(group[-1].get('bbox'), nxt.get('bbox')) < 0.5:
                    break
                if x_gap(group[-1].get('bbox'), nxt.get('bbox')) > 120:
                    break
                group.append(nxt)
                j += 1

            if len(group) == 1:
                merged.append(cur)
                idx += 1
                continue

            if any(is_code_like(g.get('text') or '') for g in group):
                merged_text = ' '.join((g.get('text') or '').strip() for g in group if g.get('text')).strip()
                merged_bbox = self._merge_bboxes([g.get('bbox') for g in group])
                merged.append({
                    'item_idx': group[0].get('item_idx'),
                    'item_type': group[0].get('item_type'),
                    'text': merged_text,
                    'bbox': merged_bbox,
                    'is_cell': group[0].get('is_cell', False),
                    'row': group[0].get('row'),
                    'col': group[0].get('col'),
                    'table_bbox': group[0].get('table_bbox'),
                    'is_hline_table_unit': group[0].get('is_hline_table_unit', False)
                })
            else:
                merged.extend(group)
            idx = j

        return merged

    def _extract_cell_content_text(self, cell):
        texts = []
        content = cell.get('content', [])
        image_counter = 0
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get('type') == 'text' and c.get('text'):
                        texts.append(c.get('text'))
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
                    'item_idx': cluster[0]['item_idx'],
                    'item_type': 'shape',
                    'text': mtext,
                    'bbox': mbbox,
                    'is_cell': False,
                    'row': None,
                    'col': None
                })
        return merged

    def _merge_bboxes(self, bboxes):
        valid = [b for b in bboxes if b and len(b) >= 4]
        if not valid:
            return None
        return [
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid)
        ]

    def _is_item_in_header_footer_zone(self, bbox, section_data, page_height_pt=842):
        if not bbox or len(bbox) < 4 or not section_data:
            return False, False
        twips = 20
        margin_top = (getattr(section_data, 'dsec_margin_top_twips', 0) or 0) / twips
        margin_bottom = (getattr(section_data, 'dsec_margin_bottom_twips', 0) or 0) / twips
        sec_h = getattr(section_data, 'dsec_page_height_twips', 0)
        ph = (sec_h / twips) if sec_h else page_height_pt
        yc = (bbox[1] + bbox[3]) / 2
        return yc < margin_top, yc > (ph - margin_bottom)

    def _filter_header_footer_items(self, pdf_units, section, page_height):
        if not section:
            return pdf_units, []
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
