from copy import deepcopy
import os
import re


class AlignmentPostprocessCleanupLinesMixin:


    def _matched_unit_key(self, unit):
        if unit.get('pdf_unit_id') is not None:
            return ('pdf_unit_id', unit['pdf_unit_id'])
        if unit.get('item_idx') is not None:
            return ('item_idx', unit['item_idx'])
        bbox = unit.get('bbox')
        if bbox and len(bbox) >= 4:
            return ('bbox', tuple(bbox))
        return None

    def _collect_alignment_unit_keys(self, alignments):
        keys = set()
        for alignment in alignments or []:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    for unit in cell.get('matched_pdf_units', []):
                        key = self._matched_unit_key(unit)
                        if key is not None:
                            keys.add(key)
            else:
                for unit in alignment.get('matched_pdf_units', []):
                    key = self._matched_unit_key(unit)
                    if key is not None:
                        keys.add(key)
        return keys

    def _restore_dropped_alignment_units(self, previous_keys, alignments, unaligned_pdf_indices, pdf_units):
        if not previous_keys:
            return unaligned_pdf_indices
        current_keys = self._collect_alignment_unit_keys(alignments)
        lost_keys = previous_keys - current_keys
        if not lost_keys:
            return unaligned_pdf_indices

        pdf_idx_by_unit_id = {
            u.get('unit_id'): idx
            for idx, u in enumerate(pdf_units or [])
            if u.get('unit_id')
        }
        pdf_idx_by_item_idx = {}
        pdf_idx_by_bbox = {}
        for idx, unit in enumerate(pdf_units or []):
            item_idx = unit.get('item_idx')
            if item_idx is not None:
                pdf_idx_by_item_idx.setdefault(item_idx, []).append(idx)
            bbox = unit.get('bbox')
            if bbox and len(bbox) >= 4:
                pdf_idx_by_bbox.setdefault(tuple(bbox), []).append(idx)

        unaligned_set = set(unaligned_pdf_indices or [])
        for key in lost_keys:
            if key[0] == 'pdf_unit_id':
                idx = pdf_idx_by_unit_id.get(key[1])
                if idx is not None:
                    unaligned_set.add(idx)
            elif key[0] == 'item_idx':
                for idx in pdf_idx_by_item_idx.get(key[1], []):
                    unaligned_set.add(idx)
            elif key[0] == 'bbox':
                for idx in pdf_idx_by_bbox.get(key[1], []):
                    unaligned_set.add(idx)
        return sorted(unaligned_set)

    def _is_line_text_unit(self, unit):
        if unit.get('is_cell'):
            return False
        item_type = unit.get('item_type')
        return item_type not in {'table', 'hline_table', 'shape', 'image'}

    def _bbox_y_overlap_ratio(self, bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0
        overlap_start = max(bbox_a[1], bbox_b[1])
        overlap_end = min(bbox_a[3], bbox_b[3])
        overlap = max(0, overlap_end - overlap_start)
        height_a = bbox_a[3] - bbox_a[1]
        height_b = bbox_b[3] - bbox_b[1]
        denom = min(height_a, height_b)
        if denom <= 0:
            return 0
        return overlap / denom

    def _alignment_single_line_units(self, units):
        if not units:
            return False
        if len(units) == 1:
            return True
        anchor_bbox = units[0].get('bbox')
        if not anchor_bbox:
            return False
        for unit in units[1:]:
            bbox = unit.get('bbox')
            if not bbox:
                continue
            if self._bbox_y_overlap_ratio(anchor_bbox, bbox) < self.LINE_OVERLAP_MIN_RATIO:
                return False
        return True

    def _cluster_units_by_line(self, units):
        if not units:
            return []
        units_sorted = sorted(
            units,
            key=lambda u: (u.get('bbox', [0, 0, 0, 0])[1], u.get('bbox', [0, 0, 0, 0])[0])
        )
        clusters = []
        for unit in units_sorted:
            bbox = unit.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            placed = False
            for cluster in clusters:
                if self._bbox_y_overlap_ratio(bbox, cluster['bbox']) >= self.LINE_OVERLAP_MIN_RATIO:
                    cluster['units'].append(unit)
                    cluster['bbox'] = self._merge_bboxes([cluster['bbox'], bbox])
                    placed = True
                    break
            if not placed:
                clusters.append({'bbox': bbox, 'units': [unit]})
        return clusters

    def _merge_line_overlap_alignments(self, alignments):
        if not alignments:
            return alignments

        line_clusters = []
        for idx, alignment in enumerate(alignments):
            if alignment.get('is_table'):
                continue
            element_text = alignment.get('element_text') or ''
            if self.MARKER_ONLY_TEXT_RE.match(element_text):
                continue
            units = [
                u for u in alignment.get('matched_pdf_units', [])
                if u.get('bbox') and self._is_line_text_unit(u)
            ]
            if not units:
                continue
            for cluster in self._cluster_units_by_line(units):
                cluster_units = cluster['units']
                matched_sum = sum(u.get('matched_count') or 0 for u in cluster_units)
                if matched_sum == 0:
                    matched_sum = sum(u.get('score') or 0 for u in cluster_units)
                line_clusters.append({
                    'alignment_idx': idx,
                    'bbox': cluster['bbox'],
                    'units': cluster_units,
                    'unit_count': len(cluster_units),
                    'matched_sum': matched_sum
                })

        if len(line_clusters) < 2:
            return alignments

        groups = []
        for info in line_clusters:
            placed = False
            for group in groups:
                if self._bbox_y_overlap_ratio(info['bbox'], group['bbox']) >= self.LINE_OVERLAP_MIN_RATIO:
                    group['items'].append(info)
                    placed = True
                    break
            if not placed:
                groups.append({'bbox': info['bbox'], 'items': [info]})

        removed_alignments = set()
        touched = set()
        for group in groups:
            items = group['items']
            alignment_ids = {item['alignment_idx'] for item in items}
            if len(alignment_ids) < 2:
                continue
            winner = max(
                items,
                key=lambda i: (
                    i['unit_count'],
                    i['matched_sum'],
                    self._get_alignment_sequence(alignments[i['alignment_idx']])
                )
            )
            winner_alignment = alignments[winner['alignment_idx']]
            winner_units = winner_alignment.get('matched_pdf_units', [])
            winner_keys = {
                self._matched_unit_key(u)
                for u in winner_units
                if self._matched_unit_key(u) is not None
            }
            for item in items:
                if item is winner:
                    continue
                loser_idx = item['alignment_idx']
                loser_alignment = alignments[loser_idx]
                if loser_alignment in (None,):
                    continue
                keys_to_remove = {
                    self._matched_unit_key(u)
                    for u in item['units']
                    if self._matched_unit_key(u) is not None
                }
                if keys_to_remove:
                    loser_alignment['matched_pdf_units'] = [
                        u for u in loser_alignment.get('matched_pdf_units', [])
                        if self._matched_unit_key(u) not in keys_to_remove
                    ]
                for unit in item['units']:
                    key = self._matched_unit_key(unit)
                    if key is not None and key in winner_keys:
                        continue
                    winner_units.append(unit)
                    if key is not None:
                        winner_keys.add(key)
                touched.add(loser_idx)
                touched.add(winner['alignment_idx'])

        for idx in touched:
            alignment = alignments[idx]
            if alignment.get('is_table'):
                continue
            if not alignment.get('matched_pdf_units'):
                removed_alignments.add(idx)
                continue
            alignment['matched_pdf_units'].sort(key=lambda u: u.get('item_idx', -1))
            self._recompute_alignment_bboxes(alignment)

        if not removed_alignments:
            return alignments
        return [a for i, a in enumerate(alignments) if i not in removed_alignments]
