class AlignmentPostprocessMixin:
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
            existing = {
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
                    if key is not None and key in existing:
                        continue
                    winner_units.append(unit)
                    if key is not None:
                        existing.add(key)
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

    def _add_units_to_unaligned(self, unaligned_set, pdf_idx_by_unit_id, units):
        for unit in units or []:
            unit_id = unit.get('pdf_unit_id')
            if unit_id and unit_id in pdf_idx_by_unit_id:
                unaligned_set.add(pdf_idx_by_unit_id[unit_id])

    def _filter_units_by_item_gap(self, units):
        if not units or len(units) < 2:
            return units, [], False

        units_sorted = sorted(units, key=lambda u: u.get('item_idx', -1))
        item_indices = [u.get('item_idx') for u in units_sorted if u.get('item_idx') is not None]
        if len(item_indices) < 2:
            return units_sorted, [], False

        clusters = []
        cluster = [units_sorted[0]]
        prev_idx = units_sorted[0].get('item_idx')
        for unit in units_sorted[1:]:
            idx = unit.get('item_idx')
            if idx is None or prev_idx is None:
                cluster.append(unit)
            elif (idx - prev_idx) <= self.MATCHED_UNIT_MAX_ITEM_GAP:
                cluster.append(unit)
            else:
                clusters.append(cluster)
                cluster = [unit]
            prev_idx = idx
        if cluster:
            clusters.append(cluster)

        if len(clusters) <= 1:
            return units_sorted, [], False

        def cluster_score(items):
            score = sum(u.get('matched_count') or 0 for u in items)
            if score == 0:
                score = sum(u.get('score') or 0 for u in items)
            if score == 0:
                score = len(items)
            return (score, len(items))

        best_cluster = max(clusters, key=cluster_score)
        removed = [u for u in units_sorted if u not in best_cluster]

        if len(best_cluster) < self.MATCHED_UNIT_MIN_CLUSTER_SIZE:
            return [], removed, True
        return best_cluster, removed, False

    def _filter_sparse_matched_units(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments:
            return alignments, unaligned_pdf_indices

        pdf_idx_by_unit_id = {
            u.get('unit_id'): idx
            for idx, u in enumerate(pdf_units or [])
            if u.get('unit_id')
        }
        unaligned_set = set(unaligned_pdf_indices or [])
        filtered = []

        for alignment in alignments:
            if alignment.get('is_table') and alignment.get('cells'):
                new_cells = []
                for cell in alignment.get('cells') or []:
                    units = cell.get('matched_pdf_units', [])
                    kept, removed, drop = self._filter_units_by_item_gap(units)
                    if drop:
                        self._add_units_to_unaligned(unaligned_set, pdf_idx_by_unit_id, units)
                        continue
                    if removed:
                        self._add_units_to_unaligned(unaligned_set, pdf_idx_by_unit_id, removed)
                    cell['matched_pdf_units'] = kept
                    if kept:
                        new_cells.append(cell)
                alignment['cells'] = new_cells
                if not new_cells:
                    continue
                self._recompute_alignment_bboxes(alignment)
                filtered.append(alignment)
            else:
                units = alignment.get('matched_pdf_units', [])
                kept, removed, drop = self._filter_units_by_item_gap(units)
                if drop:
                    self._add_units_to_unaligned(unaligned_set, pdf_idx_by_unit_id, units)
                    continue
                if removed:
                    self._add_units_to_unaligned(unaligned_set, pdf_idx_by_unit_id, removed)
                alignment['matched_pdf_units'] = kept
                self._recompute_alignment_bboxes(alignment)
                filtered.append(alignment)

        return filtered, sorted(unaligned_set)

    def _absorb_unaligned_by_y_overlap(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments or not unaligned_pdf_indices:
            return alignments, unaligned_pdf_indices

        candidates = []
        for alignment in alignments:
            if alignment.get('is_table'):
                continue
            bbox = alignment.get('merged_bbox')
            if not bbox or len(bbox) < 4:
                continue
            center_y = (bbox[1] + bbox[3]) / 2
            candidates.append((alignment, bbox, center_y))

        if not candidates:
            return alignments, unaligned_pdf_indices

        absorbed = set()
        touched = set()
        for idx in unaligned_pdf_indices:
            unit = pdf_units[idx]
            if not self._is_line_text_unit(unit):
                continue
            bbox = unit.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            unit_center_y = (bbox[1] + bbox[3]) / 2

            best = None
            best_score = None
            for alignment, abox, ay in candidates:
                overlap = self._bbox_y_overlap_ratio(bbox, abox)
                if overlap < self.LINE_OVERLAP_MIN_RATIO:
                    continue
                score = (abs(unit_center_y - ay), -overlap)
                if best_score is None or score < best_score:
                    best_score = score
                    best = alignment

            if not best:
                continue

            existing_keys = {
                self._matched_unit_key(u)
                for u in best.get('matched_pdf_units', [])
                if self._matched_unit_key(u) is not None
            }
            unit_entry = {
                'pdf_unit_id': unit['unit_id'],
                'item_idx': unit['item_idx'],
                'item_type': unit['item_type'],
                'text': unit['text'],
                'bbox': unit['bbox'],
                'matched_count': 0,
                'score': 0,
                'is_cell': unit['is_cell'],
                'is_hline_table_unit': unit.get('is_hline_table_unit', False),
                'row': unit.get('row'),
                'col': unit.get('col'),
                'absorbed': True,
                'absorbed_by_overlap': True,
                'debug': {}
            }
            unit_key = self._matched_unit_key(unit_entry)
            if unit_key is None or unit_key in existing_keys:
                continue
            best.setdefault('matched_pdf_units', []).append(unit_entry)
            best['matched_pdf_units'].sort(key=lambda u: u.get('item_idx', -1))
            touched.add(id(best))
            absorbed.add(idx)

        if absorbed:
            for alignment in alignments:
                if id(alignment) in touched:
                    self._recompute_alignment_bboxes(alignment)

        remaining = [i for i in unaligned_pdf_indices if i not in absorbed]
        return alignments, remaining

    def _absorb_unaligned_into_alignments(self, alignments, unaligned_indices, pdf_units):
        if not alignments or not unaligned_indices:
            return alignments, unaligned_indices

        absorbed = set()
        for alignment in alignments:
            if alignment.get('is_table'):
                continue
            bbox = alignment.get('merged_bbox')
            if not bbox or len(bbox) < 4:
                continue
            candidates = []
            for idx in unaligned_indices:
                unit = pdf_units[idx]
                unit_bbox = unit.get('bbox')
                if not unit_bbox or len(unit_bbox) < 4:
                    continue
                if self._is_bbox_inside(unit_bbox, bbox, tol=5):
                    candidates.append(unit)
                    absorbed.add(idx)
            if candidates:
                new_units = [
                    {
                        'pdf_unit_id': u['unit_id'],
                        'item_idx': u['item_idx'],
                        'item_type': u['item_type'],
                        'text': u['text'],
                        'bbox': u['bbox'],
                        'matched_count': 0,
                        'score': 0,
                        'is_cell': u['is_cell'],
                        'is_hline_table_unit': u.get('is_hline_table_unit', False),
                        'row': u.get('row'),
                        'col': u.get('col'),
                        'absorbed': True,
                        'debug': {}
                    }
                    for u in candidates
                ]
                alignment.setdefault('matched_pdf_units', []).extend(new_units)
                alignment['matched_pdf_units'].sort(key=lambda x: x.get('item_idx'))

                if alignment.get('merged_bbox'):
                    merged_bbox = alignment.get('merged_bbox')
                    for u in candidates:
                        unit_bbox = u.get('bbox')
                        if unit_bbox and len(unit_bbox) >= 4:
                            merged_bbox[0] = min(merged_bbox[0], unit_bbox[0])
                            merged_bbox[1] = min(merged_bbox[1], unit_bbox[1])
                            merged_bbox[2] = max(merged_bbox[2], unit_bbox[2])
                            merged_bbox[3] = max(merged_bbox[3], unit_bbox[3])

        remaining = [i for i in unaligned_indices if i not in absorbed]
        print(
            f"[Absorb] Absorbed {len(absorbed)} unaligned PDF units into alignments, "
            f"{len(remaining)} remaining unaligned"
        )
        return alignments, remaining

    def _is_bbox_inside(self, inner, outer, tol=5):
        if not inner or not outer:
            return False
        cx = (inner[0] + inner[2]) / 2
        cy = (inner[1] + inner[3]) / 2
        return (outer[0] - tol <= cx <= outer[2] + tol) and (outer[1] - tol <= cy <= outer[3] + tol)

    def _get_alignment_min_item_idx(self, alignment):
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

    def _get_alignment_sequence(self, alignment):
        seq = alignment.get('element_sequence')
        if seq is None:
            return 0
        try:
            return int(seq)
        except (TypeError, ValueError):
            return 0

    def _is_bbox_fully_contained(self, inner_bbox, outer_bbox, tolerance=2):
        if not inner_bbox or not outer_bbox or len(inner_bbox) < 4 or len(outer_bbox) < 4:
            return False
        if (
            abs(inner_bbox[0] - outer_bbox[0]) < tolerance and
            abs(inner_bbox[1] - outer_bbox[1]) < tolerance and
            abs(inner_bbox[2] - outer_bbox[2]) < tolerance and
            abs(inner_bbox[3] - outer_bbox[3]) < tolerance
        ):
            return False
        return (
            inner_bbox[0] >= outer_bbox[0] - tolerance and
            inner_bbox[1] >= outer_bbox[1] - tolerance and
            inner_bbox[2] <= outer_bbox[2] + tolerance and
            inner_bbox[3] <= outer_bbox[3] + tolerance
        )

    def _is_punctuation_only(self, text):
        if not text:
            return False
        for ch in text:
            if ch.isalnum():
                return False
        return True

    def _cleanup_punctuation_alignments(self, alignments):
        if not alignments:
            return alignments

        cleaned = []
        for alignment in alignments:
            if alignment.get('is_table') or alignment.get('is_image_part'):
                cleaned.append(alignment)
                continue
        text = alignment.get('element_text') or ''
        if self._is_punctuation_only(text):
            stripped = ''.join(text.split())
            if len(stripped) <= 1:
                continue
            if not alignment.get('matched_pdf_units'):
                continue
        cleaned.append(alignment)
        return cleaned

    def _recompute_alignment_bboxes(self, alignment):
        if alignment.get('is_table') and alignment.get('cells'):
            cell_bboxes = [c.get('merged_bbox') for c in alignment.get('cells') if c.get('merged_bbox')]
            alignment['merged_bbox'] = self._merge_bboxes(cell_bboxes) if cell_bboxes else None
            return
        if alignment.get('matched_pdf_units'):
            alignment['merged_bbox'] = self._merge_bboxes(
                [u.get('bbox') for u in alignment.get('matched_pdf_units') if u.get('bbox')]
            )

    def _resolve_shape_alignment_conflicts(self, alignments, pdf_units):
        if not alignments:
            return alignments, []

        shape_indices = []
        for idx, unit in enumerate(pdf_units):
            if unit.get('item_type') == 'shape':
                shape_indices.append(idx)

        if not shape_indices:
            return alignments, []

        pdf_unit_by_id = {u.get('unit_id'): u for u in pdf_units if u.get('unit_id')}
        shape_conflicts = []
        debug = []

        for alignment in alignments:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells', []):
                    for unit in cell.get('matched_pdf_units', []):
                        unit_id = unit.get('pdf_unit_id')
                        if not unit_id:
                            continue
                        pdf_unit = pdf_unit_by_id.get(unit_id)
                        if not pdf_unit or pdf_unit.get('item_type') != 'shape':
                            continue
                        shape_conflicts.append((alignment, unit_id))
            else:
                for unit in alignment.get('matched_pdf_units', []):
                    unit_id = unit.get('pdf_unit_id')
                    if not unit_id:
                        continue
                    pdf_unit = pdf_unit_by_id.get(unit_id)
                    if not pdf_unit or pdf_unit.get('item_type') != 'shape':
                        continue
                    shape_conflicts.append((alignment, unit_id))

        if not shape_conflicts:
            return alignments, debug

        for alignment, unit_id in shape_conflicts:
            pdf_unit = pdf_unit_by_id.get(unit_id)
            if not pdf_unit:
                continue
            shape_item_idx = pdf_unit.get('item_idx')
            if shape_item_idx is None:
                continue
            conflict = {
                'unit_id': unit_id,
                'item_idx': shape_item_idx,
                'element_id': alignment.get('element_id'),
                'element_sequence': alignment.get('element_sequence'),
                'resolved': False
            }
            debug.append(conflict)

        # Remove shape units from non-image alignments
        for alignment, unit_id in shape_conflicts:
            if alignment.get('is_image_part'):
                continue
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells', []):
                    cell_units = cell.get('matched_pdf_units', [])
                    new_units = [u for u in cell_units if u.get('pdf_unit_id') != unit_id]
                    if len(new_units) != len(cell_units):
                        cell['matched_pdf_units'] = new_units
            else:
                units = alignment.get('matched_pdf_units', [])
                new_units = [u for u in units if u.get('pdf_unit_id') != unit_id]
                if len(new_units) != len(units):
                    alignment['matched_pdf_units'] = new_units

        # Recompute bounding boxes after removal
        for alignment, _ in shape_conflicts:
            self._recompute_alignment_bboxes(alignment)

        return alignments, debug

    def _attach_shape_clusters_to_next_alignment(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments or not unaligned_pdf_indices:
            return alignments, unaligned_pdf_indices, []

        shape_indices = [
            idx for idx in unaligned_pdf_indices
            if pdf_units[idx].get('item_type') == 'shape'
        ]
        if not shape_indices:
            return alignments, unaligned_pdf_indices, []

        # Cluster adjacent shape units by item_idx gap
        shape_indices.sort()
        clusters = []
        cluster = [shape_indices[0]]
        for idx in shape_indices[1:]:
            if idx - cluster[-1] <= 1:
                cluster.append(idx)
            else:
                clusters.append(cluster)
                cluster = [idx]
        if cluster:
            clusters.append(cluster)

        remaining_unaligned = [i for i in unaligned_pdf_indices if i not in shape_indices]
        if not clusters:
            return alignments, remaining_unaligned, []

        debug = []
        for cluster in clusters:
            cluster_units = [pdf_units[i] for i in cluster]
            merged_bbox = self._merge_bboxes([u.get('bbox') for u in cluster_units])
            if not merged_bbox:
                continue

            # Find the next alignment after the last shape unit
            target_alignment = None
            for alignment in alignments:
                min_item_idx = self._get_alignment_min_item_idx(alignment)
                if min_item_idx is None:
                    continue
                if min_item_idx > cluster[-1]:
                    target_alignment = alignment
                    break

            if not target_alignment:
                continue

            debug.append({
                'cluster_start': cluster[0],
                'cluster_end': cluster[-1],
                'attached_to': target_alignment.get('element_id')
            })

            # Attach merged unit as shape placeholder
            merged_unit = {
                'pdf_unit_id': f"pdf_shape_cluster_{cluster[0]}",
                'item_idx': cluster[0],
                'item_type': 'shape',
                'text': '',
                'bbox': merged_bbox,
                'matched_count': 0,
                'score': 0,
                'is_cell': False,
                'absorbed': True,
                'debug': {}
            }

            target_alignment.setdefault('matched_pdf_units', []).append(merged_unit)
            target_alignment['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))
            self._recompute_alignment_bboxes(target_alignment)

        return alignments, remaining_unaligned, debug
