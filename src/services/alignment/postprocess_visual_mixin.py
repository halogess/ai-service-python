from copy import deepcopy
import os
import re

class AlignmentPostprocessVisualMixin:
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
            if (
                pdf_units[idx].get('item_type') == 'shape' and
                not pdf_units[idx].get('is_chart_visual') and
                not pdf_units[idx].get('is_docling_picture_area') and
                not pdf_units[idx].get('suppress_text_alignment')
            )
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
                'item_idx': cluster_units[0].get('item_idx', cluster[0]),
                'item_type': 'shape',
                'text': '',
                'bbox': merged_bbox,
                'matched_count': 0,
                'score': 0,
                'is_cell': False,
                'absorbed': True,
                'source_item_type': cluster_units[0].get('source_item_type'),
                'is_chart_visual': any(unit.get('is_chart_visual', False) for unit in cluster_units),
                'is_docling_picture_area': any(unit.get('is_docling_picture_area', False) for unit in cluster_units),
                'suppress_text_alignment': any(unit.get('suppress_text_alignment', False) for unit in cluster_units),
                'debug': {}
            }

            target_alignment.setdefault('matched_pdf_units', []).append(merged_unit)
            target_alignment['matched_pdf_units'].sort(key=lambda x: x.get('item_idx', 0))
            self._recompute_alignment_bboxes(target_alignment)

        return alignments, remaining_unaligned, debug

    def _attach_chart_visuals_to_chart_alignments(
        self,
        alignments,
        unaligned_pdf_indices,
        pdf_units,
        openxml_units,
        min_openxml_idx=None,
        page_sequence_range=None
    ):
        if not unaligned_pdf_indices or not pdf_units or not openxml_units:
            return alignments, unaligned_pdf_indices, []

        chart_visual_indices = [
            idx for idx in unaligned_pdf_indices
            if pdf_units[idx].get('is_chart_visual') and pdf_units[idx].get('bbox')
        ]
        if not chart_visual_indices:
            return alignments, unaligned_pdf_indices, []

        seq_min = seq_max = None
        if page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range

        aligned_sequences = sorted(
            {
                self._get_alignment_sequence(alignment)
                for alignment in (alignments or [])
                if (
                    not alignment.get('is_table') and
                    alignment.get('matched_pdf_units') and
                    self._get_alignment_sequence(alignment) > 0
                )
            }
        )
        if aligned_sequences:
            local_seq_min = aligned_sequences[0] - 1
            local_seq_max = aligned_sequences[-1] + 1
            seq_min = min(seq_min, local_seq_min) if seq_min is not None else local_seq_min
            seq_max = max(seq_max, local_seq_max) if seq_max is not None else local_seq_max

        def center_y(bbox):
            if not bbox or len(bbox) < 4:
                return None
            return (bbox[1] + bbox[3]) / 2

        chart_visual_indices.sort(
            key=lambda idx: (
                pdf_units[idx].get('item_idx', 10**9),
                center_y(pdf_units[idx].get('bbox')) or 10**9,
            )
        )

        def cluster_indices(indices):
            clusters = []
            cluster = []
            previous_idx = None
            previous_bbox = None
            for idx in indices:
                unit = pdf_units[idx]
                bbox = unit.get('bbox')
                item_idx = unit.get('item_idx')
                should_join = False
                if cluster:
                    if (
                        previous_idx is not None and item_idx is not None and
                        (item_idx - previous_idx) <= 1
                    ):
                        should_join = True
                    elif previous_bbox and bbox:
                        should_join = self._bbox_y_overlap_ratio(previous_bbox, bbox) > 0.3
                if cluster and not should_join:
                    clusters.append(cluster)
                    cluster = []
                cluster.append(idx)
                previous_idx = item_idx
                previous_bbox = bbox
            if cluster:
                clusters.append(cluster)
            return clusters

        def build_matched_unit(pdf_unit):
            return {
                'pdf_unit_id': pdf_unit.get('unit_id'),
                'item_idx': pdf_unit.get('item_idx'),
                'item_type': pdf_unit.get('item_type'),
                'source_item_type': pdf_unit.get('source_item_type'),
                'text': pdf_unit.get('text'),
                'bbox': pdf_unit.get('bbox'),
                'matched_count': 0,
                'score': 0,
                'is_cell': pdf_unit.get('is_cell', False),
                'is_hline_table_unit': pdf_unit.get('is_hline_table_unit', False),
                'row': pdf_unit.get('row'),
                'col': pdf_unit.get('col'),
                'is_chart_visual': True,
                'visual_only_match': True,
                'debug': {}
            }

        def alignment_needs_visual_units(alignment):
            units = alignment.get('matched_pdf_units') or []
            if not units:
                return True
            for unit in units:
                if unit.get('is_chart_visual'):
                    return False
                if unit.get('item_type') in ('image', 'shape', 'hline_table'):
                    return False
            return True

        def resolve_target_priority(target):
            target_type = target.get('target_type')
            if target_type == 'visual_slot':
                return 1
            if target_type == 'chart':
                text = ''
                if target.get('alignment'):
                    text = target['alignment'].get('element_text') or ''
                else:
                    text = (target.get('openxml_unit') or {}).get('text') or ''
                return 0 if self._normalize_text(text).strip() else 2
            return 3

        chart_clusters = cluster_indices(chart_visual_indices)
        if not chart_clusters:
            return alignments, unaligned_pdf_indices, []

        existing_chart_targets = []
        for alignment in alignments:
            target_type = None
            if alignment.get('is_openxml_chart'):
                target_type = 'chart'
            elif alignment.get('is_openxml_visual_slot'):
                target_type = 'visual_slot'
            if not target_type or not alignment_needs_visual_units(alignment):
                continue
            existing_chart_targets.append({
                'alignment': alignment,
                'openxml_idx': (alignment.get('openxml_indices') or [alignment.get('openxml_idx')])[0],
                'openxml_unit': None,
                'target_type': target_type,
                'target_priority': resolve_target_priority({
                    'alignment': alignment,
                    'target_type': target_type,
                }),
            })

        used_openxml_indices = self._collect_matched_openxml_indices(alignments)
        seen_elem_ids = {
            target['alignment'].get('element_id')
            for target in existing_chart_targets
            if target.get('alignment')
        }
        chart_candidates = []
        for openxml_idx, openxml_unit in enumerate(openxml_units):
            if not (
                openxml_unit.get('is_openxml_chart') or
                openxml_unit.get('is_openxml_visual_slot')
            ):
                continue
            if openxml_idx in used_openxml_indices:
                continue
            if min_openxml_idx is not None and openxml_idx < int(min_openxml_idx or 0):
                continue
            elem_seq = openxml_unit.get('elem_seq')
            if seq_min is not None and seq_max is not None:
                if elem_seq is None or elem_seq < seq_min or elem_seq > seq_max:
                    continue
            elem_id = openxml_unit.get('elem_id')
            if elem_id in seen_elem_ids:
                continue
            target_type = 'chart' if openxml_unit.get('is_openxml_chart') else 'visual_slot'
            if target_type == 'visual_slot' and not self._is_valid_visual_slot_target(
                openxml_units,
                openxml_idx,
                page_sequence_range=(seq_min, seq_max)
            ):
                continue
            seen_elem_ids.add(elem_id)
            chart_candidates.append({
                'alignment': None,
                'openxml_idx': openxml_idx,
                'openxml_unit': openxml_unit,
                'target_type': target_type,
                'target_priority': resolve_target_priority({
                    'openxml_unit': openxml_unit,
                    'target_type': target_type,
                }),
            })

        chart_targets = existing_chart_targets + chart_candidates
        chart_targets.sort(
            key=lambda target: (
                target.get('target_priority', 1),
                (
                    (target.get('alignment') or {}).get('element_sequence')
                    if target.get('alignment')
                    else (target.get('openxml_unit') or {}).get('elem_seq')
                ) or 0,
                target.get('openxml_idx') or 0,
            )
        )
        if not chart_targets:
            return alignments, unaligned_pdf_indices, []

        remaining_unaligned = set(unaligned_pdf_indices or [])
        debug = []
        for cluster, target in zip(chart_clusters, chart_targets):
            cluster_units = [pdf_units[idx] for idx in cluster if pdf_units[idx].get('bbox')]
            if not cluster_units:
                continue

            matched_units = [build_matched_unit(unit) for unit in cluster_units]
            merged_bbox = self._merge_bboxes([unit.get('bbox') for unit in cluster_units])

            if target.get('alignment'):
                alignment = target['alignment']
                alignment.setdefault('matched_pdf_units', []).extend(matched_units)
                alignment['matched_pdf_units'].sort(key=lambda unit: unit.get('item_idx', -1))
                alignment['is_chart_visual_attachment'] = True
                alignment['matched_by_visual_only'] = True
                if target.get('target_type') == 'visual_slot':
                    alignment['is_openxml_visual_slot'] = True
                    alignment['is_image_part'] = True
                    alignment['visual_slot_promoted'] = True
                    alignment['repair_reason'] = 'visual_slot_attach'
                self._recompute_alignment_bboxes(alignment)
                mode = 'augment_existing'
                element_id = alignment.get('element_id')
                element_sequence = alignment.get('element_sequence')
            else:
                openxml_unit = target.get('openxml_unit') or {}
                target_type = target.get('target_type')
                alignment = {
                    'element_id': openxml_unit.get('elem_id'),
                    'element_sequence': openxml_unit.get('elem_seq'),
                    'element_type': openxml_unit.get('elem_type'),
                    'is_table': False,
                    'element_text': openxml_unit.get('text', ''),
                    'matched_pdf_units': matched_units,
                    'merged_bbox': merged_bbox,
                    'cells': None,
                    'is_text_part': bool(openxml_unit.get('is_text_part', False)),
                    'is_image_part': target_type == 'visual_slot',
                    'unit_id': str(openxml_unit.get('elem_id')),
                    'openxml_indices': [target.get('openxml_idx')],
                    'openxml_idx': target.get('openxml_idx'),
                    'image_index': openxml_unit.get('image_index'),
                    'font_families': openxml_unit.get('font_families', []),
                    'style_ids': openxml_unit.get('style_ids', []),
                    'is_code_font': openxml_unit.get('is_code_font', False),
                    'is_code_style': openxml_unit.get('is_code_style', False),
                    'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                    'is_openxml_chart': bool(openxml_unit.get('is_openxml_chart', False)),
                    'is_openxml_visual_slot': bool(openxml_unit.get('is_openxml_visual_slot', False)),
                    'is_chart_caption_text': bool(openxml_unit.get('is_chart_caption_text', False)),
                    'is_chart_visual_attachment': True,
                    'matched_by_visual_only': True,
                    'visual_slot_promoted': target_type == 'visual_slot',
                    'repair_reason': 'visual_slot_attach' if target_type == 'visual_slot' else 'chart_visual_attach',
                }
                alignments.append(alignment)
                mode = 'create_new'
                element_id = openxml_unit.get('elem_id')
                element_sequence = openxml_unit.get('elem_seq')

            for pdf_idx in cluster:
                remaining_unaligned.discard(pdf_idx)

            debug.append({
                'mode': mode,
                'element_id': element_id,
                'element_sequence': element_sequence,
                'openxml_idx': target.get('openxml_idx'),
                'cluster_pdf_indices': list(cluster),
                'cluster_item_indices': [pdf_units[idx].get('item_idx') for idx in cluster],
                'cluster_item_types': [pdf_units[idx].get('item_type') for idx in cluster],
                'cluster_source_item_types': [pdf_units[idx].get('source_item_type') for idx in cluster],
                'target_type': target.get('target_type') or 'chart',
            })

        if debug:
            alignments.sort(key=lambda alignment: alignment.get('element_sequence') or 0)
        return alignments, sorted(remaining_unaligned), debug

    def _prune_visual_alignment_text_units(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments:
            return alignments, unaligned_pdf_indices, []

        vertical_gap_max = self._read_float_env(
            'ALIGNMENT_VISUAL_CAPTION_BAND_MAX_GAP',
            48.0,
            min_value=0.0
        )
        x_overlap_min = self._read_float_env(
            'ALIGNMENT_VISUAL_CAPTION_BAND_MIN_X_OVERLAP',
            0.5,
            min_value=0.0,
            max_value=1.0
        )

        previous_keys = self._collect_alignment_unit_keys(alignments)
        debug = []

        for alignment in alignments:
            if not self._is_visual_target_alignment(alignment):
                continue

            units = list(alignment.get('matched_pdf_units') or [])
            if not units:
                continue

            visual_bbox = self._alignment_visual_bbox(alignment)
            if not visual_bbox:
                continue

            figure_key = self._extract_figure_key(alignment.get('element_text'))
            kept_units = []
            dropped_units = []
            kept_caption_units = 0

            for unit in units:
                if self._is_visual_pdf_unit(unit):
                    kept_units.append(unit)
                    continue

                unit_text = str(unit.get('text') or '').strip()
                unit_bbox = unit.get('bbox')
                unit_figure_key = self._extract_figure_key(unit_text)

                if figure_key and unit_figure_key and unit_figure_key != figure_key:
                    dropped_units.append(unit)
                    continue

                if unit_figure_key and figure_key and unit_figure_key == figure_key:
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                in_caption_band = self._is_caption_band_bbox(
                    unit_bbox,
                    visual_bbox,
                    vertical_gap_max=vertical_gap_max,
                    x_overlap_min=x_overlap_min
                )
                if in_caption_band:
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                if self._is_caption_like_text(unit_text):
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                dropped_units.append(unit)

            if not dropped_units:
                continue

            alignment['matched_pdf_units'] = sorted(
                kept_units,
                key=lambda unit: unit.get('item_idx', -1)
            )
            self._recompute_alignment_bboxes(alignment)
            alignment['repair_reason'] = 'picture_overlap_prune'
            debug.append({
                'element_id': alignment.get('element_id'),
                'element_sequence': alignment.get('element_sequence'),
                'dropped_unit_count': len(dropped_units),
                'kept_unit_count': len(kept_units),
                'kept_caption_unit_count': kept_caption_units,
                'visual_bbox': list(visual_bbox) if visual_bbox else None,
            })

        if debug:
            unaligned_pdf_indices = self._restore_dropped_alignment_units(
                previous_keys,
                alignments,
                unaligned_pdf_indices,
                pdf_units
            )
        return alignments, unaligned_pdf_indices, debug
