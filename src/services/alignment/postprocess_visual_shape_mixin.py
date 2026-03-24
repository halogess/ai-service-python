from copy import deepcopy
import os
import re


class AlignmentPostprocessVisualShapeMixin:


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
