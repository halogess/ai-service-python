from typing import Dict, List

from services.docling_fusion_collect_mixin import DoclingFusionCollectMixin
from services.docling_fusion_matching_mixin import DoclingFusionMatchingMixin
from services.docling_fusion_postprocess_mixin import DoclingFusionPostprocessMixin


class DoclingFusionRulesMixin(
    DoclingFusionCollectMixin,
    DoclingFusionMatchingMixin,
    DoclingFusionPostprocessMixin,
):
    def fuse_alignments_with_docling(
        self,
        alignments: List[Dict],
        header_footer_units: List[Dict],
        docling_predictions: List[Dict]
    ) -> List[Dict]:
        fused_results = []
        has_docling = bool(docling_predictions)
        all_aligned_items = self._collect_aligned_items(alignments, header_footer_units)
        used_indices = set()

        if has_docling:
            for doc_item in docling_predictions:
                doc_bbox = doc_item.get('bbox')
                if not doc_bbox:
                    continue

                matching_items = self._collect_docling_matching_items(
                    all_aligned_items,
                    doc_bbox,
                    used_indices,
                )
                if not matching_items:
                    continue

                if doc_item.get('label') == 'table':
                    matching_items = self._canonicalize_table_matches(matching_items)

                for matched in matching_items:
                    used_indices.add(matched['idx'])

                context = self._summarize_docling_matching_items(doc_item, matching_items)
                if context['should_merge']:
                    fused_results.append(
                        self._build_merged_docling_result(matching_items, doc_item, context)
                    )
                else:
                    fused_results.extend(
                        self._build_individual_docling_results(matching_items, doc_item)
                    )

        self._append_unmatched_fusion_items(fused_results, all_aligned_items, used_indices)

        if has_docling:
            self._promote_picture_prediction_overlaps(fused_results, docling_predictions)
        self._promote_caption_neighbors(fused_results)
        self._downgrade_docling_label_noise(fused_results)
        self._sort_fused_results(fused_results)
        return fused_results
