from functools import cmp_to_key
from typing import Dict, List


class DoclingFusionPostprocessMixin:
    def _promote_picture_prediction_overlaps(self, fused_results: List[Dict], docling_predictions: List[Dict]):
        picture_preds = [
            doc_item for doc_item in (docling_predictions or [])
            if doc_item.get('label') == 'picture' and doc_item.get('bbox')
        ]
        if not picture_preds:
            return

        for result in fused_results:
            if result.get('label') == 'picture':
                continue
            if (
                not result.get('has_pdf_image') and
                not result.get('is_openxml_chart') and
                not result.get('is_openxml_visual_slot')
            ):
                continue
            bbox = result.get('bbox')
            if not bbox:
                continue
            if any(self.calculate_overlap(bbox, doc_item['bbox']) > 0 for doc_item in picture_preds):
                result['label'] = 'picture'
                result['docling_label'] = 'picture'

    def _promote_caption_neighbors(self, fused_results: List[Dict]):
        picture_results = [r for r in fused_results if r.get('label') == 'picture' and r.get('bbox')]
        if picture_results:
            for result in fused_results:
                if result.get('label') not in ('text', 'paragraph', 'unknown'):
                    continue
                text = (result.get('text') or '').strip()
                if not self._is_caption_candidate(text):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                if (bbox[3] - bbox[1]) > self.CAPTION_LINE_MAX_HEIGHT:
                    continue
                if self._has_item_above(bbox, picture_results) or self._has_item_below(bbox, picture_results):
                    result['label'] = 'caption'

        caption_results = [r for r in fused_results if r.get('label') == 'caption' and r.get('bbox')]
        if picture_results and caption_results:
            for result in fused_results:
                if result.get('label') not in ('text', 'paragraph', 'unknown'):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                if (bbox[3] - bbox[1]) > self.CAPTION_LINE_MAX_HEIGHT:
                    continue
                above_picture = self._has_item_above(bbox, picture_results)
                below_picture = self._has_item_below(bbox, picture_results)
                above_caption = self._has_item_above(
                    bbox,
                    caption_results,
                    require_x_overlap=False,
                )
                below_caption = self._has_item_below(
                    bbox,
                    caption_results,
                    require_x_overlap=False,
                )
                if (above_picture and below_caption) or (above_caption and below_picture):
                    result['label'] = 'caption'

        if picture_results:
            for result in fused_results:
                if result.get('label') != 'caption' or not result.get('is_openxml_chart'):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                caption_key = self._extract_figure_key(result.get('text'))
                if not caption_key:
                    continue

                neighbor_picture = None
                for picture in picture_results:
                    picture_bbox = picture.get('bbox')
                    if not picture_bbox or len(picture_bbox) < 4:
                        continue
                    if (
                        self._has_item_above(bbox, [picture], require_x_overlap=False) or
                        self._has_item_below(bbox, [picture], require_x_overlap=False)
                    ):
                        neighbor_picture = picture
                        break

                if not neighbor_picture:
                    continue

                picture_key = self._extract_figure_key(neighbor_picture.get('text'))
                if picture_key and picture_key != caption_key:
                    result['label'] = 'text'
                    result['element_id'] = None

    def _downgrade_docling_label_noise(self, fused_results: List[Dict]):
        picture_results = [r for r in fused_results if r.get('label') == 'picture' and r.get('bbox')]

        for result in fused_results:
            label = result.get('label')
            text = (result.get('text') or '').strip()
            bbox = result.get('bbox')

            if label == 'table':
                if (
                    self._is_narrative_text(text) and
                    not result.get('has_table_units') and
                    not self._is_table_element_item(result)
                ):
                    result['label'] = 'text'
            elif label == 'picture':
                if (
                    self._is_narrative_text(text) and
                    not result.get('has_pdf_image') and
                    not result.get('has_shape_units') and
                    not result.get('is_openxml_chart') and
                    not result.get('is_openxml_visual_slot')
                ):
                    result['label'] = 'text'
            elif label == 'caption':
                has_picture_neighbor = False
                if bbox and picture_results:
                    has_picture_neighbor = (
                        self._has_item_above(bbox, picture_results) or
                        self._has_item_below(bbox, picture_results)
                    )
                if (
                    self._is_narrative_text(text, min_chars=100, min_words=12) and
                    not has_picture_neighbor
                ):
                    result['label'] = 'text'

    def _sort_fused_results(self, fused_results: List[Dict]):
        def sort_key(item):
            return item.get('bbox') or [0, 0, 0, 0]

        def compare(a, b):
            a_bbox = sort_key(a)
            b_bbox = sort_key(b)
            y_diff = a_bbox[1] - b_bbox[1]
            if abs(y_diff) > 10:
                return -1 if y_diff < 0 else 1
            x_diff = a_bbox[0] - b_bbox[0]
            return -1 if x_diff < 0 else (1 if x_diff > 0 else 0)

        fused_results.sort(key=cmp_to_key(compare))
