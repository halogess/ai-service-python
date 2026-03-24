import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DoclingFusionGeometryMixin:
    def calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate overlap ratio between two bboxes.
        Uses intersection-over-minimum-area ratio.
        
        Args:
            bbox1: [x0, y0, x1, y1]
            bbox2: [x0, y0, x1, y1]
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # Calculate intersection
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])
        
        if x0 >= x1 or y0 >= y1:
            return 0.0
        
        intersection_area = (x1 - x0) * (y1 - y0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Use smaller area for ratio calculation
        min_area = min(bbox1_area, bbox2_area)
        return intersection_area / min_area if min_area > 0 else 0.0

    def get_bbox_margin_zone(self, bbox: List[float]) -> Optional[str]:
        """
        Determine if bbox center is in header or footer margin zone.
        
        Args:
            bbox: [x0, y0, x1, y1]
            
        Returns:
            'header', 'footer', or None (body area)
        """
        if not bbox or len(bbox) < 4 or not self.section_data:
            return None
        
        page_height = self.section_data.get('page_height_pt', 842)
        margin_top = self.section_data.get('margin_top_pt', 72)
        margin_bottom = self.section_data.get('margin_bottom_pt', 72)
        
        # Calculate Y center of bbox
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Header zone: Y center < marginTop
        if y_center < margin_top:
            return 'header'
        
        # Footer zone: Y center > (pageHeight - marginBottom)
        if y_center > page_height - margin_bottom:
            return 'footer'
        
        return None

    def correct_header_footer_label(self, label: str, bbox: List[float]) -> str:
        """
        Correct page_header/page_footer labels if not in appropriate margin zone.
        
        Args:
            label: Original Docling label
            bbox: [x0, y0, x1, y1]
            
        Returns:
            Corrected label (may be changed to 'text')
        """
        if label not in ('page_header', 'page_footer'):
            return label
        
        zone = self.get_bbox_margin_zone(bbox)
        
        if label == 'page_header' and zone != 'header':
            return 'text'
        
        if label == 'page_footer' and zone != 'footer':
            return 'text'
        
        return label

    @staticmethod
    def _bbox_area(bbox: Optional[List[float]]) -> float:
        if not bbox or len(bbox) < 4:
            return 0.0
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        return width * height

    @staticmethod
    def merge_bboxes(bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> Optional[List[float]]:
        """Merge two bboxes into one encompassing bbox."""
        if not bbox1:
            return bbox2
        if not bbox2:
            return bbox1
        return [
            min(bbox1[0], bbox2[0]),  # x0
            min(bbox1[1], bbox2[1]),  # y0
            max(bbox1[2], bbox2[2]),  # x1
            max(bbox1[3], bbox2[3])   # y1
        ]

    def _x_overlap_ratio(self, bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> float:
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        x0 = max(bbox1[0], bbox2[0])
        x1 = min(bbox1[2], bbox2[2])
        if x0 >= x1:
            return 0.0
        w1 = bbox1[2] - bbox1[0]
        w2 = bbox2[2] - bbox2[0]
        min_w = min(w1, w2)
        return (x1 - x0) / min_w if min_w > 0 else 0.0

    def _has_item_above(
        self,
        bbox: List[float],
        items: List[Dict],
        require_x_overlap: bool = True
    ) -> bool:
        for item in items:
            ibox = item.get('bbox')
            if not ibox or len(ibox) < 4:
                continue
            if ibox[3] <= bbox[1]:
                gap = bbox[1] - ibox[3]
                if gap <= self.CAPTION_VERTICAL_GAP_MAX and (
                    not require_x_overlap or
                    self._x_overlap_ratio(bbox, ibox) >= self.CAPTION_X_OVERLAP_MIN
                ):
                    return True
        return False

    def _has_item_below(
        self,
        bbox: List[float],
        items: List[Dict],
        require_x_overlap: bool = True
    ) -> bool:
        for item in items:
            ibox = item.get('bbox')
            if not ibox or len(ibox) < 4:
                continue
            if ibox[1] >= bbox[3]:
                gap = ibox[1] - bbox[3]
                if gap <= self.CAPTION_VERTICAL_GAP_MAX and (
                    not require_x_overlap or
                    self._x_overlap_ratio(bbox, ibox) >= self.CAPTION_X_OVERLAP_MIN
                ):
                    return True
        return False
