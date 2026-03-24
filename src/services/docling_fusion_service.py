"""
Docling Fusion Service
Handles fusion of Docling predictions with alignment bboxes.
Ports logic from classification.html's fuseAlignmentWithDocling() function.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from services.docling_fusion_geometry_mixin import DoclingFusionGeometryMixin
from services.docling_fusion_heuristics_mixin import DoclingFusionHeuristicsMixin
from services.docling_fusion_rules_mixin import DoclingFusionRulesMixin

logger = logging.getLogger(__name__)



class DoclingFusionService(
    DoclingFusionGeometryMixin,
    DoclingFusionHeuristicsMixin,
    DoclingFusionRulesMixin,
):
    """
    Service to fuse Docling classification predictions with alignment bboxes.
    
    Key operations:
    1. Calculate overlap between alignment bboxes and Docling predictions
    2. Correct page_header/page_footer labels based on margin zones
    3. Merge multiple shapes under same Docling 'picture' label
    4. Generate fallback labels from element_type when no Docling match
    """
    
    OVERLAP_THRESHOLD = 0.3  # 30% overlap required for matching
    CAPTION_LINE_MAX_HEIGHT = 24
    CAPTION_VERTICAL_GAP_MAX = 60
    CAPTION_X_OVERLAP_MIN = 0.3
    TABLE_HEADER_FRAGMENT_MAX_CELLS = 6
    TABLE_FRAGMENT_MAX_CELLS = 8
    TABLE_DOMINANCE_MIN_RATIO = 3.0
    CODE_FONT_MARKERS = (
        'courier',
        'lucida',
        'consola',
        'monospace',
        'menlo',
        'monaco',
        'fira code',
        'source code',
        'jetbrains mono',
        'inconsolata',
        'cascadia',
        'terminal',
    )
    CODE_STYLE_MARKERS = (
        'code',
        'algoritma',
        'algorithm',
        'segmenprogram',
        'segmen_program',
        'programcontent',
        'listing',
        'source',
        'monospace',
    )
    CODE_KEYWORD_REGEX = re.compile(
        r'\b(?:'
        r'def|class|return|if|else|elif|for|while|import|from|try|except|finally|with|'
        r'function|const|let|var|public|private|protected|static|void|int|float|double|'
        r'string|select|insert|update|delete|create|join|where|group by|order by'
        r')\b',
        re.IGNORECASE
    )
    CAPTION_TEXT_REGEX = re.compile(
        r'^\s*(?:gambar|figure|fig\.?|tabel|table)\s*\d',
        re.IGNORECASE
    )
    
    def __init__(self, section_data: Optional[Dict] = None):
        """
        Initialize fusion service.
        
        Args:
            section_data: Dict with page margins and dimensions:
                - page_height_pt: Page height in PDF points (default 842)
                - margin_top_pt: Top margin in points (default 72)
                - margin_bottom_pt: Bottom margin in points (default 72)
        """
        self.section_data = section_data or {}
