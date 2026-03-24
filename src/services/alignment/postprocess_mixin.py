from copy import deepcopy
import os
import re

from .postprocess_cleanup_mixin import AlignmentPostprocessCleanupMixin
from .postprocess_paragraph_mixin import AlignmentPostprocessParagraphMixin
from .postprocess_visual_mixin import AlignmentPostprocessVisualMixin



class AlignmentPostprocessMixin(
    AlignmentPostprocessParagraphMixin,
    AlignmentPostprocessCleanupMixin,
    AlignmentPostprocessVisualMixin,
):
    MARKER_ONLY_TEXT_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s*[:.)]?\s*$')
    FIGURE_KEY_RE = re.compile(
        r'\b(?:gambar|figure|fig\.?|tabel|table)\s*(\d+(?:\.\d+)*)',
        re.IGNORECASE
    )
    VISUAL_CAPTION_RE = re.compile(
        r'^\s*(?:gambar|figure|fig\.?|tabel|table)\s*\d+(?:\.\d+)?\b',
        re.IGNORECASE
    )
    CAPTION_FRAGMENT_LEAD_RE = re.compile(
        r'^\s*\d+\s*(?:gambar|figure|fig\.?|tabel|table)\s*\d',
        re.IGNORECASE
    )

