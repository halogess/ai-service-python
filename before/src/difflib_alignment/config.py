"""Configuration constants for alignment"""

import re

# Token regex untuk tokenisasi
TOKEN_RE = re.compile(
    r"\d*[A-Z\u00c0-\u00df\u0391-\u03a9][A-Za-z\u00c0-\u00ff\u0370-\u03ff_]*(?=\d{1,2}(?!\d))|"
    r"\d*[A-Za-z\u00c0-\u00ff\u0370-\u03ff_]+\d*|\d+(?:\.\d+)*|[^\w\s]",
    flags=re.UNICODE
)

# Formula alignment config
FORMULA_EXPAND_VERT_TOP_RATIO = 0.5
FORMULA_EXPAND_VERT_BOTTOM_RATIO = 0.3
FORMULA_EXPAND_HORZ_RIGHT_RATIO = 0.2
FORMULA_EXPAND_HORZ_RIGHT_MAX_PX = 100
FORMULA_EXPAND_HORZ_LEFT_RATIO = 0.15
FORMULA_EXPAND_HORZ_LEFT_MAX_PX = 60
FORMULA_PAGE_MARGIN_LEFT = 50
FORMULA_PAGE_MARGIN_RIGHT = 520
FORMULA_MERGE_Y_THRESHOLD = 15
TABLE_MERGE_Y_MAX_GAP = 150

# Bbox merging config
DEFAULT_MERGE_X_GAP = 2.0
DEFAULT_MERGE_Y_OVERLAP = 0.5
