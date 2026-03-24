import difflib
import json
import logging
import os
import re
from datetime import datetime

from sqlalchemy.orm import Session

from models import (
    Bab,
    Dokumen,
    DokumenElemen,
    DokumenElemenVisual,
    DokumenFormatParagraf,
    DokumenFormatText,
    DokumenNote,
    DokumenPart,
    DokumenSection,
)
from utils.cross_page_claims import analyze_cross_page_entries

logger = logging.getLogger(__name__)


from .claim_geometry_mixin import MergingExtractionClaimGeometryMixin
from .claim_same_page_mixin import MergingExtractionClaimSamePageMixin
from .claim_target_assignment_mixin import MergingExtractionClaimTargetAssignmentMixin
from .claim_backfill_mixin import MergingExtractionClaimBackfillMixin
from .claim_duplicate_mixin import MergingExtractionClaimDuplicateMixin


class MergingExtractionClaimRepairMixin(
    MergingExtractionClaimGeometryMixin,
    MergingExtractionClaimSamePageMixin,
    MergingExtractionClaimTargetAssignmentMixin,
    MergingExtractionClaimBackfillMixin,
    MergingExtractionClaimDuplicateMixin,
):


    pass
