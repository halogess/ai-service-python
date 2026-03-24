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


from .target_assignment_loading_mixin import MergingExtractionTargetAssignmentLoadingMixin
from .target_assignment_scoring_mixin import MergingExtractionTargetAssignmentScoringMixin


class MergingExtractionTargetAssignmentMixin(
    MergingExtractionTargetAssignmentLoadingMixin,
    MergingExtractionTargetAssignmentScoringMixin,
):


    pass
