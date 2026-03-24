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


from .fusion_duplicate_sync_mixin import MergingExtractionFusionDuplicateSyncMixin
from .fusion_picture_repair_mixin import MergingExtractionFusionPictureRepairMixin
from .fusion_table_collapse_mixin import MergingExtractionFusionTableCollapseMixin


class MergingExtractionFusionRepairsMixin(
    MergingExtractionFusionDuplicateSyncMixin,
    MergingExtractionFusionPictureRepairMixin,
    MergingExtractionFusionTableCollapseMixin,
):


    pass
