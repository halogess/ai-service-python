
import os
import json
import logging
import difflib
import re
from datetime import datetime
from sqlalchemy.orm import Session
from models import Bab, Dokumen, DokumenSection, DokumenPart, DokumenElemen, DokumenElemenVisual, DokumenNote, DokumenFormatText, DokumenFormatParagraf
from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from services.docling_service import DoclingService
from services.docling_fusion_service import DoclingFusionService
from services.visualization_service import VisualizationService
from utils.cross_page_claims import analyze_cross_page_entries
from database import SessionLocal
from services.merging_extraction import (
    MergingExtractionClaimRepairMixin,
    MergingExtractionFusionRepairsMixin,
    MergingExtractionPersistenceMixin,
    MergingExtractionStructuralLabelsMixin,
    MergingExtractionTargetAssignmentMixin,
)

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")
VISUALIZATION_OUTPUT = os.getenv("VISUALIZATION_OUTPUT", "visualization_output")


from services.merging_extraction.process_pipeline_mixin import MergingExtractionProcessPipelineMixin
from services.merging_extraction.pipeline_items_mixin import MergingExtractionPipelineItemsMixin
from services.merging_extraction.header_footer_mixin import MergingExtractionHeaderFooterMixin
from services.merging_extraction.labeling_mixin import MergingExtractionLabelingMixin


class MergingExtractionService(
    MergingExtractionProcessPipelineMixin,
    MergingExtractionPipelineItemsMixin,
    MergingExtractionHeaderFooterMixin,
    MergingExtractionLabelingMixin,
    MergingExtractionTargetAssignmentMixin,
    MergingExtractionStructuralLabelsMixin,
    MergingExtractionFusionRepairsMixin,
    MergingExtractionClaimRepairMixin,
    MergingExtractionPersistenceMixin,
):


    FOOTNOTE_LABELS = {"footnote"}

    FOOTNOTE_MATCH_MIN_RATIO = 0.55

    FOOTNOTE_OVERLAP_THRESHOLD = 0.3

    FOOTNOTE_LOG_PATH = os.path.join("logs", "footnote_matches.txt")

    DUPLICATE_SEQUENCE_GAP_THRESHOLD = 2

    SHORT_DUPLICATE_UNIT_LEN = 12

    BAB_TITLE_REGEX = re.compile(r'^\s*bab\b', re.IGNORECASE)

    SUBCHAPTER_TITLE_REGEX = re.compile(r'^\s*\d+(?:\s*\.\s*\d+)+\.?(?:\s+.+)?$', re.IGNORECASE)

    CODE_TITLE_HEADER_REGEX = re.compile(
        r'^\s*(?:segmen\s*program|listing|algoritma|algorithm|kode\s*program|script)\b',
        re.IGNORECASE
    )

    CODE_TITLE_FLEX_REGEX = re.compile(
        r'^\s*(?:program|segmen\s*progr?am|listing|algoritma|algorithm|kode\s*program|script)\s*'
        r'\d+(?:\.\d+)*(?:\s*[:.)-])?(?:\s+.+)?$',
        re.IGNORECASE
    )

    CODE_LINE_NUMBER_REGEX = re.compile(r'^\s*\d{1,3}\s*[:.)]\s*')

    CODE_TEXT_HINT_REGEX = re.compile(
        r'\b(?:def|class|return|if|else|elif|for|while|import|from|public|private|protected|'
        r'static|void|int|float|double|string|bool|yield|await|select|insert|update|delete|'
        r'create|join|where)\b',
        re.IGNORECASE
    )

    LIST_NUMERIC_REGEX = re.compile(r'^\s*\d+(?!\s*\.\s*\d)(?:[.)])', re.IGNORECASE)

    LIST_ALPHA_REGEX = re.compile(r'^\s*[a-z](?:[.)])', re.IGNORECASE)

    LIST_TEXTUAL_BULLET_REGEX = re.compile(r'^\s*[oO0](?=\s+)')

    LIST_BULLET_REGEX = re.compile(
        r'^\s*(?:'
        r'[\u2022\u2023\u25e6\u2043\u2219\u00b7\u2024\u25aa\u25cf\u25cb\u25ef\u25c9\u25a0\u25a1\u25c6\u25c7\u2713\u2714\u2717\u2718\u2610\u2611\u2612\u2794\u27a4\*\-\u2013\u2014\.\+]'
        r'|[^\w\s](?=\s|$)'
        r')'
    )

    FIGURE_PANEL_MARKER_REGEX = re.compile(r'^\s*\([a-z]\)\s*$', re.IGNORECASE)

    def __init__(self):
        self.alignment_service = AlignmentService()
        self.docling_service = DoclingService()
        self.fusion_service = DoclingFusionService()
        self.visualization_service = VisualizationService(output_dir=VISUALIZATION_OUTPUT)

    @staticmethod
    def _is_env_enabled_default_true(env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _read_positive_int_env(env_name: str, default_value: int) -> int:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else default_value
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _read_float_env(env_name: str, default_value: float, min_value: float = None, max_value: float = None) -> float:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return default_value
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    @staticmethod
    def _canonical_ref_tipe(ref_tipe: str) -> str:
        if ref_tipe == 'buku':
            return 'bab'
        return ref_tipe
