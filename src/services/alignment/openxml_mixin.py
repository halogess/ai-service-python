import json
import logging
import os
import re

from models import DokumenElemen, DokumenSection, DokumenPart, DokumenFormatText, DokumenFormatParagraf

logger = logging.getLogger(__name__)


from .openxml_sections_mixin import AlignmentOpenXmlSectionsMixin
from .openxml_style_mixin import AlignmentOpenXmlStyleMixin
from .openxml_units_mixin import AlignmentOpenXmlUnitsMixin


class AlignmentOpenXmlMixin(
    AlignmentOpenXmlSectionsMixin,
    AlignmentOpenXmlStyleMixin,
    AlignmentOpenXmlUnitsMixin,
):


    IMAGE_PLACEHOLDER_ONLY_RE = re.compile(
        r'^\s*(?:\[img(?::\d+)?\]\s*)+$',
        re.IGNORECASE
    )

    CHART_CAPTION_TEXT_RE = re.compile(
        r'^\s*(?:gambar|figure|fig\.?|grafik|graph|chart|tabel|table)\s*\d',
        re.IGNORECASE
    )

    TOC_BAB_STUB_RE = re.compile(r'^bab\s*(\d{1,2})$', re.IGNORECASE)

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

    FONT_KEY_MARKERS = (
        'font',
        'rfonts',
        'ascii',
        'hansi',
        'eastasia',
        'typeface',
    )
