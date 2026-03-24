import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

from .matching_char_mixin import AlignmentMatchingCharMixin
from .matching_execution_mixin import AlignmentMatchingExecutionMixin
from .matching_metadata_mixin import AlignmentMatchingMetadataMixin
from .matching_retry_mixin import AlignmentMatchingRetryMixin
from .matching_trace_mixin import AlignmentMatchingTraceMixin



class AlignmentMatchingMixin(
    AlignmentMatchingMetadataMixin,
    AlignmentMatchingRetryMixin,
    AlignmentMatchingExecutionMixin,
    AlignmentMatchingCharMixin,
    AlignmentMatchingTraceMixin,
):
    MARKER_ONLY_TEXT_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s*[:.)]?\s*$')
    PROGRAM_SEGMENT_HEADING_RE = re.compile(
        r'\bsegmen\s*program\s*\d+(?:\.\d+)*(?:\s*\(\s*lanjutan\s*\))?',
        re.IGNORECASE,
    )
    STRUCTURED_BLOCK_HEADING_RE = re.compile(
        r'\b(?P<kind>segmen\s*program|algoritma)\s*'
        r'(?P<number>\d+(?:\.\d+)*)'
        r'(?P<continuation>\s*\(\s*lanjutan\s*\))?',
        re.IGNORECASE,
    )
    CODE_LINE_NUMBER_RE = re.compile(r'^\s*\d{1,3}\s*[:.)]\s*')
    CODE_TEXT_HINT_RE = re.compile(
        r'\b(?:const|let|var|function|return|if|else|await|async|class|'
        r'import|from|public|private|static|void|final|map|jsondecode|http|get|post|'
        r'emit|console|socket|response|statuscode)\b',
        re.IGNORECASE,
    )

