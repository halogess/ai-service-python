import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


from .matching_metadata_block_mixin import AlignmentMatchingMetadataBlockMixin
from .matching_metadata_confidence_mixin import AlignmentMatchingMetadataConfidenceMixin


class AlignmentMatchingMetadataMixin(
    AlignmentMatchingMetadataBlockMixin,
    AlignmentMatchingMetadataConfidenceMixin,
):


    pass
