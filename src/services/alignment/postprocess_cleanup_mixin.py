from copy import deepcopy
import os
import re


from .postprocess_cleanup_lines_mixin import AlignmentPostprocessCleanupLinesMixin
from .postprocess_cleanup_absorb_mixin import AlignmentPostprocessCleanupAbsorbMixin


class AlignmentPostprocessCleanupMixin(
    AlignmentPostprocessCleanupLinesMixin,
    AlignmentPostprocessCleanupAbsorbMixin,
):


    pass
