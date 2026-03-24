from copy import deepcopy
import os
import re


from .postprocess_paragraph_context_mixin import AlignmentPostprocessParagraphContextMixin
from .postprocess_paragraph_rescue_mixin import AlignmentPostprocessParagraphRescueMixin


class AlignmentPostprocessParagraphMixin(
    AlignmentPostprocessParagraphContextMixin,
    AlignmentPostprocessParagraphRescueMixin,
):


    pass
