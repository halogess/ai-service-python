import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


from .matching_char_core_mixin import AlignmentMatchingCharCoreMixin
from .matching_char_remainder_mixin import AlignmentMatchingCharRemainderMixin


class AlignmentMatchingCharMixin(
    AlignmentMatchingCharCoreMixin,
    AlignmentMatchingCharRemainderMixin,
):


    pass
