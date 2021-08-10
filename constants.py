from enum import Enum

ENTITY_START_MARKER = "<<m>>"
ENTITY_END_MARKER = "<</m>>"
CLS = "[CLS]"
SEP = "[SEP]"
ENTITY_PAD_IDX = -1
COREF_PAD_IDX = -1
NOT_COMB = "NO_COMB"
RELATION_UNKNOWN = "RELATION-UNKNOWN"

class SpanType(Enum):
    TEXT = 1
    MARKERS = 2