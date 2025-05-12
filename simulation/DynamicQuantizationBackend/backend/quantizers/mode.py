from enum import Enum

from backend.quantizers.dynamic import DynamicQuantizer
from backend.quantizers.estimate import EstimateQuantizer
from backend.quantizers.static import StaticQuantizer

class QuantMode(Enum):
    FP = (0, None)

    ESTIMATE = (1, EstimateQuantizer)

    DYNAMIC = (2, DynamicQuantizer)
    
    STATIC = (3, StaticQuantizer)

    def __new__(cls, value, quantizer):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.quantizer = quantizer
        return obj