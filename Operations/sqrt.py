import math
from Operations.operation import Operation


class Sqrt(Operation):
    """N-th root: calc(x, y) = x^(1/y), i.e. the y-th root of x"""

    @staticmethod
    def calc(value1, value2):
        try:
            if value2 == 0 or value2 == 1:  # undefined or identity (x^1 = x)
                return float('nan')
            if value1 == 0 or value1 == 1:  # trivial: n-th root of 0 or 1 is always 0 or 1
                return float('nan')
            if value1 < 0:
                return float('nan')
            return math.pow(value1, 1.0 / value2)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float('nan')

    @classmethod
    @property
    def is_commutative(cls):
        return False

    @classmethod
    @property
    def priority(cls):
        return 2

    @staticmethod
    def to_string(value1, value2):
        return "sqrt[" + value2 + "](" + value1 + ")"
