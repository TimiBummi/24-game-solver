import math
from Operations.operation import Operation


class Log(Operation):
    """Logarithm: calc(x, y) = log base y of x"""

    @staticmethod
    def calc(value1, value2):
        try:
            if value1 <= 0 or value2 <= 0 or value2 == 1:
                return float('nan')
            if value1 == 1:  # log_n(1) = 0 always — trivial
                return float('nan')
            return math.log(value1, value2)
        except (ValueError, ZeroDivisionError):
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
        return "log[" + value2 + "](" + value1 + ")"
