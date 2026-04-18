import math
from Operations.operation import Operation


class Exp(Operation):
    """Exponentiation: calc(x, y) = x^y"""

    @staticmethod
    def calc(value1, value2):
        try:
            if value1 == 1:  # 1^n = 1 always — trivial
                return float('nan')
            if value2 == 1:  # x^1 = x always — identity
                return float('nan')
            return math.pow(value1, value2)
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
        return value1 + "^" + value2
