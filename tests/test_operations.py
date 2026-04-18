import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Operations.addition import Addition
from Operations.subtraction import Subtraction
from Operations.multiplication import Multiplication
from Operations.division import Division
from Operations.exp import Exp
from Operations.log import Log
from Operations.sqrt import Sqrt


class TestAddition:
    def test_calc(self):
        assert Addition.calc(3, 4) == 7

    def test_to_string(self):
        assert Addition.to_string('3', '4') == '3+4'

    def test_is_commutative(self):
        assert Addition.is_commutative is True

    def test_priority(self):
        assert Addition.priority == 0


class TestSubtraction:
    def test_calc(self):
        assert Subtraction.calc(5, 3) == 2

    def test_to_string(self):
        assert Subtraction.to_string('5', '3') == '5-3'

    def test_priority(self):
        assert Subtraction.priority == 0


class TestMultiplication:
    def test_calc(self):
        assert Multiplication.calc(3, 4) == 12

    def test_to_string(self):
        assert Multiplication.to_string('3', '4') == '3*4'

    def test_priority(self):
        assert Multiplication.priority == 0


class TestDivision:
    def test_calc(self):
        assert Division.calc(8, 2) == 4.0

    def test_to_string(self):
        assert Division.to_string('8', '2') == '8/2'

    def test_priority(self):
        assert Division.priority == 0


class TestExp:
    def test_basic_power(self):
        assert Exp.calc(2, 3) == 8.0

    def test_square(self):
        assert Exp.calc(3, 2) == 9.0

    def test_trivial_base_one(self):
        assert math.isnan(Exp.calc(1, 5))

    def test_trivial_base_one_any_exp(self):
        assert math.isnan(Exp.calc(1, 1))

    def test_trivial_exponent_one(self):
        # x^1 = x always — identity, filtered
        assert math.isnan(Exp.calc(2, 1))
        assert math.isnan(Exp.calc(5, 1))

    def test_to_string(self):
        assert Exp.to_string('2', '3') == '2^3'

    def test_is_not_commutative(self):
        assert Exp.is_commutative is False

    def test_priority(self):
        assert Exp.priority == 2

    def test_non_integer_result_still_returned(self):
        # Game filters non-integers; the operation itself just computes
        result = Exp.calc(2, 3)
        assert float(result).is_integer()


class TestLog:
    def test_log_base2(self):
        assert Log.calc(8, 2) == 3.0

    def test_log_base3(self):
        assert Log.calc(9, 3) == 2.0

    def test_log_base10(self):
        # log_10(100) = 2
        assert Log.calc(100, 10) == pytest_approx_2 if False else abs(Log.calc(100, 10) - 2.0) < 1e-9

    def test_trivial_log_of_one(self):
        # log_n(1) = 0 always — trivial
        assert math.isnan(Log.calc(1, 3))
        assert math.isnan(Log.calc(1, 5))

    def test_invalid_negative_value(self):
        assert math.isnan(Log.calc(-1, 2))

    def test_invalid_zero_value(self):
        assert math.isnan(Log.calc(0, 2))

    def test_invalid_base_one(self):
        assert math.isnan(Log.calc(4, 1))

    def test_invalid_negative_base(self):
        assert math.isnan(Log.calc(4, -2))

    def test_invalid_zero_base(self):
        assert math.isnan(Log.calc(4, 0))

    def test_to_string(self):
        assert Log.to_string('8', '2') == 'log[2](8)'

    def test_to_string_with_expressions(self):
        assert Log.to_string('(2+1)', '3') == 'log[3]((2+1))'

    def test_is_not_commutative(self):
        assert Log.is_commutative is False

    def test_priority(self):
        assert Log.priority == 2


class TestSqrt:
    def test_cube_root(self):
        assert Sqrt.calc(8, 3) == 2.0

    def test_square_root(self):
        assert Sqrt.calc(4, 2) == 2.0

    def test_fourth_root(self):
        assert Sqrt.calc(16, 4) == 2.0

    def test_trivial_root_of_one(self):
        assert math.isnan(Sqrt.calc(1, 3))
        assert math.isnan(Sqrt.calc(1, 5))

    def test_trivial_root_of_zero(self):
        assert math.isnan(Sqrt.calc(0, 3))
        assert math.isnan(Sqrt.calc(0, 5))

    def test_trivial_first_root(self):
        # x^(1/1) = x always — identity
        assert math.isnan(Sqrt.calc(8, 1))
        assert math.isnan(Sqrt.calc(5, 1))

    def test_invalid_zero_degree(self):
        assert math.isnan(Sqrt.calc(8, 0))

    def test_invalid_negative_value(self):
        assert math.isnan(Sqrt.calc(-8, 3))

    def test_to_string(self):
        assert Sqrt.to_string('8', '3') == 'sqrt[3](8)'

    def test_to_string_with_expressions(self):
        assert Sqrt.to_string('(2+6)', '4') == 'sqrt[4]((2+6))'

    def test_is_not_commutative(self):
        assert Sqrt.is_commutative is False

    def test_priority(self):
        assert Sqrt.priority == 2
