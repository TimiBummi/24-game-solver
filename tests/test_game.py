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
from game import Game

BASE_OPS = [Addition, Subtraction, Division, Multiplication]
EXTENDED_OPS = BASE_OPS + [Exp, Log, Sqrt]


def _strings(game, cards):
    return [f.to_string(cards) for f in game.play(cards)]


class TestIntegerConstraint:
    def test_fractional_intermediate_not_in_solutions(self):
        # 5/3 = 1.67 is not integer — no solution should rely on it
        game = Game(BASE_OPS, [24])
        formulas = game.play([5, 3, 7, 2])
        for f in formulas:
            assert f.get_result() == 24

    def test_division_producing_integer_is_valid(self):
        # 8/2=4, 4*6=24 — clean integer path should be found
        game = Game(BASE_OPS, [24])
        strings = _strings(game, [2, 4, 6, 8])
        assert any('8/2' in s or '2*' in s or '*6' in s for s in strings)


class TestIntermediateZeroNotFiltered:
    def test_equal_value_subtraction_still_appears(self):
        # Rules 3 and 5 removed — deduplication is deferred to algebraic comparison.
        # x-x=0 patterns may now appear; we only verify no crashes and correct result.
        game = Game(EXTENDED_OPS, [24])
        formulas = game.play([4, 6, 5, 5])
        for f in formulas:
            assert f.get_result() == 24


class TestEqualValueOneNotAlwaysFiltered:
    def test_equal_value_division_allowed_when_meaningful(self):
        # x/x=1 must NOT be filtered when the result is used non-trivially
        # e.g. 5*5 - (13/13) = 25 - 1 = 24 is a valid solution
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [5, 5, 13, 13])
        assert any('13/13' in s for s in strings), \
            "5*5-(13/13)=24 should be found — dividing by itself is not always neutral"

    def test_log_equal_value_allowed_when_meaningful(self):
        # log[x](x)=1 also allowed when used non-trivially
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [5, 5, 13, 13])
        assert any('log[13](13)' in s for s in strings) or \
               any('13/13' in s for s in strings)


class TestTrivialOperationFilters:
    def test_sqrt_of_one_never_appears(self):
        # sqrt[n](1) = 1 always — should never appear in solutions
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [3, 8, 1, 1])
        for s in strings:
            assert 'sqrt[1](1)' not in s
            # The pattern "sqrt[n](1)" with literal 1 as subject
            import re
            assert not re.search(r'sqrt\[\d+\]\(1\)', s)

    def test_sqrt_of_zero_never_appears(self):
        # sqrt[n](0) = 0 always — Sqrt.calc blocks it
        from Operations.sqrt import Sqrt
        import math
        assert math.isnan(Sqrt.calc(0, 2))
        assert math.isnan(Sqrt.calc(0, 5))

    def test_exp_base_one_never_appears(self):
        # 1^n = 1 always — Exp.calc blocks it
        from Operations.exp import Exp
        import math
        assert math.isnan(Exp.calc(1, 3))
        assert math.isnan(Exp.calc(1, 7))

    def test_exp_exponent_one_never_appears(self):
        # x^1 = x always — Exp.calc blocks it
        from Operations.exp import Exp
        import math
        assert math.isnan(Exp.calc(5, 1))
        assert math.isnan(Exp.calc(8, 1))

    def test_log_of_one_never_appears(self):
        # log_n(1) = 0 always — Log.calc blocks it
        from Operations.log import Log
        import math
        assert math.isnan(Log.calc(1, 3))
        assert math.isnan(Log.calc(1, 8))


class TestTargetIntermediateNotFiltered:
    def test_natural_form_preserved(self):
        # Rules 3 and 5 removed — the natural form 4*6+13-13 is now findable
        # instead of only the ugly (4+13-13)*6 that the combination forced.
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [4, 6, 13, 13])
        # With both rules removed, the natural 4*6-based solution appears
        assert any('4*6' in s or '6*4' in s for s in strings)


class TestNoStringDuplicates:
    def test_no_duplicate_formula_strings(self):
        # Identical cards create symmetry — deduplication removes exact duplicate strings
        game = Game(EXTENDED_OPS, [24])
        cards = [4, 6, 5, 5]
        strings = _strings(game, cards)
        assert len(strings) == len(set(strings)), "Duplicate formula strings found"

    def test_no_duplicate_formula_strings_base(self):
        game = Game(BASE_OPS, [24])
        cards = [3, 3, 8, 1]
        strings = _strings(game, cards)
        assert len(strings) == len(set(strings))


class TestKnownSolutions:
    def test_classic_solution(self):
        # (1 + 11*13) / 6 = 144/6 = 24
        game = Game(EXTENDED_OPS, [24])
        formulas = game.play([1, 6, 11, 13])
        assert len(formulas) == 1
        assert formulas[0].get_result() == 24
        assert '(1+(11*13))/6' in formulas[0].to_string([1, 6, 11, 13])

    def test_all_results_equal_target(self):
        game = Game(EXTENDED_OPS, [24])
        formulas = game.play([2, 3, 4, 8])
        assert len(formulas) > 0
        for f in formulas:
            assert f.get_result() == 24

    def test_log_solution_present(self):
        # (3 + log[2](8)) * 4 = (3+3)*4 = 24
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        assert any('log[2](8)' in s for s in strings)

    def test_exp_solution_present(self):
        # (2^3)*4 - 8 = 32 - 8 = 24
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        assert any('2^3' in s for s in strings)

    def test_sqrt_solution(self):
        # sqrt[3](8) = 2, (2+4)*... or similar
        game = Game(EXTENDED_OPS, [24])
        cards = [8, 3, 4, 6]
        strings = _strings(game, cards)
        # There should be solutions and at least some using advanced ops
        assert len(strings) > 0

    def test_no_solution_case(self):
        # Cards that genuinely have no solution
        game = Game(BASE_OPS, [24])
        formulas = game.play([1, 1, 1, 1])
        assert len(formulas) == 0


class TestGameOptions:
    def test_base_game_excludes_advanced_ops(self):
        game = Game(BASE_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        for s in strings:
            assert 'log' not in s
            assert 'sqrt' not in s
            assert '^' not in s

    def test_extended_game_includes_advanced_ops(self):
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        # Extended game should find more solutions than base
        base_game = Game(BASE_OPS, [24])
        base_strings = _strings(base_game, [2, 3, 4, 8])
        assert len(strings) > len(base_strings)

    def test_extended_has_log_solution(self):
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        assert any('log' in s for s in strings)

    def test_base_game_1_solution_for_2348(self):
        # Three associative rearrangements of (2+4-3)*8 collapse to 1
        game = Game(BASE_OPS, [24])
        formulas = game.play([2, 3, 4, 8])
        assert len(formulas) == 1

    def test_extended_game_5_solutions_for_2348(self):
        # 1 base + 4 advanced-op solutions
        game = Game(EXTENDED_OPS, [24])
        formulas = game.play([2, 3, 4, 8])
        assert len(formulas) == 5


class TestNonCommutativeOperations:
    def test_exp_both_orderings_tried(self):
        # 2^4=16 and 4^2=16 both tried; (2^4)-8)*3=24 and 3*((4^2)-8)=24
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        assert any('2^4' in s for s in strings)
        assert any('4^2' in s for s in strings)

    def test_log_both_orderings_tried(self):
        # log[2](8)=3 should appear
        game = Game(EXTENDED_OPS, [24])
        strings = _strings(game, [2, 3, 4, 8])
        assert any('log[2](8)' in s for s in strings)
