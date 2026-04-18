import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from expr_tree import (
    Num, Neg, Inv, BinOp, Sum, Prod,
    build_tree, normalize, serialize, canonical_key,
)
from formula import Formula
from Operations.addition import Addition
from Operations.subtraction import Subtraction
from Operations.multiplication import Multiplication
from Operations.division import Division
from Operations.exp import Exp
from Operations.log import Log
from Operations.sqrt import Sqrt


# --- Helper to build a Formula from step tuples ---

def _make_formula(result, steps, cards):
    """Create a Formula with given steps. Each step is (OpClass, first_ind, second_ind)."""
    f = Formula(result)
    for op, first, second in steps:
        f.add_step(op, first, second)
    return f


class TestBuildTree:
    def test_single_card(self):
        f = Formula(5)
        tree = build_tree(f, [5])
        assert isinstance(tree, Num)
        assert tree.value == 5

    def test_simple_addition(self):
        # 3+5 = 8: step (+, 0, 1)
        f = _make_formula(8, [(Addition, 0, 1)], [3, 5])
        tree = build_tree(f, [3, 5])
        assert isinstance(tree, BinOp)
        assert tree.op == '+'
        assert tree.left.value == 3
        assert tree.right.value == 5

    def test_mul_then_add(self):
        # (4*6)+5 = 29: steps are [outer, inner] = [(+, 0, 1), (*, 0, 1)]
        # Cards: [4, 6, 5] → after *: [24, 5] → after +: [29]
        # But actually with 3 cards: pair (0,1) mul → [24, 5], pair (0,1) add → [29]
        # Steps stored: outermost first. add_step called during unwind:
        # first +(0,1) is added, then *(0,1) is added.
        # So steps = [(+, 0, 1), (*, 0, 1)]
        # Reversed for build_tree: first (*, 0, 1), then (+, 0, 1)
        f = _make_formula(29, [(Addition, 0, 1), (Multiplication, 0, 1)], [4, 6, 5])
        tree = build_tree(f, [4, 6, 5])
        # Should be Add(Mul(4, 6), 5)
        assert isinstance(tree, BinOp) and tree.op == '+'
        assert isinstance(tree.left, BinOp) and tree.left.op == '*'
        assert tree.left.left.value == 4
        assert tree.left.right.value == 6
        assert tree.right.value == 5


class TestSerialize:
    def test_num(self):
        assert serialize(Num(42)) == '42'

    def test_neg(self):
        assert serialize(Neg(Num(5))) == 'neg(5)'

    def test_inv(self):
        assert serialize(Inv(Num(3))) == 'inv(3)'

    def test_binop(self):
        assert serialize(BinOp('^', Num(2), Num(3))) == '(2^3)'

    def test_sum(self):
        assert serialize(Sum([Num(1), Num(2)])) == 'sum(1,2)'

    def test_prod(self):
        assert serialize(Prod([Num(3), Num(4)])) == 'prod(3,4)'


class TestNormalizeIdentityRemoval:
    def test_sum_removes_zero(self):
        # x + 0 → x
        result = normalize(Sum([Num(5), Num(0)]))
        assert isinstance(result, Num) and result.value == 5

    def test_prod_removes_one(self):
        # x * 1 → x
        result = normalize(Prod([Num(7), Num(1)]))
        assert isinstance(result, Num) and result.value == 7

    def test_neg_zero_is_zero(self):
        result = normalize(Neg(Num(0)))
        assert isinstance(result, Num) and result.value == 0

    def test_inv_one_is_one(self):
        result = normalize(Inv(Num(1)))
        assert isinstance(result, Num) and result.value == 1


class TestNormalizeNegInvSimplification:
    def test_double_neg(self):
        result = normalize(Neg(Neg(Num(5))))
        assert isinstance(result, Num) and result.value == 5

    def test_double_inv(self):
        result = normalize(Inv(Inv(Num(3))))
        assert isinstance(result, Num) and result.value == 3

    def test_neg_distributes_over_sum(self):
        # -(a + b) → sum(neg(a), neg(b))
        result = normalize(Neg(Sum([Num(3), Num(5)])))
        assert isinstance(result, Sum)
        serials = [serialize(c) for c in result.children]
        assert 'neg(3)' in serials
        assert 'neg(5)' in serials

    def test_inv_distributes_over_prod(self):
        # 1/(a * b) → prod(inv(a), inv(b))
        result = normalize(Inv(Prod([Num(3), Num(5)])))
        assert isinstance(result, Prod)
        serials = [serialize(c) for c in result.children]
        assert 'inv(3)' in serials
        assert 'inv(5)' in serials


class TestNormalizePairCancellation:
    def test_sum_cancels_neg_pair(self):
        # 5 + neg(5) → 0
        result = normalize(Sum([Num(5), Neg(Num(5))]))
        assert isinstance(result, Num) and result.value == 0

    def test_prod_cancels_inv_pair(self):
        # 5 * inv(5) → 1
        result = normalize(Prod([Num(5), Inv(Num(5))]))
        assert isinstance(result, Num) and result.value == 1

    def test_subtraction_cancels(self):
        # 5 - 5 = 0: BinOp('-', 5, 5) → Sum([5, Neg(5)]) → cancel → 0
        result = normalize(BinOp('-', Num(5), Num(5)))
        assert isinstance(result, Num) and result.value == 0

    def test_division_cancels(self):
        # 13 / 13 = 1: BinOp('/', 13, 13) → Prod([13, Inv(13)]) → cancel → 1
        result = normalize(BinOp('/', Num(13), Num(13)))
        assert isinstance(result, Num) and result.value == 1

    def test_add_then_subtract_cancels(self):
        # (4+5)-5 → sum(4, 5, neg(5)) → cancel 5 → 4
        expr = BinOp('-', BinOp('+', Num(4), Num(5)), Num(5))
        result = normalize(expr)
        assert isinstance(result, Num) and result.value == 4

    def test_mul_then_divide_cancels(self):
        # (4*5)/5 → prod(4, 5, inv(5)) → cancel 5 → 4
        expr = BinOp('/', BinOp('*', Num(4), Num(5)), Num(5))
        result = normalize(expr)
        assert isinstance(result, Num) and result.value == 4

    def test_nested_mul_chain_cancels(self):
        # ((4*5)*6)/5 → prod(4,5,6,inv(5)) → cancel 5 → prod(4,6)
        expr = BinOp('/', BinOp('*', BinOp('*', Num(4), Num(5)), Num(6)), Num(5))
        result = normalize(expr)
        assert isinstance(result, Prod)
        s = serialize(result)
        assert s == 'prod(4,6)'


class TestNormalizeAdvancedOps:
    def test_log_same_operands(self):
        # log[5](5) = 1
        result = normalize(BinOp('log', Num(5), Num(5)))
        assert isinstance(result, Num) and result.value == 1

    def test_sqrt_cancels_exp(self):
        # sqrt[5](4^5) = 4
        result = normalize(BinOp('sqrt', BinOp('^', Num(4), Num(5)), Num(5)))
        assert isinstance(result, Num) and result.value == 4

    def test_log_cancels_exp(self):
        # log[5](5^4) = 4
        result = normalize(BinOp('log', BinOp('^', Num(5), Num(4)), Num(5)))
        assert isinstance(result, Num) and result.value == 4

    def test_exp_cancels_sqrt(self):
        # (sqrt[3](x))^3 = x
        result = normalize(BinOp('^', BinOp('sqrt', Num(8), Num(3)), Num(3)))
        assert isinstance(result, Num) and result.value == 8

    def test_exp_cancels_log(self):
        # 5^(log[5](x)) = x
        result = normalize(BinOp('^', Num(5), BinOp('log', Num(7), Num(5))))
        assert isinstance(result, Num) and result.value == 7


class TestNormalizeCommutativity:
    def test_sum_sorted(self):
        # 5 + 3 → sum(3, 5) (sorted)
        r1 = serialize(normalize(BinOp('+', Num(5), Num(3))))
        r2 = serialize(normalize(BinOp('+', Num(3), Num(5))))
        assert r1 == r2

    def test_prod_sorted(self):
        # 6 * 4 → prod(4, 6) (sorted)
        r1 = serialize(normalize(BinOp('*', Num(6), Num(4))))
        r2 = serialize(normalize(BinOp('*', Num(4), Num(6))))
        assert r1 == r2

    def test_associative_rearrangement(self):
        # (2-3)+4, (2+4)-3, and 2-(3-4) all normalize the same
        e1 = BinOp('+', BinOp('-', Num(2), Num(3)), Num(4))
        e2 = BinOp('-', BinOp('+', Num(2), Num(4)), Num(3))
        e3 = BinOp('-', Num(2), BinOp('-', Num(3), Num(4)))
        s1 = serialize(normalize(e1))
        s2 = serialize(normalize(e2))
        s3 = serialize(normalize(e3))
        assert s1 == s2 == s3


class TestMeaningfulSubtractionPreserved:
    def test_sub_of_one_not_simplified(self):
        # 5*5 - (13/13) = 25 - 1: the 1 is subtracted, NOT used as multiplicative identity
        # Canonical should include the subtraction of 1
        expr = BinOp('-', BinOp('*', Num(5), Num(5)), BinOp('/', Num(13), Num(13)))
        result = normalize(expr)
        s = serialize(result)
        # The 13/13 cancels to 1 in the Prod, but the subtraction remains
        assert 'neg(1)' in s, f"Expected neg(1) in canonical form, got: {s}"
        assert 'prod(5,5)' in s, f"Expected prod(5,5) in canonical form, got: {s}"


class TestCanonicalKeyEndToEnd:
    def test_all_4655_formulas_same_key(self):
        """All 31 original formulas for [4,6,5,5] should produce the same canonical key."""
        from game import Game
        EXTENDED_OPS = [Addition, Subtraction, Division, Multiplication, Exp, Log, Sqrt]
        game = Game(EXTENDED_OPS, [])  # no result filter — we filter manually

        cards = [4, 6, 5, 5]
        raw = game._Game__play_recursively(cards)
        raw = [f for f in raw if f.get_result() == 24]

        # String dedup
        seen = set()
        unique = []
        for f in raw:
            s = f.to_string(cards)
            if s not in seen:
                seen.add(s)
                unique.append(f)

        assert len(unique) == 31, f"Expected 31 string-unique formulas, got {len(unique)}"

        keys = {canonical_key(f, cards) for f in unique}
        assert len(keys) == 1, f"Expected 1 canonical key, got {len(keys)}: {keys}"

    def test_distinct_solutions_have_different_keys(self):
        """3*(8+1)-3 and 3*(8-1)+3 for [3,3,8,1] should have different keys."""
        from game import Game
        BASE_OPS = [Addition, Subtraction, Division, Multiplication]
        game = Game(BASE_OPS, [24])
        formulas = game.play([3, 3, 8, 1])
        assert len(formulas) == 2
        keys = [canonical_key(f, [3, 3, 8, 1]) for f in formulas]
        assert keys[0] != keys[1]
