"""
Canonical expression tree for algebraic deduplication of 24-game formulas.

Strategy:
1. Build a tree from Formula steps
2. Normalize: subtraction → sum(neg), division → prod(inv)
3. Flatten associative chains (nested sums/products)
4. Cancel matching term/Neg and factor/Inv pairs
5. Remove identity elements (0 in sums, 1 in products)
6. Sort commutative operand lists
7. Serialize to a canonical string for dedup
"""

from Operations.addition import Addition
from Operations.subtraction import Subtraction
from Operations.multiplication import Multiplication
from Operations.division import Division
from Operations.exp import Exp
from Operations.log import Log
from Operations.sqrt import Sqrt


# --- Node types ---

class Num:
    def __init__(self, value):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        self.value = value


class Neg:
    def __init__(self, child):
        self.child = child


class Inv:
    def __init__(self, child):
        self.child = child


class BinOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right


class Sum:
    def __init__(self, children):
        self.children = children


class Prod:
    def __init__(self, children):
        self.children = children


# --- Operation class → op string mapping ---

_OP_SYMBOL = {
    Addition: '+',
    Subtraction: '-',
    Multiplication: '*',
    Division: '/',
    Exp: '^',
    Log: 'log',
    Sqrt: 'sqrt',
}


# --- Tree building ---

def build_tree(formula, cards):
    """Build an expression tree from a Formula and the original card values."""
    nodes = [Num(v) for v in cards]
    for step in reversed(formula.steps):
        op = _OP_SYMBOL[step["operation"]]
        first = step["first_ind"]
        second = step["second_ind"]
        nodes[first] = BinOp(op, nodes[first], nodes[second])
        nodes.pop(second)
    return nodes[0]


# --- Serialization ---

def serialize(expr):
    """Return a canonical string representation of the expression."""
    if isinstance(expr, Num):
        return str(expr.value)
    if isinstance(expr, Neg):
        return 'neg(' + serialize(expr.child) + ')'
    if isinstance(expr, Inv):
        return 'inv(' + serialize(expr.child) + ')'
    if isinstance(expr, BinOp):
        return '(' + serialize(expr.left) + expr.op + serialize(expr.right) + ')'
    if isinstance(expr, Sum):
        return 'sum(' + ','.join(serialize(c) for c in expr.children) + ')'
    if isinstance(expr, Prod):
        return 'prod(' + ','.join(serialize(c) for c in expr.children) + ')'


# --- Normalization ---

def normalize(expr):
    """Normalize an expression tree for canonical comparison."""
    if isinstance(expr, Num):
        return expr

    if isinstance(expr, Neg):
        inner = normalize(expr.child)
        if isinstance(inner, Neg):
            return inner.child
        if isinstance(inner, Num) and inner.value == 0:
            return Num(0)
        if isinstance(inner, Sum):
            return normalize(Sum([Neg(c) for c in inner.children]))
        return Neg(inner)

    if isinstance(expr, Inv):
        inner = normalize(expr.child)
        if isinstance(inner, Inv):
            return inner.child
        if isinstance(inner, Num) and inner.value == 1:
            return Num(1)
        if isinstance(inner, Prod):
            return normalize(Prod([Inv(c) for c in inner.children]))
        return Inv(inner)

    if isinstance(expr, BinOp):
        return _normalize_binop(expr)

    if isinstance(expr, Sum):
        return _normalize_sum(expr)

    if isinstance(expr, Prod):
        return _normalize_prod(expr)


def _normalize_binop(expr):
    a = normalize(expr.left)
    b = normalize(expr.right)
    op = expr.op

    # --- Advanced cancellation for non-associative ops ---

    # log[x](x) → 1
    if op == 'log' and serialize(a) == serialize(b):
        return Num(1)

    # sqrt[n](x^n) → x
    if op == 'sqrt' and isinstance(a, BinOp) and a.op == '^':
        if serialize(b) == serialize(a.right):
            return a.left

    # log[b](b^x) → x
    if op == 'log' and isinstance(a, BinOp) and a.op == '^':
        if serialize(b) == serialize(a.left):
            return a.right

    # (sqrt[n](x))^n → x
    if op == '^' and isinstance(a, BinOp) and a.op == 'sqrt':
        if serialize(b) == serialize(a.right):
            return a.left

    # b^(log[b](x)) → x
    if op == '^' and isinstance(b, BinOp) and b.op == 'log':
        if serialize(a) == serialize(b.right):
            return b.left

    # --- Convert arithmetic ops to Sum/Prod ---

    if op == '+':
        return _normalize_sum(Sum([a, b]))
    if op == '-':
        return _normalize_sum(Sum([a, normalize(Neg(b))]))
    if op == '*':
        return _normalize_prod(Prod([a, b]))
    if op == '/':
        return _normalize_prod(Prod([a, normalize(Inv(b))]))

    # Non-arithmetic ops stay as BinOp
    return BinOp(op, a, b)


def _normalize_sum(expr):
    # Flatten nested sums and normalize children
    flat = []
    for child in expr.children:
        nc = normalize(child)
        if isinstance(nc, Sum):
            flat.extend(nc.children)
        else:
            flat.append(nc)

    # Cancel matching term / Neg(term) pairs
    flat = _cancel_neg_pairs(flat)

    # Remove additive identity (Num(0))
    flat = [c for c in flat if not (isinstance(c, Num) and c.value == 0)]

    if len(flat) == 0:
        return Num(0)
    if len(flat) == 1:
        return flat[0]

    flat.sort(key=serialize)
    return Sum(flat)


def _normalize_prod(expr):
    # Flatten nested products and normalize children
    flat = []
    for child in expr.children:
        nc = normalize(child)
        if isinstance(nc, Prod):
            flat.extend(nc.children)
        else:
            flat.append(nc)

    # Cancel matching factor / Inv(factor) pairs
    flat = _cancel_inv_pairs(flat)

    # Remove multiplicative identity (Num(1))
    flat = [c for c in flat if not (isinstance(c, Num) and c.value == 1)]

    if len(flat) == 0:
        return Num(1)
    if len(flat) == 1:
        return flat[0]

    flat.sort(key=serialize)
    return Prod(flat)


def _cancel_neg_pairs(children):
    """Cancel matching x / Neg(x) pairs by canonical serial."""
    positives = []
    negatives = []
    for c in children:
        if isinstance(c, Neg):
            negatives.append(c)
        else:
            positives.append(c)

    remaining_pos = list(positives)
    remaining_neg = list(negatives)

    for neg in list(negatives):
        inner_key = serialize(neg.child)
        for i, pos in enumerate(remaining_pos):
            if serialize(pos) == inner_key:
                remaining_pos.pop(i)
                remaining_neg.remove(neg)
                break

    return remaining_pos + remaining_neg


def _cancel_inv_pairs(children):
    """Cancel matching x / Inv(x) pairs by canonical serial."""
    factors = []
    inverses = []
    for c in children:
        if isinstance(c, Inv):
            inverses.append(c)
        else:
            factors.append(c)

    remaining_fac = list(factors)
    remaining_inv = list(inverses)

    for inv in list(inverses):
        inner_key = serialize(inv.child)
        for i, fac in enumerate(remaining_fac):
            if serialize(fac) == inner_key:
                remaining_fac.pop(i)
                remaining_inv.remove(inv)
                break

    return remaining_fac + remaining_inv


# --- Public API ---

def canonical_key(formula, cards):
    """Return a canonical string key for algebraic deduplication."""
    tree = build_tree(formula, cards)
    normalized = normalize(tree)
    return serialize(normalized)
