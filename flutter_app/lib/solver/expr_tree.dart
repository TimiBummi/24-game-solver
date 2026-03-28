import 'formula.dart';

/// Canonical expression tree used to detect arithmetically equivalent formulas.
///
/// Strategy:
/// 1. Build a tree from [Formula] steps
/// 2. Normalize: subtraction→add(neg), division→mul(inv)
/// 3. Flatten associative chains (nested sums/products)
/// 4. Sort commutative operand lists
/// 5. Serialize to a canonical string for dedup
sealed class Expr {
  /// Build an expression tree from a [Formula] and the original card values.
  static Expr fromFormula(Formula formula, List<int> originalValues) {
    final nodes = originalValues.map((v) => Num(v.toDouble()) as Expr).toList();
    // Steps are stored in reverse order (outermost operation first)
    final reversed = formula.steps.reversed.toList();
    for (final step in reversed) {
      final a = nodes[step.firstIndex];
      final b = nodes[step.secondIndex];
      nodes[step.firstIndex] = BinOp(step.operation.symbol, a, b);
      nodes.removeAt(step.secondIndex);
    }
    return nodes.first;
  }

  /// Return a canonical string representation for deduplication.
  static String canonicalKey(Formula formula, List<int> originalValues) {
    final tree = fromFormula(formula, originalValues);
    final normalized = _normalize(tree);
    return _serialize(normalized);
  }

  // ── Normalization ──

  static Expr _normalize(Expr e) {
    switch (e) {
      case Num():
        return e;
      case Neg():
        final inner = _normalize(e.child);
        // neg(neg(x)) → x
        if (inner is Neg) return inner.child;
        // neg(num) → Num(-n)
        if (inner is Num) return Num(-inner.value);
        // neg(sum(a,b,...)) → sum(neg(a), neg(b), ...)
        if (inner is Sum) {
          return _normalize(Sum(inner.children.map((c) => Neg(c)).toList()));
        }
        return Neg(inner);
      case Inv():
        final inner = _normalize(e.child);
        // inv(inv(x)) → x
        if (inner is Inv) return inner.child;
        // inv(prod(a,b,...)) → prod(inv(a), inv(b), ...)
        if (inner is Prod) {
          return _normalizeProd(
              Prod(inner.children.map((c) => Inv(c)).toList()));
        }
        return Inv(inner);
      case BinOp():
        return _normalizeBinOp(e);
      case Sum():
        return _normalizeSum(e);
      case Prod():
        return _normalizeProd(e);
    }
  }

  static Expr _normalizeBinOp(BinOp e) {
    final a = _normalize(e.left);
    final b = _normalize(e.right);
    switch (e.op) {
      case '+':
        return _normalizeSum(Sum([a, b]));
      case '-':
        // a - b → sum(a, neg(b))
        return _normalizeSum(Sum([a, _normalize(Neg(b))]));
      case '*':
        return _normalizeProd(Prod([a, b]));
      case '/':
        // a / b → prod(a, inv(b))
        return _normalizeProd(Prod([a, _normalize(Inv(b))]));
      default:
        return BinOp(e.op, a, b);
    }
  }

  static Expr _normalizeSum(Sum e) {
    // Flatten nested sums and normalize children
    final flat = <Expr>[];
    for (final child in e.children) {
      final nc = _normalize(child);
      if (nc is Sum) {
        flat.addAll(nc.children);
      } else {
        flat.add(nc);
      }
    }

    if (flat.length == 1) return flat.first;

    // Sort for canonical ordering
    flat.sort((a, b) => _serialize(a).compareTo(_serialize(b)));
    return Sum(flat);
  }

  static Expr _normalizeProd(Prod e) {
    // Flatten nested products and normalize children
    final flat = <Expr>[];
    void addFlat(Expr nc) {
      if (nc is Prod) {
        for (final c in nc.children) {
          addFlat(c);
        }
      } else {
        flat.add(nc);
      }
    }
    for (final child in e.children) {
      addFlat(_normalize(child));
    }

    // Distribute products over sums: if any child is a Sum, expand
    // Only do one level of distribution to avoid exponential blowup
    final sumIndex = flat.indexWhere((c) => c is Sum);
    if (sumIndex != -1) {
      final sum = flat[sumIndex] as Sum;
      final rest = List<Expr>.of(flat)..removeAt(sumIndex);
      final expanded = sum.children.map((term) {
        final factors = [...rest, term];
        return _normalize(Prod(factors));
      }).toList();
      return _normalizeSum(Sum(expanded));
    }

    if (flat.length == 1) return flat.first;

    // Sort for canonical ordering
    flat.sort((a, b) => _serialize(a).compareTo(_serialize(b)));
    return Prod(flat);
  }

  // ── Serialization ──

  static String _serialize(Expr e) {
    switch (e) {
      case Num():
        // Use integer representation when possible
        if (e.value == e.value.truncateToDouble()) {
          return e.value.toInt().toString();
        }
        return e.value.toStringAsFixed(6);
      case Neg():
        return 'neg(${_serialize(e.child)})';
      case Inv():
        return 'inv(${_serialize(e.child)})';
      case BinOp():
        return '(${_serialize(e.left)}${e.op}${_serialize(e.right)})';
      case Sum():
        final parts = e.children.map(_serialize).join(',');
        return 'sum($parts)';
      case Prod():
        final parts = e.children.map(_serialize).join(',');
        return 'prod($parts)';
    }
  }
}

class Num extends Expr {
  final double value;
  Num(this.value);
}

class Neg extends Expr {
  final Expr child;
  Neg(this.child);
}

class Inv extends Expr {
  final Expr child;
  Inv(this.child);
}

class BinOp extends Expr {
  final String op;
  final Expr left;
  final Expr right;
  BinOp(this.op, this.left, this.right);
}

class Sum extends Expr {
  final List<Expr> children;
  Sum(this.children);
}

class Prod extends Expr {
  final List<Expr> children;
  Prod(this.children);
}
