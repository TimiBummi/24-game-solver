import 'formula.dart';

/// Canonical expression tree used to detect arithmetically equivalent formulas.
///
/// Strategy:
/// 1. Build a tree from [Formula] steps
/// 2. Normalize: subtraction→add(neg), division→mul(inv)
/// 3. Flatten associative chains (nested sums/products)
/// 4. Cancel matching term/Neg and factor/Inv pairs
/// 5. Remove identity elements (0 in sums, 1 in products)
/// 6. Sort commutative operand lists
/// 7. Serialize to a canonical string for dedup
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
        // neg(0) → 0
        if (inner is Num && inner.value == 0) return Num(0);
        // neg(sum(a,b,...)) → sum(neg(a), neg(b), ...)
        if (inner is Sum) {
          return _normalize(Sum(inner.children.map((c) => Neg(c)).toList()));
        }
        return Neg(inner);
      case Inv():
        final inner = _normalize(e.child);
        // inv(inv(x)) → x
        if (inner is Inv) return inner.child;
        // inv(1) → 1
        if (inner is Num && inner.value == 1) return Num(1);
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

    // Cancel matching term / Neg(term) pairs
    _cancelNegPairs(flat);

    // Remove additive identity (Num(0))
    flat.removeWhere((c) => c is Num && c.value == 0);

    if (flat.isEmpty) return Num(0);
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

    // Cancel matching factor / Inv(factor) pairs
    _cancelInvPairs(flat);

    // Remove multiplicative identity (Num(1))
    flat.removeWhere((c) => c is Num && c.value == 1);

    if (flat.isEmpty) return Num(1);
    if (flat.length == 1) return flat.first;

    // Sort for canonical ordering
    flat.sort((a, b) => _serialize(a).compareTo(_serialize(b)));
    return Prod(flat);
  }

  // ── Pair cancellation ──

  /// Cancel matching x / Neg(x) pairs in-place by canonical serial.
  static void _cancelNegPairs(List<Expr> children) {
    final negIndices = <int>[];
    for (int i = 0; i < children.length; i++) {
      if (children[i] is Neg) negIndices.add(i);
    }
    final toRemove = <int>{};
    for (final ni in negIndices) {
      if (toRemove.contains(ni)) continue;
      final innerKey = _serialize((children[ni] as Neg).child);
      for (int i = 0; i < children.length; i++) {
        if (i == ni || toRemove.contains(i)) continue;
        if (children[i] is! Neg && _serialize(children[i]) == innerKey) {
          toRemove.addAll([ni, i]);
          break;
        }
      }
    }
    final sorted = toRemove.toList()..sort((a, b) => b.compareTo(a));
    for (final i in sorted) {
      children.removeAt(i);
    }
  }

  /// Cancel matching x / Inv(x) pairs in-place by canonical serial.
  static void _cancelInvPairs(List<Expr> children) {
    final invIndices = <int>[];
    for (int i = 0; i < children.length; i++) {
      if (children[i] is Inv) invIndices.add(i);
    }
    final toRemove = <int>{};
    for (final ii in invIndices) {
      if (toRemove.contains(ii)) continue;
      final innerKey = _serialize((children[ii] as Inv).child);
      for (int i = 0; i < children.length; i++) {
        if (i == ii || toRemove.contains(i)) continue;
        if (children[i] is! Inv && _serialize(children[i]) == innerKey) {
          toRemove.addAll([ii, i]);
          break;
        }
      }
    }
    final sorted = toRemove.toList()..sort((a, b) => b.compareTo(a));
    for (final i in sorted) {
      children.removeAt(i);
    }
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
