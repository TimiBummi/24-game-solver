import 'operation.dart';
import 'formula.dart';
import 'expr_tree.dart';
import 'addition.dart';
import 'subtraction.dart';
import 'multiplication.dart';
import 'division.dart';

class Game {
  static const List<Operation> defaultOperations = [
    Addition(),
    Subtraction(),
    Multiplication(),
    Division(),
  ];

  final List<Operation> operations;
  final List<int> viableResults;

  const Game({
    this.operations = defaultOperations,
    this.viableResults = const [24],
  });

  List<Formula> solve(List<int> cards) {
    final doubleCards = cards.map((c) => c.toDouble()).toList();
    final allFormulas = _solveRecursively(doubleCards);

    final filtered = viableResults.isEmpty
        ? allFormulas
        : allFormulas
            .where((f) =>
                viableResults.any((r) => (f.result - r).abs() < 0.0001))
            .toList();

    // Deduplicate by canonical expression tree (catches arithmetic equivalences)
    final seen = <String>{};
    final unique = <Formula>[];
    for (final formula in filtered) {
      final key = Expr.canonicalKey(formula, cards);
      if (seen.add(key)) unique.add(formula);
    }
    return unique;
  }

  List<Formula> _solveRecursively(List<double> values) {
    if (values.length == 1) {
      return [Formula(values[0])];
    }

    final formulas = <Formula>[];
    for (int i = 0; i < values.length - 1; i++) {
      for (int j = i + 1; j < values.length; j++) {
        // remaining = values with index j removed; result placed at index i
        final remaining = List<double>.of(values)..removeAt(j);

        for (final op in operations) {
          _tryOp(remaining, op, values[i], values[j], i, j, i, formulas);
          if (!op.isCommutative) {
            _tryOp(remaining, op, values[j], values[i], j, i, i, formulas);
          }
        }
      }
    }
    return formulas;
  }

  void _tryOp(
    List<double> remaining,
    Operation op,
    double a,
    double b,
    int firstInd,
    int secondInd,
    int placeAt,
    List<Formula> out,
  ) {
    final calculated = op.calc(a, b);
    if (calculated.isNaN || calculated.isInfinite) return;

    final next = List<double>.of(remaining)..[placeAt] = calculated;
    final subFormulas = _solveRecursively(next);
    for (final f in subFormulas) {
      out.add(f.withStep(op, firstInd, secondInd));
    }
  }
}
