import 'package:flutter_test/flutter_test.dart';
import 'package:twenty_four_solver/solver/game.dart';
import 'package:twenty_four_solver/solver/expr_tree.dart';
import 'package:twenty_four_solver/solver/formula.dart';
import 'package:twenty_four_solver/solver/operation.dart';
import 'package:twenty_four_solver/solver/addition.dart';
import 'package:twenty_four_solver/solver/subtraction.dart';
import 'package:twenty_four_solver/solver/multiplication.dart';
import 'package:twenty_four_solver/solver/division.dart';

void main() {
  const game = Game();

  group('Game solver', () {
    test('finds solutions for [1, 6, 11, 13]', () {
      final results = game.solve([1, 6, 11, 13]);
      final strings = results.map((f) => f.toDisplayString([1, 6, 11, 13])).toList();
      expect(results, isNotEmpty);
      // At least one valid solution should exist
      expect(strings.any((s) => s.endsWith('= 24')), isTrue);
    });

    test('finds solutions for [6, 6, 6, 6]', () {
      final results = game.solve([6, 6, 6, 6]);
      final strings = results.map((f) => f.toDisplayString([6, 6, 6, 6])).toList();
      expect(results, isNotEmpty);
      // The solver wraps intermediate steps in brackets
      expect(strings.any((s) => s.endsWith('= 24')), isTrue);
    });

    test('finds solutions for [1, 2, 3, 4]', () {
      final results = game.solve([1, 2, 3, 4]);
      expect(results, isNotEmpty);
      expect(
        results.map((f) => f.toDisplayString([1, 2, 3, 4])).every((s) => s.endsWith('= 24')),
        isTrue,
      );
    });

    test('returns empty for [1, 1, 1, 1]', () {
      final results = game.solve([1, 1, 1, 1]);
      expect(results, isEmpty);
    });

    test('deduplicates results', () {
      final results = game.solve([6, 6, 6, 6]);
      final strings = results.map((f) => f.toDisplayString([6, 6, 6, 6])).toList();
      // No duplicates
      expect(strings.length, equals(strings.toSet().length));
    });

    test('handles division without crashing', () {
      // Should not throw on cards that could produce division by zero
      final results = game.solve([1, 1, 1, 1]);
      expect(results, isEmpty);
    });

    test('finds solutions for [13, 13, 7, 1] including division', () {
      final results = game.solve([13, 13, 7, 1]);
      final strings = results.map((f) => f.toDisplayString([13, 13, 7, 1])).toList();
      expect(results, isNotEmpty);
      // Should find ((13*13)-1)/7 = 24
      expect(strings.any((s) => s.contains('13*13') && s.contains('/7')), isTrue,
          reason: 'Expected a solution with 13*13 and /7, got: $strings');
    });

    test('no commutative duplicates', () {
      final results = game.solve([1, 2, 3, 4]);
      final strings = results.map((f) => f.toDisplayString([1, 2, 3, 4])).toList();
      // a+b and b+a should not both appear
      expect(strings.length, equals(strings.toSet().length));
    });

    test('all results equal 24', () {
      final testCases = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [1, 5, 5, 5],
        [8, 3, 8, 3],
      ];
      for (final cards in testCases) {
        final results = game.solve(cards);
        for (final formula in results) {
          expect(
            (formula.result - 24).abs() < 0.0001,
            isTrue,
            reason: 'Formula result ${formula.result} != 24 for cards $cards',
          );
        }
      }
    });
  });

  group('Arithmetic equivalence deduplication', () {
    // Helper: build a Formula manually from a sequence of operations
    // by replaying the solver's step-recording logic.
    Formula buildFormula(List<int> cards, List<(String op, int i, int j)> steps) {
      final ops = <String, Operation>{
        '+': const Addition(),
        '-': const Subtraction(),
        '*': const Multiplication(),
        '/': const Division(),
      };
      final values = cards.map((c) => c.toDouble()).toList();
      Formula f = Formula(0); // placeholder
      final stepList = <FormulaStep>[];
      for (final (opSym, i, j) in steps) {
        final op = ops[opSym]!;
        values[i] = op.calc(values[i], values[j]);
        values.removeAt(j);
        stepList.add(FormulaStep(op, i, j));
      }
      // Build formula with correct result and reversed steps
      // (Formula stores steps outermost-first, our list is innermost-first)
      f = Formula(values[0]);
      // withStep appends, and steps are stored outermost-first (reversed),
      // so add them in reverse order
      for (final step in stepList.reversed) {
        f = f.withStep(step.operation, step.firstIndex, step.secondIndex);
      }
      return f;
    }

    test('a/(b/c) and (a*c)/b produce same canonical key', () {
      // Cards: [8, 4, 2, x] — use first 3 for this test
      // Expression 1: 8 / (4 / 2) = 4
      // Expression 2: (8 * 2) / 4 = 4
      // Using cards [8, 4, 2, 6] and only caring about canonical keys
      final cards = [8, 4, 2, 6];

      // 8 / (4 / 2): step1: 4/2→2 (indices 1,2), step2: 8/2→4 (indices 0,1)
      // After step1: [8, 2, 6], after step2: [4, 6]
      final f1 = buildFormula([8, 4, 2, 6], [('/', 1, 2), ('/', 0, 1), ('+', 0, 1)]);

      // (8 * 2) / 4: step1: 8*2→16 (indices 0,2), step2: 16/4→4 (indices 0,1)
      // After step1: [16, 4, 6], after step2: [4, 6]
      final f2 = buildFormula([8, 4, 2, 6], [('*', 0, 2), ('/', 0, 1), ('+', 0, 1)]);

      final key1 = Expr.canonicalKey(f1, cards);
      final key2 = Expr.canonicalKey(f2, cards);
      expect(key1, equals(key2),
          reason: 'a/(b/c) and (a*c)/b should have the same canonical key');
    });

    test('a-(b-c) and (a-b)+c produce same canonical key', () {
      final cards = [10, 3, 1, 2];

      // 10 - (3 - 1) + 2: step1: 3-1→2 (indices 1,2), step2: 10-2→8 (indices 0,1), step3: 8+2
      final f1 = buildFormula([10, 3, 1, 2], [('-', 1, 2), ('-', 0, 1), ('+', 0, 1)]);

      // (10 - 3) + 1 + 2: step1: 10-3→7 (indices 0,1), step2: 7+1→8 (indices 0,1), step3: 8+2
      final f2 = buildFormula([10, 3, 1, 2], [('-', 0, 1), ('+', 0, 1), ('+', 0, 1)]);

      final key1 = Expr.canonicalKey(f1, cards);
      final key2 = Expr.canonicalKey(f2, cards);
      expect(key1, equals(key2),
          reason: 'a-(b-c) and (a-b)+c should have the same canonical key');
    });

    test('dedup reduces solutions for [1, 2, 3, 4]', () {
      final results = game.solve([1, 2, 3, 4]);
      // All results must equal 24
      for (final f in results) {
        expect((f.result - 24).abs() < 0.0001, isTrue);
      }
      // With canonical dedup, we should have fewer solutions than
      // string-only dedup would produce (exact count depends on impl,
      // but there should be no arithmetic duplicates)
      final keys = results.map((f) => Expr.canonicalKey(f, [1, 2, 3, 4])).toSet();
      expect(keys.length, equals(results.length),
          reason: 'Each solution should have a unique canonical key');
    });

    test('no valid solutions are lost — all card sets still produce results', () {
      final testCases = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [1, 5, 5, 5],
        [8, 3, 8, 3],
        [6, 6, 6, 6],
      ];
      for (final cards in testCases) {
        final results = game.solve(cards);
        expect(results, isNotEmpty,
            reason: 'Cards $cards should still have solutions after dedup');
        for (final f in results) {
          expect((f.result - 24).abs() < 0.0001, isTrue,
              reason: 'All solutions for $cards must equal 24');
        }
      }
    });
  });
}
