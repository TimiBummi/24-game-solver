import 'operation.dart';

class Subtraction implements Operation {
  const Subtraction();

  @override
  double calc(double a, double b) => a - b;

  @override
  String format(String a, String b) => '$a-$b';

  // Matching Python original: set to true (non-commutative branch not used).
  // Set to false to find all solutions including reversed operand order.
  @override
  bool get isCommutative => false;

  @override
  int get priority => 0;

  @override
  String get symbol => '-';
}
