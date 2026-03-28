import 'operation.dart';

class Division implements Operation {
  const Division();

  @override
  double calc(double a, double b) => b == 0.0 ? double.nan : a / b;

  @override
  String format(String a, String b) => '$a/$b';

  // Matching Python original: set to true.
  // Set to false to find all solutions including reversed operand order.
  @override
  bool get isCommutative => false;

  @override
  int get priority => 1;

  @override
  String get symbol => '/';
}
