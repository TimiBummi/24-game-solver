abstract class Operation {
  double calc(double a, double b);
  String format(String a, String b);
  bool get isCommutative;
  int get priority;
  String get symbol;
}
