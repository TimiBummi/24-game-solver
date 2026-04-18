from Operations.operation import Operation


# Formulas don't hold the information of the actual values

class Formula:
    def __init__(self, result):
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        self.result = result
        self.steps = []

    def get_result(self):
        return self.result

    def add_step(self, operation: Operation, first_ind: int, second_ind: int):
        self.steps.append({
            "operation": operation,
            "first_ind": first_ind,
            "second_ind": second_ind
        })

    def to_string(self, values: [int]):
        # (a+d)*(c+b) would be (+, 0, 3), (+, 1, 2), (*, 0, 1) in reversed order
        formula = [value.__str__() for value in values]
        for i, step in enumerate(reversed(self.steps)):
            formula[step["first_ind"]] = step["operation"].to_string(
                formula[step["first_ind"]],
                formula[step["second_ind"]]
            )
            if i != len(self.steps)-1:
                formula[step["first_ind"]] = "(" + formula[step["first_ind"]] + ")"
            formula.pop(step["second_ind"])


        formula[0] += " = " + self.result.__str__()
        return formula[0]