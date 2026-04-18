import math

from formula import Formula
from Operations.operation import Operation
from expr_tree import canonical_key


class Game:
    def __init__(self, operations: [Operation], viable_results: [int]):
        self.operations = operations
        self.viable_results = viable_results

    def play(self, cards: [int]):
        formulas = self.__play_recursively(cards)

        if self.viable_results:
            formulas = [f for f in formulas if f.get_result() in self.viable_results]

        # Deduplicate by canonical algebraic form
        seen = {}
        for f in formulas:
            key = canonical_key(f, cards)
            if key not in seen or len(f.steps) < len(seen[key].steps):
                seen[key] = f
        return list(seen.values())

    def __play_recursively(self, values: [int]) -> [Formula]:
        if len(values) == 1:
            formula = Formula(values[0])
            return [formula]

        formulas = []
        for first_ind in range(len(values)-1):
            for second_ind in range(first_ind+1, len(values)):
                # op(first, second): result at first_ind, remove second_ind
                remaining_values = values.copy()
                remaining_values.pop(second_ind)

                for operation in self.operations:
                    formulas.extend(
                        self.__formulas_after_calc(values, remaining_values, operation, first_ind, second_ind)
                    )

                    if not operation.is_commutative:
                        # op(second, first): result at second_ind-1, remove first_ind
                        remaining_values_swap = values.copy()
                        remaining_values_swap.pop(first_ind)
                        formulas.extend(
                            self.__formulas_after_calc_swap(values, remaining_values_swap, operation, first_ind, second_ind)
                        )

        return formulas

    def __formulas_after_calc(self, values, new_values, operation, first_ind, second_ind):
        calculated_value = operation.calc(values[first_ind], values[second_ind])

        if self.__should_reject(calculated_value, values[first_ind], values[second_ind], new_values):
            return []

        new_values[first_ind] = calculated_value
        new_formulas = self.__play_recursively(new_values)

        for formula in new_formulas:
            formula.add_step(operation, first_ind, second_ind)
        return new_formulas

    def __formulas_after_calc_swap(self, values, new_values, operation, first_ind, second_ind):
        # Computes op(second, first); result goes at second_ind-1 in new_values
        # (first_ind was removed, so second_ind shifts down by 1)
        calculated_value = operation.calc(values[second_ind], values[first_ind])

        if self.__should_reject(calculated_value, values[second_ind], values[first_ind], new_values):
            return []

        new_values[second_ind - 1] = calculated_value
        new_formulas = self.__play_recursively(new_values)

        for formula in new_formulas:
            # In the original values list: result at second_ind, consumed first_ind
            formula.add_step(operation, second_ind, first_ind)
        return new_formulas

    def __should_reject(self, value, operand1, operand2, new_values):
        """Return True if this computed value should be rejected."""
        # Reject nan/inf (invalid operations)
        if math.isnan(value) or math.isinf(value):
            return True
        # Require integer intermediate values
        if not float(value).is_integer():
            return True
        return False
