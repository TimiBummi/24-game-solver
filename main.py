from Operations.addition import Addition
from Operations.division import Division
from Operations.exp import Exp
from Operations.log import Log
from Operations.multiplication import Multiplication
from Operations.sqrt import Sqrt
from Operations.subtraction import Subtraction
from game import Game

OPERATIONS_BASE = [Addition, Subtraction, Division, Multiplication]
OPERATIONS_EXTENDED = OPERATIONS_BASE + [Exp, Log, Sqrt]


def show_solutions(game, cards):
    formulas = game.play(cards)
    if formulas:
        print(f"Cards {cards} — {len(formulas)} solution(s):")
        for formula in formulas:
            print("\t" + formula.to_string(cards))
    else:
        print(f"Cards {cards} — no solution.")


def main():
    rounds = [
        [1, 6, 11, 13],
        [2, 3, 4, 8],
    ]

    print("=== Base game (+ - * /) ===")
    base_game = Game(OPERATIONS_BASE, [24])
    for cards in rounds:
        show_solutions(base_game, cards)

    print("\n=== Extended game (+ - * / ^ log sqrt) ===")
    extended_game = Game(OPERATIONS_EXTENDED, [24])
    for cards in rounds:
        show_solutions(extended_game, cards)


if __name__ == "__main__":
    main()
