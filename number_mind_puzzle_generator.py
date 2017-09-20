__author__ = 'raphey'

from number_mind import NumberMindPuzzle, match_count
from integer_programming_solver import integer_programming_solver
from random import randint


def generate_puzzle(length, guess_score_limit=3, zero_correct_limit=1):
    """
    Generates a NumberMind puzzle, with some attempt to make one that looks like the given puzzle in terms of the
    distribution of guess scores.
    One difference is that the given puzzle has substantially fewer guesses, and substantially more 3-correct guesses
    than would be expected from a random distribution.
    :param length: How many digits the puzzle has
    :param guess_score_limit: What's the largest number correct for a guess to be admissible
    :param zero_correct_limit: How many guesses with zero correct will be allowed--for the given puzzle, seemingly 1
    :return: a NumberMindPuzzle object.
    """

    solution = str(10 ** length + randint(0, 10 ** length - 1))[1:]
    guesses = []
    scores = []
    zero_correct_count = 0
    unique_solution = False
    while not unique_solution:
        new_guess = str(10 ** length + randint(0, 10 ** length - 1))[1:]
        new_score = match_count(solution, new_guess)
        if new_score == 0:
            if zero_correct_count < zero_correct_limit:
                zero_correct_count += 1
            else:
                continue
        if new_score > guess_score_limit:
            continue
        guesses.append(new_guess)
        scores.append(new_score)
        puzz = NumberMindPuzzle(guesses=guesses, scores=scores, solution=solution)
        if integer_programming_solver(puzz, persist=True, verbose=False) == 1:
            unique_solution = True
    return puzz


def print_puzzle_definition_string(length):
    """
    Since I'm not able to use the same interpreter for puzzle generation and puzzle solving, this helper function
    prints out strings I can paste directly into the code in the given puzzles document.
    """
    new_puzz = generate_puzzle(length)
    print("generated_puzzle_length{} = NumberMindPuzzle(guesses={},scores={}, "
          "solution='{}')".format(length, new_puzz.guesses, new_puzz.scores, new_puzz.solution))
    print("(Number of guesses: {})".format(new_puzz.num_guesses))

if __name__ == '__main__':
    # Generate some puzzles and put the results into a puzzle definition string
    n = 8
    num_puzzles = 10
    for _ in range(num_puzzles):
        print_puzzle_definition_string(n)