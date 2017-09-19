__author__ = 'raphey'

from number_mind import NumberMindPuzzle, match_count
from integer_programming_solver import integer_programming_solver
from random import randint


def generate_puzzle(length, guess_score_limit=3, zero_correct_limit=1):
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

new_puzz = generate_puzzle(20)

print(new_puzz.guesses)
print(new_puzz.scores)
print(new_puzz.solution)