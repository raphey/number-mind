__author__ = 'raphey'
# Separate file for solution using Google's OR tools
# Can't run with pypy, needs regular python 3 with py3-ortools installed


from ortools.linear_solver import pywraplp
from number_mind import NumberMindPuzzle
from util import start_timer, stop_timer


def integer_programming_solver(puzzle: NumberMindPuzzle, persist=False, max_sol_count=2, verbose=True):
    """
    Solves a number-mind puzzle using mixed-integer programming from Google's OR tools.
    :param puzzle: A number-mind puzzle object
    :param persist: Whether the solver should continue finding additional solutions after finding one
    :param max_sol_count: The maximum number of solutions to find before stopping. 2 is a useful default, because it
    determines puzzle uniqueness
    :param verbose: With verbose set to False, this function prints nothing at all, which is different from the other
    solving methods, since this method can be used to help create new puzzles.
    :return:
    """

    # Mixed integer solver
    solver = pywraplp.Solver('SolveIntegerProblem', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    n = puzzle.length

    # Collect all variables in two different data structures
    vars_array = []
    vars_list = []
    for a in range(n):
        vars_array.append([])
        for b in range(10):
            vars_array[a].append(solver.IntVar(0, 1, 'v' + str(a) + str(b)))
            vars_list.append(vars_array[a][b])

    # Set unique assignment constraints
    unique_assignment_constraints = []
    for a in range(n):
        unique_assignment_constraints.append(solver.Constraint(1, 1))
        for b in range(10):
            unique_assignment_constraints[a].SetCoefficient(vars_array[a][b], 1)

    # Guess/score constraints
    guess_score_constraints = []
    for i, (g, s) in enumerate(zip(puzzle.guesses, puzzle.scores)):
        h = list(map(int, g))
        guess_score_constraints.append(solver.Constraint(s, s))
        for j in range(n):
            guess_score_constraints[i].SetCoefficient(vars_array[j][h[j]], 1)

    sol_count = 0

    while sol_count < max_sol_count:
        result_status = solver.Solve()
        if result_status == 0:
            sol_count += 1
            sol = ''
            for v in vars_list:
                if v.solution_value() == 1:
                    sol += str(v)[-1]

            if verbose or not persist:
                print("{} Integer programming solver found solution.".format(sol))
            if not persist:
                return sol_count
            if verbose:
                print("  Searching for an additional solution...")

            # Add a new constraint prohibiting solution we've already found
            h = list(map(int, sol))
            guess_score_constraints.append(solver.Constraint(0, n - 1))
            for j in range(n):
                guess_score_constraints[-1].SetCoefficient(vars_array[j][h[j]], 1)
        else:
            if not persist:
                print("Integer programming solver determined that there is no solution")
            if verbose:
                print("Integer programming solver found all possible solutions. Solution count: {}".format(sol_count))
            return sol_count
    if verbose:
        print("Integer programming solver found {} solutions; there may be more.")
    return sol_count


if __name__ == '__main__':

    # Solve some problems, including a couple big ones.

    from number_mind_specific_puzzles import *

    start_timer()
    integer_programming_solver(standard_puzzle)
    stop_timer()

    start_timer()
    integer_programming_solver(generated_puzzle_length16_1)
    stop_timer()

    start_timer()
    integer_programming_solver(hard_puzzle_1)
    stop_timer()

    start_timer()
    integer_programming_solver(generated_puzzle_length20)
    stop_timer()