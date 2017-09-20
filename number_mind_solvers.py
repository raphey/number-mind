__author__ = 'raphey'

from number_mind import weighted_choice
from number_mind_specific_puzzles import *
from mimic_distribution import MimicDistribution
from util import start_timer, stop_timer
from math import exp
from random import random, randint
from itertools import combinations
import constraint
# from heapq import heappush, heappop


def prod(l):
    """
    Multiplicative summation function, returning 1.0 when l is empty
    """
    p = 1.0
    for x in l:
        p *= x
    return p


def common_string(str_a, str_b):
    """
    Returns a string with matching letters of str_a and str_b in their original position and differing
    letters replaces by underscores, for use with monitoring progress towards a known solution.
    """
    same_boolean = [str_a[i] == str_b[i] for i in range(len(str_a))]
    return ''.join(str_a[i] if same_boolean[i] else '_' for i in range(len(str_a)))


def simulated_annealing(puzzle: NumberMindPuzzle, temp0=0.5, temp_low=0.1, alpha=1.0 - 10**-7,
                        initial_state=None, use_bayes=True, verbose=False):
    """
    Simulated annealing algorithm for solving a number-mind puzzle. Starts with a random state, possibly according to a
    Bayesian probability distribution, and considers a series of random, one-digit changes, also possibly influenced by
    Bayes. Always keeps changes that reduce cost, and sometimes still keeps changes that increase cost, with more and
    more reluctance (temperature lowering) after each iteration.

    :param puzzle: A number-mind puzzle
    :param temp0: The initial temperature, i.e. willingness to increase cost. Scaled by default to be close to costs.
    :param temp_low: The temp where algorithm gives up--set so as to stop the search at a local minimum
    :param alpha: The factor 1 - eps by which temperature decreases with each iteration
    :param initial_state: The randomly selected initial state
    :param use_bayes: Whether or not Bayesian probability is used
    :param verbose: Whether or not to print ongoing progress towards a known solution.
    :return:
    """

    def sa_accept(c0, c):
        """
        Helper function for simulated annealing--decides whether or not to take a step that moves from cost c0 to c.
        """
        if c <= c0:
            return True

        a = exp((c0 - c) / temp)

        return random() < a

    if initial_state is None:
        puzzle.initialize_to_random(use_bayes)
    else:
        puzzle.initialize_to_fixed(initial_state)

    if verbose:
        print("Solving puzzle with simulated annealing...")

    temp = temp0

    i = 0

    best_cost = float('inf')

    while temp > temp_low:
        i += 1
        new_state, new_cost, new_gso = puzzle.random_successor_w_cost_gso(use_bayes)

        if new_cost == 0:
            print("{} Simulated annealing found solution after {} iterations.".format(new_state, i))
            return

        if new_cost < best_cost:
            best_cost = new_cost
            if verbose and puzzle.solution != 'UNK':
                print("Iteration: {:>6} Cost: {:>2} Comparison to known sol: {}".format(
                    i, new_cost, common_string(new_state, puzzle.solution)))

        if sa_accept(puzzle.current_cost, new_cost):
            puzzle.current_state, puzzle.current_cost, puzzle.guess_score_offset = new_state, new_cost, new_gso
        temp *= alpha

    print("Simulated annealing failed to find solution after {} iterations.".format(i))


def single_greedy_search(puzzle: NumberMindPuzzle, use_bayes=False, initial_state=None, lex_dist=1):
    """
    Hill-descending search to be used with a number-mind puzzle. Starts at a pre-specified or random point, possibly
    according to a Bayesian probability distribution, and changes digits according to whatever change will result in
    the greatest decrease in cost. With the variable lexical_distance set above 1, the state checks the results of
    multiple digit changes.
    Returns the solution string if one is found, or ''. Also returns the starting state as a string.
    """

    if initial_state is None:
        puzzle.initialize_to_random(use_bayes)
    else:
        puzzle.initialize_to_fixed(initial_state)

    start_state = puzzle.current_state

    new_state, new_cost, new_gso = puzzle.best_successor_w_cost_gso(d=lex_dist)

    while 0 < new_cost < puzzle.current_cost:
        puzzle.current_state, puzzle.current_cost, puzzle.guess_score_offset = new_state, new_cost, new_gso
        new_state, new_cost, new_gso = puzzle.best_successor_w_cost_gso(d=lex_dist)

        if new_cost == 0:
            return new_state, start_state

    return None, start_state


def repeated_greedy_search(puzzle: NumberMindPuzzle, max_tries=100000, use_bayes=False, lex_dist=1, persist=False,
                           verbose=False):
    """
    Repeated hill-descending search. For up to max_tries attempts, starts at a random point, possibly according to a
    Bayesian probability distribution, and changes digits according to whatever change will result in the greatest
    decrease in cost. If persist is true, the search continues after finding solution.
    """
    if verbose:
        print("Solving puzzle with repeated greedy search...")

    counter = 0
    for i in range(max_tries):
        sol, start = single_greedy_search(puzzle, use_bayes, None, lex_dist)

        if sol:
            counter += 1
            counter_string = ""
            if persist:
                counter_string = " ({} so far)".format(counter)
            if verbose or not persist:
                print("{} Repeated greedy search found solution after {} iterations, "
                      "starting from state {}.{}".format(sol, i, start, counter_string))
            if not persist:
                return

    if counter == 0:
        print("Repeated greedy search failed to find solution after {} iterations.".format(max_tries))
    else:
        print("Repeated greedy search solution count after {} iterations: {}".format(max_tries, counter))
    return


def mimic_greedy_search(puzzle: NumberMindPuzzle, max_tries=100000, pop_size=10000, cutoff_proportion=0.6,
                        generations=50, lex_dist=1, persist=False, verbose=False):
    """
    Repeated hill-descending search with samples taken from a trained MIMIC distribution. After training the
    distribution, for up to max_tries attempts, starts at a randomly generated point and changes digits according to
    whatever change will result in the greatest decrease in cost. If persist is true, search continues after finding a
    solution.
    """

    if verbose:
        print("Solving puzzle with MIMIC algorithm and repeated greedy search...")
    mimic_dist = MimicDistribution(puzz=puzzle, pop_size=pop_size)
    mimic_dist.train(cutoff_proportion=cutoff_proportion, generations=generations, verbose=verbose)
    counter = 0

    for i in range(max_tries):
        sample = mimic_dist.bivariate_sample()
        sol, start = single_greedy_search(puzzle, False, sample, lex_dist)

        if sol:
            counter += 1
            counter_string = ""
            if persist:
                counter_string = " ({} so far)".format(counter)
            if verbose or not persist:
                print("{} Mimic greedy search found solution after {} iterations, "
                      "starting from state {}.{}".format(sol, i, start, counter_string))
            if not persist:
                return
    if counter == 0:
        print("Mimic greedy search failed to find solution after {} iterations.".format(max_tries))
    else:
        print("Mimic greedy search solution count after {} iterations: {}".format(max_tries, counter))


def constraint_solver(puzzle: NumberMindPuzzle):
    """
    Solves a number-mind puzzle using Gustavo Niemeyer's python constraint module. This isn't suitable
    for solving the length 16 number-mind puzzle, but it can solve the length 5 puzzle.
    """

    def make_constraint_function(guess, score):
        """
        Helper function for the constraint solver. Returns a function that gives a score for a guess vs a
        proposed solution.
        """
        def cf(*args):
            num_corr = sum(guess[j] == args[j] for j in range(len(guess)))
            return num_corr == score
        return cf

    problem = constraint.Problem(constraint.RecursiveBacktrackingSolver())
    v_list = ['v' + str(a) for a in range(puzzle.length)]
    for v in v_list:
        problem.addVariable(v, list('01234567890'))
    for i in range(puzzle.num_guesses):
        problem.addConstraint(make_constraint_function(puzzle.guesses[i], puzzle.scores[i]), v_list)

    sol = problem.getSolution()
    if sol:
        sol_str = ''.join(sol[v] for v in v_list)
        print("{} Generalized constraint solver found solution.".format(sol_str))
        return
    print("Generalized constraint solver determined that puzzle has no solution.")


def genetic_algorithm(puzzle: NumberMindPuzzle, population_size=1000, civilizations=100, generations=100,
                      mutation_rate=0.5, weighted_par_sel=False, use_bayes=False, verbose=False):
    """
    Genetic algorithm for solving a number-mind-bad puzzle.

    :param puzzle: A NumberMindPuzzle
    :param population_size: The initial and ongoing size of the population of states
    :param civilizations: How many times the genetic algorithm will be run starting from a fresh population.
    :param generations: How many generations are carried out before a the algorithm gives up
    :param mutation_rate: The likelihood that a child has a single digit changed at random
    :param weighted_par_sel: Whether or not parents are selected via roulette sampling, vs uniform
    :param use_bayes: Whether the initial pop. comes from Bayesian probability distribution
    :param verbose: Whether to display ongoing updates
    :return: None, just prints solution.
    """

    def ga_child(p1, p2):
        """
        Given a two parents and a mutation rate, returns a child with each character taken at random
        from a parent, with a chance of having a single character mutated into a random digit.
        """
        c = ''
        for x, y in zip(p1, p2):
            if randint(0, 1):
                c += x
            else:
                c += y

        if random() < mutation_rate:
            a = randint(0, len(c) - 1)
            b = str(randint(0, 9))
            c = c[:a] + b + c[a + 1:]

        return c
    if verbose:
        print("Solving puzzle with genetic algorithm...")
    for civ in range(civilizations):
        if verbose:
            print("Civilization {}".format(civ))
        initial_population = [puzzle.random_state(use_bayes) for _ in range(population_size)]
        pop_and_cost = [(p, puzzle.cost(p)) for p in initial_population]

        for g in range(generations):

            # Sort population by increasing cost
            pop_and_cost.sort(lambda x: x[1])

            # Check for full solution
            if pop_and_cost[0][1] == 0:
                print("{} Genetic algorithm found solution after {} generations, in civilization {}.".format(
                    pop_and_cost[0][0], g, civ))
                return

            # Cull population back down to population size
            del pop_and_cost[population_size:]

            # Print updated progress towards known solution
            if verbose and g % 10 == 0:
                best_five = [p for p, _ in pop_and_cost[:5]]
                if puzzle.solution != 'UNK':
                    best_five_str = '\t'.join(common_string(puzzle.solution, p) for p in best_five)
                else:
                    best_five_str = '\t'.join(best_five)
                print("Generation {:>4}\t Best five: {}".format(
                    g, best_five_str))

            # Assign fitness scores to current population (only needed with weighted parent selection)
            if weighted_par_sel:
                pop, fitness = [p for p, _ in pop_and_cost], [1.0 / c for _, c in pop_and_cost]

            for _ in range(population_size):
                # Pick two parents
                if weighted_par_sel:
                    parent1 = weighted_choice(pop, fitness)
                    parent2 = weighted_choice(pop, fitness)
                else:
                    parent1 = pop_and_cost[randint(0, population_size - 1)][0]
                    parent2 = pop_and_cost[randint(0, population_size - 1)][0]

                # Make a child and add it to population along with its cost
                child = ga_child(parent1, parent2)
                pop_and_cost.append((child, puzzle.cost(child)))

    print("Genetic algorithm failed to find solution after {} civilizations of {} generations.".format(
        civilizations, generations))


def backtracking_guess_search(puzzle: NumberMindPuzzle, use_bayes=False, p_cutoff_factor=0.5):
    """
    Guess-wise backtracking search for solving a NumberMindPuzzle.
    The key logic is this: if we know a guess has, for example, two correctly placed digits, we can try picking
    one pair of digits, then make a selection for the next guess, and so on, backtracking when we hit a problem.
    :param puzzle: A number-mind-bad puzzle
    :param use_bayes: Use Bayesian probabilities to check selections within guesses from most to least likely
    :param p_cutoff_factor: We cut off selections that are less than p_cutoff_factor times mean selection probability,
    at the risk of possibly not finding a solution. 1.0 seems to work for many puzzles, but more complex puzzles might
    require values closer to 0.5, and some puzzles could have faster solutions with values > 1.0
    :return: None, just prints solution.
    """
    def bgs(g_i):
        """
        Recursive helper function that makes reference to and modifies the assignments list in the enclosing function.
        For guess[g_i], this function checks each possible selection of correct digits for validity, and if valid,
        makes that assignment and recurses, undoing the assignment if the recursive call fails.
        """

        nonlocal counter

        # Check if we've made it past the final guess, meaning we have a solution.
        if g_i == ng:
            return True

        g, s = sorted_guesses[g_i], sorted_scores[g_i]

        for selection in guess_sel_as_bool[g_i]:

            counter += 1

            # Check if assignments are valid
            assignment_valid = True

            for a in range(n):
                b = g[a]
                if selection[a]:
                    if assignments[a][b] < 0:
                        assignment_valid = False
                        break
                elif assignments[a][b] > 0:
                    assignment_valid = False
                    break

            # Move to next possible set of assignments if current set is invalid
            if not assignment_valid:
                continue

            # Make positive and negative assignments
            for a in range(n):
                b = g[a]
                if selection[a]:
                    for c in range(0, 10):
                        if c == b:
                            assignments[a][c] += 1
                        else:
                            assignments[a][c] -= 1
                else:
                    assignments[a][b] -= 1

            # Recursive call
            if bgs(g_i + 1):
                return True

            # Undo assignments because recursive call has returned False
            for a in range(n):
                b = g[a]
                if selection[a]:
                    for c in range(0, 10):
                        if c == b:
                            assignments[a][c] -= 1
                        else:
                            assignments[a][c] += 1
                else:
                    assignments[a][b] += 1

        # Every sub-branch of the recursion has failed to find a solution, so return false
        return False

    n = puzzle.length
    ng = puzzle.num_guesses

    # Sort guesses and scores by increasing score, so as to reduce early branching.
    sorted_guesses, sorted_scores = zip(*sorted(zip(puzzle.int_guesses, puzzle.scores), key=lambda x: x[1]))

    # Array of integers to track assignments. assignments[a][b] > 0 means position a has been assigned value b.
    # Less than zero means a is prohibited from having value b, and 0 means unassigned.
    assignments = [[0] * 10 for _ in range(puzzle.length)]

    # Combinatorial digit selections for each guess
    gs_list = [list(combinations(range(n), s)) for s in sorted_scores]

    search_size = prod(len(x) for x in gs_list)

    if use_bayes:

        # Get probabilities for each selection for each guess
        prob_dicts = [{} for _ in range(ng)]
        for i, sl in enumerate(gs_list):
            for sel in sl:
                prob_dicts[i][sel] = prod([puzzle.b_prob[a][str(sorted_guesses[i][a])] for a in sel])

        # For each guess, sort selections by decreasing likelihood, and cut off less likely selections.
        for i, gs in enumerate(gs_list):
            gs.sort(key=lambda sel: -prob_dicts[i][sel])
            p_cutoff = sum(prob_dicts[i][sel] for sel in gs) / len(gs_list[i]) * p_cutoff_factor
            for j, sel in enumerate(gs):
                prob = prob_dicts[i][sel]
                if prob < p_cutoff and j > 0:
                    gs_list[i] = gs[:j]
                    break

    # Save time by converting digit selections into boolean lists, i.e. (0, 3) --> [True, False, False, True, False]
    guess_sel_as_bool = [[[x in sel for x in range(n)] for sel in guess_sel] for guess_sel in gs_list]

    counter = 0

    result = bgs(0)

    if not result:
        if use_bayes:
            print("Backtracking guess search failed to find a solution after exploring {} of {:0.1e} guess search "
                  "nodes. (Cutoff may be too high.)".format(counter, search_size))
        else:
            print("Backtracking guess search concluded there is no solution by exploring {} of {:0.1e} guess search "
                  "nodes.".format(counter, search_size))
        return

    sol = ''
    for digit in assignments:
        sol += str(max(range(10), key=lambda x: digit[x]))

    print("{} Backtracking guess search found solution after exploring {} of {:0.1e} guess search nodes.".format(
        sol, counter, search_size))


def backtracking_digit_search(puzzle: NumberMindPuzzle, use_bayes=False, p_cutoff_factor=1.4, verbose=False):
    """
    Digit-wise backtracking search for solving a NumberMindPuzzle. Essentially, it assigns digits, starting with all
    zeros, and then incrementing the last digit, except that it backtracks once it determines that a current partial
    assignment will cause a problem with the needed score for a pre-made guess--either too many correct digits, or
    not enough digits remaining to get up to the needed number.
    :param puzzle: A number-mind puzzle
    :param use_bayes: Use Bayesian probabilities to check digits in order from most to least likely and also assign
    digits with the most likely candidates assigned first, to end the search sooner.
    :param p_cutoff_factor: Only has an effect with use_bayes set to True. With path cost defined as the sum of -log(p)
    for the assignments made thus far, we cut off paths after they have exceeded a maximum path cost. Maximum path cost
    is p_cutoff_factor times the sum of -log(p) of 5th most probable choice for each position.
    :param verbose: If true, prints updates every 1e5 iterations.
    :return: None, just prints solution.
    """
    def bds(digit_i):
        """
        Recursive helper function that makes reference to and modifies the digit_assignments list in the enclosing
        function. For digit_order[digit_i], this function tries each possible digit, checks for validity, and if valid,
        makes that assignment and recurses, undoing the assignment if the recursive call fails.
        """
        nonlocal counter
        nonlocal path_cost
        counter += 1

        if verbose and counter % 100000 == 0:
            current_state = '\t'.join(str(digit_assignments[x]) for x in digit_order)
            print("{}\t ({} iterations)".format(current_state, counter))

        # Check if we've made it past the final digit assignment, meaning we have a solution
        if digit_i == n:
            return True

        # a is the actual digit we'll be assigning, according to the original ordering
        a = digit_order[digit_i]

        for b in digit_possibilities[a]:

            # Assign digit
            digit_assignments[a] = b
            for i, g in enumerate(int_guesses):
                if g[a] == b:
                    correct_needed[i] -= 1

            # Increment path_cost
            if use_bayes:
                old_path_cost, path_cost = path_cost, path_cost + digit_neg_log_p_list[a][b]

            # Check if assignment is valid
            assignment_valid = True
            remaining_digits = n - (digit_i + 1)
            if use_bayes:
                if path_cost > path_cost_cutoff:
                        path_cost = old_path_cost
                        assignment_valid = False
            if assignment_valid:
                for i, cn in enumerate(correct_needed):
                    if cn < 0 or cn > remaining_digits:
                        assignment_valid = False
                        break

            # Recurse on next digit if assignment is valid
            if assignment_valid:
                if bds(digit_i + 1):
                    return True

            # Unassign digit
            if use_bayes:
                path_cost = old_path_cost
            digit_assignments[a] = -1
            for i, g in enumerate(int_guesses):
                if g[a] == b:
                    correct_needed[i] += 1

        # Return false; this branch has failed
        return False

    if verbose:
        print("Solving puzzle with backtracking digit search...")

    n = puzzle.length
    ng = puzzle.num_guesses
    int_guesses = [list(map(int, g)) for g in puzzle.guesses]
    digit_possibilities = [list(range(10)) for _ in range(n)]
    correct_needed = [puzzle.scores[i] for i in range(ng)]
    digit_assignments = [-1] * n
    digit_order = list(range(n))

    # If using Bayesian probabilities, change the order in which digit indices will be assigned to digits, and also
    # the order of those digit assignments
    if use_bayes:
        for a, dp in enumerate(digit_possibilities):
            dp.sort(key=lambda b: -puzzle.b_prob[a][str(b)])
        digit_neg_log_p_list = [[puzzle.neg_log_p[a][str(b)] for b in range(10)] for a in range(n)]
        path_cost_cutoff = sum(digit_neg_log_p_list[a][digit_possibilities[a][4]] for a in range(n)) * p_cutoff_factor
        digit_order.sort(key=lambda a: min(digit_neg_log_p_list[a]))

    # Print the digit reordering, and if applicable print the known solution.
    if verbose:
        print("{} \t (Digit indices)".format('\t'.join(map(str, digit_order))))
        if puzzle.solution != 'UNK':
            scrambled_solution = '\t'.join(puzzle.solution[x] for x in digit_order)
            print(scrambled_solution, '\t (Known solution, reordered)')
        print()

    path_cost = 0
    counter = 0
    result = bds(0)

    if result:
        sol = ''.join(map(str, digit_assignments))
        print("{} Backtracking digit search found solution after {} iterations.".format(sol, counter))
        return

    print("Backtracking digit search failed to find a solution.")


def heap_digit_search(puzzle: NumberMindPuzzle, use_bayes=False, fixed_start=True):
    """
    This is based on an approach taken by Project Euler user Sandorf. I eventually removed the code since I couldn't
    contact him to check about posting a variant of what he wrote.
    This is a digit-wise, A* like search in which the lowest cost node is expanded with single-digit variations.
    My main addition to Sandorf's idea is to speed things up by 30-40% by changing how costs are computed.
    Rather than compute each cost from scratch, I look up a stored tuple with a 'guess score offset," as described
    in the NumberMindPuzzle class. This takes advantage of the fact that the costs for one state don't differ much
    from the costs of a state that is one digit away.

    :param puzzle: A number-mind puzzle
    :param use_bayes: Use Bayesian probabilities to determine the initial state.
    :param fixed_start: Determines if the starting state is fixed or randomly determined. With use_bayes set to True,
    fixed starting state is the argmax for each digit. With use_bayes set to False, fixed starting state is all zeros.
    :return: None, prints solution.
    """

    return


# ==================================================================

if __name__ == '__main__':
    # Solve the standard puzzle a few different ways
    # This is how sample_output.txt was generated

    puzz = standard_puzzle  # Other options are in number_mind_specific_puzzles.py

    start_timer()
    simulated_annealing(puzz, use_bayes=True, verbose=True)
    stop_timer()

    print()

    start_timer()
    genetic_algorithm(puzz, verbose=True)
    stop_timer()

    print()

    start_timer()
    backtracking_guess_search(puzz, use_bayes=True)
    stop_timer()

    print()

    start_timer()
    mimic_greedy_search(puzz, verbose=True)
    stop_timer()