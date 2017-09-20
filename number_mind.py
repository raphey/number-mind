__author__ = 'raphey'
# Class for Number Mind puzzles

from random import randint, uniform
from itertools import combinations, combinations_with_replacement
from math import log


class NumberMindPuzzle(object):
    """
    A Number Mind puzzle, based on Project Euler 185.
    Includes the guesses and scores needed to specify the problem, as well as attributes and methods
    to be used by various solving algorithms.
    """
    def __init__(self, guesses, scores, solution='UNK'):
        """
        Guesses are the pre-made guesses that specify the puzzle, along with their respective scores.
        A known solution can be specified, for comparison while solving.
        """
        self.guesses = guesses
        self.int_guesses = [list(map(int, g)) for g in self.guesses]
        self.scores = scores
        self.num_guesses = len(guesses)

        if len(guesses) > 0:
            self.length = len(guesses[0])
        else:
            if solution == 'UNK':
                raise Exception("Trying to initialize number puzzle without guesses or specified solution")
            self.length = len(solution)

        self.solution = solution

        # Bayesian probabilities are computed whether or not a given solving algorithm uses them
        self.b_prob = bayesian_probability_dicts(guesses, scores)
        self.neg_log_p = [dict((v, -log(p + 1e-12)) for v, p in d.items()) for d in self.b_prob]
        self.entropy = [sum(self.b_prob[a][b] * self.neg_log_p[a][b] for b in '0123456789')
                        for a in range(self.length)]
        # Argmax digit value for each digit.
        self.best_guess = ''.join(max((b for b in '0123456789'), key=lambda x: self.b_prob[a][x])
                                  for a in range(self.length))

        # The three attributes below need to be set with one of the initialization functions
        self.current_state = ''
        self.guess_score_offset = [None] * self.num_guesses
        self.current_cost = None

    def random_state(self, use_bayes=False):
        """
        Gets a random string of digits with the correct length for the puzzle, optionally influenced
        by Bayesian probabilities.
        """
        if use_bayes:
            random_state = ''
            for i in range(self.length):
                random_state += weighted_choice(self.b_prob[i].keys(), self.b_prob[i].values())
        else:
            random_state = str(10 ** self.length + randint(0, 10 ** self.length - 1))[1:]
        return random_state

    def initialize_to_random(self, use_bayes=False):
        """
        Initializes current_state to a random value, optionally influenced by Bayesian probabilities.
        Also calls a function to set the current_cost and guess_score_offset attributes.
        """
        self.current_state = self.random_state(use_bayes)
        self.set_cost_and_gso()

    def initialize_to_fixed(self, initial_state):
        """
        Initializes current_state to a specified string.
        Also calls a function to set the current_cost and guess_score_offset attributes.
        """
        self.current_state = initial_state
        self.set_cost_and_gso()

    def set_cost_and_gso(self):
        """
        Sets the current_cost and guess_score_offset attributes. The latter is a list of integers
        that tracks how the guesses compare to the current state relative to their scores. A valid
        solution would have a guess_score_offset of [0, 0, 0, ...]. A solution whose only flaw was
        one extra correct digit in the first guess would have a guess_score_offset of [1, 0, 0, ...].
        """
        for i in range(self.num_guesses):
            matches = match_count(self.current_state, self.guesses[i])
            self.guess_score_offset[i] = matches - self.scores[i]
        self.current_cost = sum(abs(x) for x in self.guess_score_offset)

    def get_new_cost_and_gso(self, change_list):
        """
        Returns a cost and a new guess score offset for a new state, where the new state is specified
        by a list of (digit_place, new_character) changes to be made relative to current_state.
        """
        new_gso = self.guess_score_offset[:]
        for a, b in change_list:                # Digit in position a is being changed to digit b
            if self.current_state[a] == b:      # Do nothing if the new character is the old one
                continue
            for i in range(self.num_guesses):
                if self.guesses[i][a] == self.current_state[a]:     # Old digit matches guess i at position a
                    new_gso[i] -= 1                                 # One fewer match with that guess
                elif self.guesses[i][a] == b:                       # New digitmatches guess i at position a
                    new_gso[i] += 1                                 # One more match with that guess

        return sum(abs(x) for x in new_gso), new_gso

    def cost(self, state):
        """
        Cost function that can be fed a state string (not necessarily the puzzle's current state),
        returning the sum of the absolute values of the differences between guess scores for the
        true solution and guess scores for the given state string.
        This computes the scores from scratch, meaning it can't take advantage of the state string
        being a small variation of a previously scored state.
        """
        c = 0
        for i in range(self.num_guesses):
            matches = match_count(state, self.guesses[i])
            c += abs(matches - self.scores[i])
        return c

    def simple_best_successor(self, state):
        """
        Returns the lowest-cost state that is one digit away from a given state.
        """
        best_state = state
        best_cost = float('inf')
        for a in range(self.length):
            for b in '0123456789':
                new_state = state[:a] + b + state[a + 1:]
                new_cost = self.cost(new_state)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = new_state
        return best_state

    def best_successor_w_cost_gso(self, d=1):
        """
        Returns the lowest-cost state that is lexical distance d away from the current state.
        Also returns the cost and the guess score offset for the best state.
        """
        best_state = self.current_state
        best_cost = float('inf')
        best_gso = []
        for ac in combinations(range(self.length), d):
            for bc in combinations_with_replacement('0123456789', d):
                new_cost, new_gso = self.get_new_cost_and_gso(zip(ac, bc))
                if new_cost < best_cost:
                    best_cost = new_cost
                    s = self.current_state
                    for a, b in zip(ac, bc):
                        s = s[:a] + b + s[a + 1:]
                    best_state = s
                    best_gso = new_gso
        return best_state, best_cost, best_gso

    def random_successor(self, state, use_bayes=False):
        """
        Returns a random successor that is one digit from a given state, optionally influenced
        by Bayesian probability regarding which digit is changed and to what value it's changed.
        Also returns change index and new digit for use with other functions.
        """
        if use_bayes:
            a = weighted_choice(range(self.length), [1.0 - self.b_prob[i][state[i]] for i in range(self.length)])
            b = weighted_choice(self.b_prob[a].keys(), self.b_prob[a].values())
        else:
            a = randint(0, self.length - 1)
            b = str(randint(0, 9))

        return state[:a] + b + state[a + 1:], a, b

    def random_successor_w_cost_gso(self, use_bayes=False):
        """
        Returns a random successor that is one digit from the current state, optionally influenced
        by Bayesian probability regarding which digit is changed and to what value it's changed.
        Also returns the cost and the guess score offset for the random successor state.
        """

        new_state, a, b = self.random_successor(self.current_state, use_bayes)
        new_cost, new_gso = self.get_new_cost_and_gso([(a, b)])

        return new_state, new_cost, new_gso


def bayesian_probability_dicts(guesses, scores):
    """
    Given a list of Number Mind guess strings and their whole number scores, returns a list of
    dictionaries called prob_dicts, where prob_dicts[0]['3'] gives the probability that the
    true solution has the digit 3 in position 0.
    """

    def bayes_prob_helper(a, b):
        """
        Helper function that goes from the occurrences of a given digit/position in guesses with
        various scores to the Bayesian probability of that digit occurring in that position.

        For example: given the information that, with some number of random five-digit guesses,
        we had one guess with one right with a 3 in position 0, and another guess with two right
        that had a 3 in position 0, our guess occurrences for 3 would be [0, 1, 1, 0, 0, 0], and
        this function would calculate a probability 0.6 that the first digit is a 3.

        This hinges on the assumption that the guesses are generated randomly.
        """

        f = 1.0
        if guess_occurrence_matrix[a][b][0] > 0 or guess_nonoccurrence_matrix[a][b][n] > 0:
            return 0.0
        if guess_occurrence_matrix[a][b][n] > 1:
            return 1.0
        for j in range(1, n):
            x_j = guess_occurrence_matrix[a][b][j]
            f *= ((n - j) / (j * 9)) ** x_j
        for j in range(1, n):
            x_j = guess_nonoccurrence_matrix[a][b][j]
            f *= (j / (n - j) + 8.0 / 9.0) ** x_j
        return 1 / (1 + 9 * f)

    n = len(guesses[0])
    guess_occurrence_matrix = [[[0] * (n + 1) for _ in range(10)] for _ in range(n)]
    guess_nonoccurrence_matrix = [[[0] * (n + 1) for _ in range(10)] for _ in range(n)]

    for i, g in enumerate(guesses):
        s = scores[i]
        for a in range(0, n):
            d = int(g[a])
            guess_occurrence_matrix[a][d][s] += 1
            for e in range(10):
                if e != d:
                    guess_nonoccurrence_matrix[a][e][s] += 1

    raw_probs = [[bayes_prob_helper(a, b) for b in range(10)] for a in range(n)]

    sums_by_place = [sum(raw_probs[a]) for a in range(n)]

    prob_dicts = [dict((str(b), raw_probs[a][b] / sums_by_place[a]) for b in range(10)) for a in range(n)]

    return prob_dicts


def weighted_choice(choices, weights):
    """
    Based on a post by StackOverflow user Ned Batchelder.
    Given a list of choices and weights, picks one choice randomly according to weight.
    """
    total = sum(weights)
    r = uniform(0, total)
    running_sum = 0.0
    for c, w in zip(choices, weights):
        if running_sum + w >= r:
            return c
        running_sum += w


def match_count(str_a, str_b):
    """
    Returns the number of matching characters/positions between str_a and str_b
    """
    return sum(str_a[i] == str_b[i] for i in range(len(str_a)))