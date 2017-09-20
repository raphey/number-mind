__author__ = 'raphey'

from math import log
from random import shuffle
from number_mind import NumberMindPuzzle, weighted_choice


class MimicDistribution(object):
    """
    MIMIC distribution object designed to create random samples in the number-mind puzzle space that have low cost.
    This is based on work by De Bonet, Isbell, and Viola.
    Once the distribution has been initialized and trained with the self.train() method, samples can be generated using
    self.bivariate_sample().
    """
    def __init__(self, puzz: NumberMindPuzzle, pop_size=10000):
        """
        Constructor for a Mimic distribution object. Should be passed a number-mind puzzle and a population size which
        represents how many new samples will be generated with each successive generation of training.
        """
        self._puzzle = puzz
        self._n = self._puzzle.length
        self.pop_size = pop_size
        self._pop = [self._puzzle.random_state() for _ in range(pop_size)]
        self._pop_and_cost = [(p, self._puzzle.cost(p)) for p in self._pop]

        # Normally, culled pop is smaller than pop, but stats methods are based on culled_pop; we may want initial stats
        self._culled_pop = self._pop[:]

        # These four properties are set by the two methods that immediately follow.
        self.uni_p = []
        self.bi_p = []
        self.uni_ent = []
        self.bi_ent = []

        self._set_probabilities()
        self._set_entropies()

        self._pop_and_cost.sort(key=lambda x: x[1])

        self.var_order = list(range(self._n))

        # These two properties will be set at the outset of training.
        self.cutoff_proportion = 0.0
        self._cutoff_cost = 0

    def train(self, cutoff_proportion=0.6, generations=50, verbose=False):
        """
        Method to train a mimic distribution object for a certain number of generations, each time culling every
        sample that has a higher cost than a population percentile determined by cutoff_proportion. A cutoff proportion
        of 0.5 would take any samples with cost up to and including the median value (possibly more than half the
        population size). A cutoff_proportion of 0.1 would only take the best 10%, plus additional samples that tied
        the 10th percentile value.
        Verbose mode prints updates with the total entropy (shouldn't get too low!) and population samples.
        """

        self.cutoff_proportion = cutoff_proportion
        self._cutoff_cost = self._pop_and_cost[int(self.pop_size * self.cutoff_proportion)][1]

        if verbose:
            print("Training Mimic distribution for {} generations with population size {} and cutoff {}".format(
                generations, self.pop_size, self.cutoff_proportion
            ))
            low = self._pop_and_cost[0][1]
            med = self._pop_and_cost[self.pop_size // 2][1]
            high = self._pop_and_cost[-1][1]
            print("Initial pop. \t Lowest, median, highest costs: {:2d} {:2d} {:2d}".format(low, med, high))

            total_bayes_ent = sum(self._puzzle.entropy)
            total_uniform_ent = -self._n * log(0.1)

            print("\t\t\t\t Total univariate entropy: {:0.2f} (compared to {:0.2f} for Bayesian dist. and "
                  "{:0.2f} for uniform dist.)".format(sum(self.uni_ent), total_bayes_ent, total_uniform_ent))

        for g in range(generations):

            self._culled_pop = [p for p, c in self._pop_and_cost if c <= self._cutoff_cost]

            self._set_probabilities()
            self._set_entropies()
            self._set_variable_order()

            self._pop = [self.bivariate_sample(self.var_order) for _ in range(self.pop_size)]
            self._pop_and_cost = [(p, self._puzzle.cost(p)) for p in self._pop]
            self._pop_and_cost.sort(key=lambda x: x[1])

            low = self._pop_and_cost[0][1]
            med = self._pop_and_cost[int(self.pop_size / 2)][1]
            high = self._pop_and_cost[-1][1]
            self._cutoff_cost = self._pop_and_cost[int(self.pop_size * self.cutoff_proportion)][1]
            if verbose and g % 10 == 0:
                print("Generation {} \t Lowest, median, highest costs: {:2d} {:2d} {:2d}".format(g, low, med, high))
                print("\t\t\t\t Total univariate entropy: {:0.2f}".format(sum(self.uni_ent)))
        if verbose:
            print("Mimic training complete.")
            print("Final univariate entropy: {:0.2f} (compared to {:0.2f} for Bayesian dist. and {:0.2f} for "
                  "uniform dist.)".format(sum(self.uni_ent), total_bayes_ent, total_uniform_ent))

    def univariate_sample(self, eps=0.01):
        """
        Creates a single new string using the univariate probabilities. This isn't meant to be used except to
        demonstrate that univariate sampling won't produce strings that are as good. eps is some noise that's
        added to the probabilities, so even digits with probability zero have a chance of being selected.
        """
        samp = ''
        for digit_dist in self.uni_p:
            samp += weighted_choice('0123456789', [d_p + eps for d_p in digit_dist])
        return samp

    def bivariate_sample(self, var_order=None, eps=0.01):
        """
        Creates a single new string by taking one univariate probability sample to get one digit, and then taking a
        sequence of bivariate probabilities to get subsequent digits. For example, it might pick digit 4 based on the
        univariate distribution for digit 4, and then pick digit 7 based on the bivariate distribution for digit 7,
        given the particular value of digit 4 that was picked, and so on.
        Digit order is determined by the parameter variable order, which defaults to the var_order object property.
        """
        if var_order is None:
            var_order = self.var_order

        digit_list = ['_'] * self._n

        # First digit
        v0 = var_order[0]
        digit_list[v0] = weighted_choice('0123456789', [d_prob + eps for d_prob in self.uni_p[v0]])

        # Subsequent digits
        for i in range(1, len(var_order)):
            last_v = var_order[i - 1]
            last_d = int(digit_list[last_v])
            v = var_order[i]
            digit_list[v] = weighted_choice('0123456789', [d_p + eps for d_p in self.bi_p[last_v][last_d][v]])

        return ''.join(digit_list)

    def random_variable_order(self):
        """
        Returns a random variable order of integers in range(0, n). This isn't meant to be used except to demonstrate
        that a random variable order is less effective than the calculated variable order.
        """
        x = list(range(self._n))
        shuffle(x)
        return x

    def _set_probabilities(self):
        """
        Updates self.uni_p and self.bi_p to be the univariate and bivariate probability distributions for each place
        and pair of place in self._culled_pop.
        self.uni_p[a][b] is probability that digit in place a has value b.
        self.bi__p[a][b][a2][b2] is probability that digit in place a2 has value b2, given that digit a has value b.
        The bivariate probability distribution has two particularities: bivariate_p[x][y][x][y] is 0.0 rather than 1.0,
        but that shouldn't matter, and the bivariate probability for a prior condition that never occurs is 0.0.
        """
        n = self._n

        eps = 1e-12     # To avoid getting problems with dividing by zero

        univariate_counter = [[0] * 10 for _ in range(n)]
        bivariate_counter = [[[[0] * 10 for _ in range(n)] for _ in range(10)] for _ in range(n)]

        for p in self._culled_pop:
            for a, b in enumerate(p):
                univariate_counter[a][int(b)] += 1
                for a2 in range(a + 1, n):
                    b2 = p[a2]
                    bivariate_counter[a][int(b)][a2][int(b2)] += 1
                    bivariate_counter[a2][int(b2)][a][int(b)] += 1

        self.uni_p = [[univariate_counter[a][b] / len(self._culled_pop) for b in range(10)] for a in range(n)]
        self.bi_p = [[[[bivariate_counter[a][b][a2][b2] / (univariate_counter[a][b] + eps)
                        for b2 in range(10)]
                       for a2 in range(n)]
                      for b in range(10)]
                     for a in range(n)]

    def _set_entropies(self):
        """
        Updates self.uni_ent and self.bi_ent to the new univariate and bivariate entropy values for self._culled_pop.
        self.uni_ent[3] is the entropy for place 3. 0.0 means certainty, and 2.3 means a totally uniform distribution.
        self.bi_ent[4][2] is the entropy for place 2, once you know the value of place 4. Varies between 0.0 and 2.3.
        """
        n = self._n

        self.uni_ent = [sum(self.neg_x_log_x(p) for p in a) for a in self.uni_p]

        bi_ent = [[0] * n for _ in range(n)]
        for a in range(n):
            for a2 in range(n):
                if a == a2:
                    continue
                bi_ent[a][a2] = sum(self.neg_x_log_x(self.bi_p[a][b][a2][b2]) for b in range(10) for b2 in range(10))
        self.bi_ent = bi_ent

    def _set_variable_order(self):
        """
        Updates self.var_order to give the most promising order for the given sampling statistics. Starts by picking
        the lowest univariate entropy place, then picks the lowest bivariate entropy place that could follow that, and
        so on. This is a faster (and theoretically nearly as good) alternative to a more thorough search for the
        minimum entropy path from start to finish.
        """
        n = self._n
        var_set = set(range(n))
        v = min(var_set, key=lambda x: self.uni_ent[x])
        order = [v]
        var_set.remove(v)
        for _ in range(n - 1):
            v_last = v
            v = min(var_set, key=lambda x: self.bi_ent[v_last][x])
            order.append(v)
            var_set.remove(v)
        self.var_order = order

    @staticmethod
    def neg_x_log_x(p):
        """
        Helper function that returns p * log(p), dealing correctly with the limiting case where p = 0.
        """
        if p == 0.0:
            return 0.0
        return -p * log(p)