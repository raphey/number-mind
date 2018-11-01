## Solving a Number Mind puzzle ten different ways

### Overview
This is a collection of algorithms I wrote to solve “Number Mind” puzzles, a variation of the board game [MasterMind](https://en.wikipedia.org/wiki/Mastermind_(board_game)) described in [Project Euler problem 185](https://projecteuler.net/problem=185). The basic idea is to deduce an unknown string of digits from clues in the form of a series of premade “guesses” along with the number of correct digits in the guesses. 

I wrote these because this problem seemed to have a particularly diverse set of available approaches, and I thought it would be fun to play with some new algorithms and optimizations. Of course if you want to solve the problem yourself, you should do that before continuing,<sup>1</sup> and if you have another method or improvement to suggest, I’d love to hear about it. Thanks very much to [Chris Gearhart](https://github.com/cgearhart) for helping with these algorithms and telling me about MIMIC and LAHC.

### Algorithms and shorthand names
I solved the problem using:
- repeated greedy search (RGS)
- simulated annealing (SA)
- a genetic algorithm (GA)
- two types of backtracking search, based on guesses and digits (BGS and BDS)
- a heap search that repeatedly expanded the lowest-cost node yet encountered (Heap)
- repeated greedy search using a bivariate probability distribution model to generate low-cost starting points (RGS w MIMIC)
- a generalized constraint library for python (Constraint)
- a fast C-based mixed-integer programming library that’s part of Google’s OR tools, which I also used to create a function to generate new puzzles (MIP)
- late acceptance hill climbing (LAHC)
- Bayesian probability distributions that substantially enhanced the performance of many of the above algorithms

### Results

[Here’s some sample output](sample_output.txt), and here’s how the algorithms performed:

| Algorithm    | Avg. solve (s)| Avg. # cost evals |
| ------------ | ------------- | ----------------- |
| BGS w Bayes  | 1.0           | -                 |
| MIP          | 1.3           | -                 |
| SA w Bayes   | 1.9           | 5.1e5             |
| SA           | 3.7           | 1.4e6             |
| LAHC w Bayes | 5.8           | 1.5e6             |
| Heap w Bayes | 6.1           | 6.3e5             |
| Heap         | 10            | 9.5e5             |
| LAHC         | 39            | 8.2e6             |
| GA w Bayes   | 41            | 1.6e6             |
| GA           | 51            | 2.1e6             |
| BGS          | 54            | -                 |
| RGS w Bayes  | 75            | 4.8e7             |
| RGS w MIMIC  | 78            | 3.1e7             |
| RGS          | 270           | 1.7e8             |
| BDS w Bayes  | 960           | -                 |
| BDS          | ≈1e6          | -                 |
| Constraint   | ≈1e10         | -                 |

The dashes in the last column are for functions that didn’t use cost evaluations. The last two numbers are estimates I made by either letting the algorithm chug along and monitoring progress, or extrapolating from smaller cases.

Backtracking guess-wise search with Bayesian probability was the winner, but after the race, Mixed-Integer Programming refused to shake hands, muttering about parameter tuning. Fair enough—the Bayesian modifications to BGS and BDS both used probabilistic cutoffs to narrow the search space. To keep things mostly fair, I found the most aggressive cutoffs that would find a solution and changed them by a factor of two, figuring that was conservative enough for generalization. I tried a bonus round with three new generated puzzles of length 16, and while RGS w Bayes beat MIP in a similar fashion on two of them, it failed to get a solution for the 3rd, while MIP managed to get a solution in 0.067s. 

I also ran an exhibition match between BGS w Bayes and MIP, trying a much harder problem with 20 digits. MIP won that by a huge margin, 28s to 1800s, which is generating a lot of excitement for next season.

RGS w MIMIC deserves honorable mention for nearly beating RGS w Bayes while using far fewer cost evaluations.

For all of these, I used a 2.9 GHz processor, and I’m using PyPy for all algorithms except for the mixed-integer programming, since Google’s OR tools won’t work with PyPy. This probably didn’t impact MIP’s solve time, since nearly all of what it’s doing is in C anyway.

### File descriptions

- [number_mind_solvers.py](number_mind_solvers.py): Collection of solving algorithms
- [number_mind.py](number_mind.py): The main Number Mind puzzle class
- [number_mind_specific_puzzles.py](number_mind_specific_puzzles.py): A handful of Number Mind puzzles, including the standard puzzle, an unsolvable one, and some generated longer ones
- [sample_output.txt](sample_output.txt): Output from solving the standard puzzle with SA, GA, RGS, and MIMIC
- [mimic_distribution.py](mimic_distribution.py): A class for training a MIMIC distribution
- [integer_programming_solver.py](integer_programming_solver.py): The MIP solver, using Google’s OR tools
- [number_mind_puzzle_generator.py](number_mind_puzzle_generator.py): Generates new Number Mind puzzles using the MIP solver
- [util.py](util.py): Some tools I used for timing and comparison
- [number_mind_bayesian_probabilities.pdf](number_mind_bayesian_probabilities.pdf): Derivation of the probabilities I used with some algorithms
_________________________________________

### Additional information

#### Cost function
The algorithms that used a cost function all used the same one. For a given state, the cost function was the sum of how “wrong” the guess scores were, when the guesses were compared to state in question, with the wrongness being the absolute value of the difference in how many were correct vs how many were supposed to be correct. The optimal way to calculate cost varied by algorithm, since algorithms with states that were changing one digit at a time allowed for some shortcuts.

#### Bayesian probabilities
I calculated the Bayesian probability of the solution having a particular digit in a particular place, given the distribution of guess scores with that digit either occurring or not occurring in that place. [Here’s my derivation](number_mind_bayesian_probabilities.pdf). These probabilities were able to make many of the algorithms substantially faster.

### Repeated greedy search
This was able to solve the original puzzle in a matter of minutes, and it may have won in terms of coding time + solve time. I also implemented the ability to change multiple digits, but the additional score lookups canceled the benefits of the extra reach. 

#### Simulated annealing
Magic! As long as you get the parameters tuned just right. It’s amazing how consistently this is able to find the solution, and it does so in seconds, but to use this on an unfamiliar problem would require writing some extra code to vary the parameters.

#### Genetic algorithm
Also very cool. I played with this for a while without success, and eventually I found that the only approach that worked consistently was to start the entire process over many times. Most of my initial populations went nowhere, but one out of every ten or so was able to produce a solution within 100 generations. This was also the algorithm where the use of Bayesian probability didn’t help much, presumably because the increased homogeneity of the initial populations partly canceled the lower average cost.

#### Digit-wise backtracking search
The idea here was to start filling digits in left to right with 0s, then stop and backtrack when a required score for a guess was no longer possible, incrementing the last successfully assigned 0 to a 1, and so forth. Unfortunately, it’s possible to assign a large number of digits (10-13) before any backtracking is necessary, which made this method unfeasible. When I used the calculated probabilities to reorder the digits within each place and reorder the places (most to least certain of top choice), it was possible to get a solution in under an hour. This could be cut down to half an hour with an extra type of pruning that abandoned paths once they became too improbable relative to a state using the median probable digit for each place. This required parameter tuning, and it ran the risk of missing the solution.

#### Guess-wise backtracking search
This inverted things relative to the digit-wise search. If the first guess had a score of 2 and the second had a score of 1, this method first asked, which 2 were correct in the first guess, and then assuming it was one particular pair, which 1 was correct for the second guess, and so on. This allowed for much more drastic pruning, particularly when the guesses were ordered by increasing score, since the 0 right and 1 right guesses branch less than the 2 right and 3 right guesses. This got the answer in under a minute, and once probabilities were added in, the time cut down to 10 seconds. If some probabilistic cutoffs were added for each guess (for example, dropping possibilities that were less than half as likely as the average possibility), it could get the correct answer in less than a second.

#### Heap search
This wasn’t my idea—it came from a PE forum post by user Sandorf, but it was so interesting that I wanted to try it myself. The idea is to keep a heap/priority queue of states explored so far, continually popping the lowest-cost state off the heap and re-adding every unexplored one-digit variation of that state. It gets an answer in around 10 seconds. I was able to chop 30% off the time by modifying the scoring to take advantage of the fact that each new state was a one-digit variation of an existing state, but this came at a cost of more memory to store a dictionary of cost-related data for each explored state.

#### Repeated greedy search with the MIMIC algorithm
This was my favorite approach, based on the algorithm described in [this 1997 paper](https://www.cc.gatech.edu/~isbell/papers/isbell-mimic-nips-1997.pdf) by De Bonet, Isbell, and Viola. MIMIC is a method by which progressively lower-cost states can be generated by repeatedly creating populations, culling down to some subset of lower-cost members, looking at the probability distributions of the remaining members, and using those to create new populations. The bivariate aspect is very cool—it’s not just concerned with the frequency of digit 5 in place 3, it wants to know the frequency of digit 5 in place 3, given that the digit in place 1 is a 6. This lets it model multivariate spaces that couldn’t be captured with univariate distributions. The entropy minimization method for selecting the order in which you pick variables when generating samples was also really interesting. 

My implementation of MIMIC took about 30 seconds to train, achieving median costs of 20, vs 29 for random data and 22 for univariate Bayesian sampling. I could have used the MIMIC samples with a number of different other algorithms, but I ended up only using it with repeated greedy search. I was able to see conclusively that the MIMIC-generated samples achieved better results than the random and Bayesian samples. MIMIC samples arrived at solutions about as fast as Bayesian samples, even with 40% of the time spent training rather than on greedy search, and the MIMIC samples required only 2/3 as many cost function calls, including those used for training. MIMIC samples substantially outperformed random sampling both in terms of time and cost function calls.

#### Generalized constraint solver
I thought it would be cool to try plugging the problem into a pre-made tool, and I found [one by Gustavo Niemeyer written in pure python](https://labix.org/python-constraint). It works for the n=5 small case, but it seems like it might take years to solve the full problem. This tool affords the luxury of completely general variable constraints (as opposed to only allowing linear equation constraints), but it comes at a steep cost.

#### Mixed-integer programming
Google’s OR tools include a python-wrapped tool that does mixed-integer programming. The problem can be reframed as a set of 10*n binary variables, where v35=1 might, for instance, mean that the digit in place 3 is a 5. (I got this idea from the PE forum.) It was extremely satisfying to see this use “branch-and-cut” to get a conclusive, non-random answer in only a second, and this tool also allowed me to create new puzzles, since it could determine whether a given puzzle had a unique solution.

#### Late acceptance hill-climbing
As described in [this 2012 paper](https://pdfs.semanticscholar.org/28cf/19a305a3b02ba2f34497bebace9f74340ade.pdf) by Burke and Bykov, LAHC is a hill-climbing variation in which a random iteration is admitted either if it’s an improvement relative to where we currently are, or if it’s an improvement to where we were L iterations earlier, where L is a parameter. If it’s rejected, we stay in the same place for one iteration and try again. It has one fewer parameter to tune than SA, and a comparable amount of magic.

_________________________________________

<sup>1</sup>Project Euler strongly discourages participants from posting solutions online and spoiling others’ fun, but I felt this post was in keeping with the spirit of learning/exploration. Plus the problem is nine years old.
