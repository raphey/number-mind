###Overview
This is a collection of algorithms I wrote to solve “Number Mind” puzzles, as introduced in Project Euler problem 185. I wrote these because this problem seemed to have a particularly diverse set of available approaches, and I thought it would be fun to explore some new approaches and optimizations. You should probably read the problem description first, and of course if you want to solve the problem yourself, you should do that before continuing.* Thanks very much to Chris Gearheart for helping with these algorithms and telling me about MIMIC.

###Methods used
I solved the problem using:
- repeated hill descent
- simulated annealing
- a genetic algorithm
- two types of backtracking search
- a heap search that repeatedly expanded the lowest-cost node yet encountered
- repeated hill-descent using a bivariate probability distribution model (MIMIC) to generate low-cost starting points
- a generalized constraint library for python
- a fast C-based mixed-integer programming library that’s part of Google’s OR tools, which I also used to create a function to generate new puzzles
- Bayesian probability distributions that substantially enhanced the performance of many of the above algorithms

###Hardware
Since I cite some times below, it seems worth mentioning that I’m working with a 2.9 GHz processor, and I’m using PyPy for all algorithms except for the mixed-integer programming, since Google’s OR tools won’t work with PyPy.

###Cost function
Many of the above algorithms used the same cost function. For a given state, my cost function was the sum of how “wrong” the guess scores were, when the guesses were compared to state in question, with the wrongness being the absolute value of the difference in how many were correct vs how many were supposed to be correct. The optimal way to calculate cost varied by algorithm, since algorithms with states that were changing one digit at a time allowed for some time-saving.

###Bayesian probabilities
I calculated the Bayesian probability of the solution having a particular digit in a particular place, given the distribution of guess scores with that digit either occurring or not occurring in that place. [Here’s my derivation](docs/NumberMindBayesianProbabilities.pdf). These probabilities were able to make many of the algorithms I used substantially faster.

###Repeated hill-descent
This was able to solve the original puzzle in a matter of minutes. I also implemented the ability to change multiple digits, but the additional score lookups canceled the benefits of the extra reach. 

###Simulated annealing
Magic! As long as you get the parameters tuned just right. It’s amazing how consistently this is able to find the solution, and it does so in seconds, but to use this on an unfamiliar problem would require writing some extra code to vary the parameters.

###Genetic algorithm
Also very cool. I played with this for a while without success, and eventually I found that the only approach that worked consistently was to start the entire process over many times. Most of my initial populations went nowhere, but every 10 or so were able to produce a solution within 100 generations. This was also the only algorithm with a random element where the calculated probabilities help.

###Digit-wise backtracking search
The idea here was to start filling digits in left to right with 0s, then stop and backtrack when a required score for a guess was no longer possible, incrementing the last successfully assigned 0 to a 1, and so forth. Unfortunately, it’s possible to assign a large number of digits (10-13) before any backtracking is necessary, which made this method unfeasible. When I used the calculated probabilities to reorder the digits within each place and reorder the places (most to least certain of top choice), it was possible to get a solution in an hour. This could be cut down to half an hour with an extra type of pruning that abandoned paths once they became too improbable relative to a state using the median probable digit for each place. This required parameter tuning, and it ran the risk of missing the solution.

###Guess-wise backtracking search
This inverted things relative to the digit-wise search. If the first guess had a score of 2 and the second had a score of 1, this method first asked, which 2 were correct in guess 1, and then assuming it was one particular pair, which 1 was correct for guess 2, and so on. This allowed for much more drastic pruning, particularly when the guesses were ordered by increasing score, since the 0 right and 1 right guesses branch less than the 2 right and 3 right guesses. This got the answer in a minute, and once probabilities were added in, the time cut down to 10 seconds. If some probabilistic cutoffs were added for each guess (for example, dropping possibilities that were less than half as likely as the average possibility), it could get the correct answer in less than a second.

###Heap search
This wasn’t my idea—it came from a PE forum post by user Sandorf, but it was so interesting that I wanted to try it myself. The idea is to keep a heap/priority queue of states explored so far, continually popping the lowest-cost state off the heap and re-adding every unexplored one-digit variation of that state. It gets an answer in around 10 seconds. I was able to chop 30% off the time by modifying the scoring to take advantage of the fact that each new state was a one-digit variation of an existing state, but this came at a cost of more memory to store a dictionary of cost-related data for each explored state.

###Hill-descent with the MIMIC algorithm
This was my favorite approach. MIMIC is a method by which progressively lower-cost states can be generated by repeatedly creating populations, culling down to some subset of lower-cost members, looking at the probability distributions of the remaining members, and using those to create new populations. The bivariate aspect is very cool—it’s not just concerned with the frequency of digit 5 in place 3, it wants to know the frequency of digit 5 in place 3, given that the digit in place 1 is a 6. This lets it model multivariate spaces that couldn’t be captured with univariate distributions. The entropy minimization method for selecting the order in which you pick variables when generating samples was also really interesting. 

My implementation of MIMIC took about 30 seconds to train, achieving median costs of 20 vs 29 for random data and 22 for univariate Bayesian sampling. I could have used the MIMIC samples with a number of different other algorithms, but I ended up only using it with random hill-climbing. I was able to see conclusively that the MIMIC-generated samples achieved better results than the random and Bayesian samples.

###Generalized constraint solver
I thought it would be cool to try plugging the problem into a pre-made tool, and I found one by Gustavo Niemeyer written in pure python. It works for the n=5 small case, but it seems like it might take years to solve the full problem. This tool affords the luxury of completely general variable constraints (as opposed to only allowing linear equation constraints), but it comes at a steep cost.

###Mixed-integer programming
Google’s OR tools includes a python-wrapped tool that does mixed-integer programming. The problem can be reframed as a set of 10*n binary variables, where v35=1 might, for instance, mean that the digit in place 3 is a 5. (I got this idea from the PE forum.) It was extremely satisfying to see this use “branch-and-cut” to get a conclusive, non-random answer in only a second, and this tool also allowed me to create new puzzles, since it could determine whether a given puzzle had a unique solution.

*Project Euler strongly discourages participants from posting solutions online and spoiling others’ fun, but I felt this post was in keeping with the spirit of learning/exploration. Plus the problem is ten years old.