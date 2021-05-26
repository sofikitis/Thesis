
# Learning rates from paper
UTILITY_FORMULATOR_LEARNING_RATE = 0.1
RETRIEVAL_FORMULATOR_LEARNING_RATE = 0.1

# K for topK users
K = 100

# K2 for the top movies
K2 = 750

# N for evaluation metrics @N
N = [10, 30]

# set to 0 if you want 0s to be included in the evaluation 1 otherwise
DELETE_ZEROS = 0

# the currently running module
MODULE = 0

# Number of iterations for game and convergence functions
ITERATIONS = 50

# set to 0 if you want to skip convergence and 1 otherwise
FIND_CONVERGENCE = 0

# convergence for  cf and cb (10^-4)
CONV_CF = 0.0001
CONV_DF = 0.0001

# number of top rated movies from each user to use for hybrid1
HIGH_MOVIES = 10
