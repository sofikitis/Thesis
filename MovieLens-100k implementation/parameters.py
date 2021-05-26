
# Learning rates from paper
UTILITY_FORMULATOR_LEARNING_RATE = 0.1
RETRIEVAL_FORMULATOR_LEARNING_RATE = 0.1

# First filtering with K2 for H1 or H2
K2 = 200

# K is the relevant users found from every filtering
K = 100

# N for evaluation metrics @N
N = [10, 30]

# set to 0 if you want 0s to be included in the evaluation 1 otherwise
DELETE_ZEROS = 1

# the currently running module
MODULE = 0

# Number of iterations for game and convergence functions
ITERATIONS = 50
DF_ITERATIONS = 25


# set to 0 if you want to skip convergence and 1 otherwise
FIND_CONVERGENCE = 0

# convergence for  cf and cb (10^-4)
CONV_CF = 0.0001
CONV_DF = 0.0001
