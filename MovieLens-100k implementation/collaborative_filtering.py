import numpy as np
import math
from operator import itemgetter
import parameters as p

# Parameters imported from parameters file
UTILITY_FORMULATOR_LEARNING_RATE = p.UTILITY_FORMULATOR_LEARNING_RATE
RETRIEVAL_FORMULATOR_LEARNING_RATE = p.RETRIEVAL_FORMULATOR_LEARNING_RATE
ITERATIONS = p.ITERATIONS
K = p.K
K2 = p.K2


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0


# Function to check if a profile exist in an array of profiles
def array_in_list(arr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, arr)), False)


# Function to calculate the relevance score of a user given a target user
def relevance_function(target_user, user, retrieval_parameter):

    omega = np.dot(np.multiply(target_user, retrieval_parameter), user)

    score = sigmoid(omega)

    return score


# Function to calculate utility
def calculate_utility(target_user, topk_users, all_users, retrieval_parameter):
    in_topk_sum = 0
    not_in_topk_sum = 0

    for cur_user in all_users:

        term = relevance_function(target_user, cur_user, retrieval_parameter)

        if array_in_list(cur_user, topk_users):
            if term != 0:
                in_topk_sum += math.log(term)

        else:
            if term != 0:
                not_in_topk_sum += math.log(term)

    # calculate utility
    utility = (1 / K) * in_topk_sum - (1 / (len(all_users) - K)) * not_in_topk_sum

    return utility


# Function to calculate the relevance of all user profiles given the target user
def calculate_relevance_scores(target_user, all_users, retrieval_parameter):
    scores = []
    i = 0

    for cur_user in all_users:
        cur_score = relevance_function(target_user, cur_user, retrieval_parameter)
        scores.append([cur_score, i])
        i += 1

    # Return sorted results to find top-k easily
    scores = sorted(scores, key=itemgetter(0), reverse=True)
    return scores


# Function to find the topK users. Returns the user object and a list with the users ids and scores
def find_topk_users(target_user, all_users, retrieval_parameter):
    scores = calculate_relevance_scores(target_user, all_users, retrieval_parameter)

    topk_users_id = scores[0:K]
    topk_users = []

    # Create a list with the topk (relevant) users
    for cur_id in topk_users_id:
        topk_users.append(all_users[cur_id[1], :])

    return [topk_users_id, topk_users]


# Function to update query
def query_reformulation(target_user, topk_users, all_users):
    in_topk_sum = 0
    not_in_topk_sum = 0

    for cur_user in all_users:

        cur_theta = sigmoid(np.inner(target_user, cur_user))

        if array_in_list(cur_user, topk_users):

            term = (1 - cur_theta) * cur_user
            in_topk_sum = in_topk_sum + term

        else:
            term = cur_theta * cur_user
            not_in_topk_sum = not_in_topk_sum + term

    # calculate gradient ascent
    gradient_ascent = (1 / K) * in_topk_sum - (1 / (len(all_users) - K)) * not_in_topk_sum

    # calculate the rate the query will be updated
    rate = UTILITY_FORMULATOR_LEARNING_RATE * gradient_ascent

    # update only the values in query that have ratings
    # target_user = [target_user[i] + rate[i] if target_user[i] != 0 else 0 for i in range(0, len(target_user))]

    target_user = [target_user[i] + rate[i] for i in range(len(target_user))]

    return target_user


# Function to update retrieval model
def retrieval_model_reformulation(target_user, relevant_users, all_users, retrieval_parameter):
    relevant_sum = 0
    non_relevant_sum = 0

    for cur_user in all_users:

        cur_score = relevance_function(target_user, cur_user, retrieval_parameter)

        if array_in_list(cur_user, relevant_users):
            term = (1 - cur_score) * (target_user * cur_user)
            relevant_sum = relevant_sum + term

        else:
            term = cur_score * (target_user * cur_user)
            non_relevant_sum = non_relevant_sum + term

    gradient_ascent = (1 / K) * relevant_sum - (1 / (len(all_users) - K)) * non_relevant_sum
    return retrieval_parameter + RETRIEVAL_FORMULATOR_LEARNING_RATE * gradient_ascent


# Query Reformulation convergence function
def convergenceQ(target_user, all_users, retrieval_parameter):
    old_util = 0

    for i in range(p.ITERATIONS):
        print(i, end=" ")

        # Find the topK (relevant) users
        topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)

        # Update Query
        target_user = query_reformulation(target_user, topk_users, all_users)

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_user, topk_users, all_users, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)
    print()
    return topk_users


def convergenceM(target_user, all_users, retrieval_parameter):
    old_util = 0

    # Find the topK (relevant) users
    topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)
    for i in range(p.ITERATIONS):
        print(i, end=" ")

        # Update retrieval parameter
        retrieval_parameter = retrieval_model_reformulation(target_user, topk_users, all_users, retrieval_parameter)

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_user, topk_users, all_users, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)

    print()
    return topk_users


# in this function both query and retrieval model are updated
def game(target_user, all_users, retrieval_parameter):

    global K
    # if module is 1 (Hybrid1) set K to K2
    if p.MODULE == 1:
        K = p.K2
    else:
        K = p.K

    old_util = 0

    # Find the topK (relevant) users
    topk_users_id, relevant_users = find_topk_users(target_user, all_users, retrieval_parameter)
    for i in range(p.ITERATIONS):
        print(i, end=" ")

        # Find the topK (relevant) users
        topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)

        # Update Query
        target_user = query_reformulation(target_user, topk_users, all_users)

        # Update retrieval parameter
        retrieval_parameter = retrieval_model_reformulation(target_user, relevant_users, all_users, retrieval_parameter)

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_user, topk_users, all_users, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    # Find the final topK (relevant) users
    topk_users_id, topk_users = find_topk_users(target_user, all_users, retrieval_parameter)

    print()
    return topk_users


def run_collaborative_filtering(target_user, profiles_matrix):

    global K
    # if module is 1 (Hybrid1) set K to TOP_K
    if p.MODULE == 1:
        K = p.K2
    else:
        K = p.K

    # Set retrieval parameter
    n = len(target_user)
    retrieval_parameter = np.ones(n)

    # ConvQ
    topk_users_ConvQ = convergenceQ(target_user=target_user, all_users=profiles_matrix,
                                    retrieval_parameter=retrieval_parameter)

    # ConvM
    topk_users_ConvM = convergenceM(target_user=target_user, all_users=profiles_matrix,
                                    retrieval_parameter=retrieval_parameter)

    # Initialize game
    topk_users_game = game(target_user=target_user, all_users=profiles_matrix,
                           retrieval_parameter=retrieval_parameter)

    return [topk_users_ConvQ, topk_users_ConvM, topk_users_game]
