import numpy as np
import math
from operator import itemgetter
import parameters as p


# Parameters imported from parameters file
UTILITY_FORMULATOR_LEARNING_RATE = p.UTILITY_FORMULATOR_LEARNING_RATE
RETRIEVAL_FORMULATOR_LEARNING_RATE = p.RETRIEVAL_FORMULATOR_LEARNING_RATE
ITERATIONS = p.ITERATIONS
K = p.K


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

    flag = [pr for pr in list_arrays if pr.ide == arr.ide]

    if flag:
        return True
    else:
        return False


# Function to calculate the relevance score of a user given a target user
def relevance_function(target_user, user, retrieval_parameter):

    # Find the movies they both have rated
    common_rated_movies = np.intersect1d(target_user.movies, user.movies)

    # Find the position of the common rated movies
    # and multiply the ratings for the common movies

    omega = 0
    for movie in common_rated_movies:
        target_users_rating = np.where(target_user.movies == movie)[0]
        c_users_rating = np.where(user.movies == movie)[0]

        position_t = target_users_rating[0]
        position_c = c_users_rating[0]

        rating_t = target_user.ratings[position_t]
        rating_c = user.ratings[position_c]

        omega = omega + rating_t*rating_c*retrieval_parameter[position_t]

    # omega = np.dot(np.multiply(target_user, retrieval_parameter), user)

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

    for cur_user in all_users:

        cur_user_id = cur_user.ide

        cur_score = relevance_function(target_user, cur_user, retrieval_parameter)
        scores.append([cur_score, cur_user_id])

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
        profile = [pr for pr in all_users if pr.ide == cur_id[1]][0]
        topk_users.append(profile)

    return [topk_users_id, topk_users]


# Function to find the most relevant users to use for the collaborative filtering
def find_relevant_users(target_user, all_users, retrieval_parameter, n_of_users):
    scores = calculate_relevance_scores(target_user, all_users, retrieval_parameter)
    relevant_users_id = scores[0:n_of_users]
    relevant_users = []

    # Create a list with the topk (relevant) users
    for cur_id in relevant_users_id:
        profile = [pr for pr in all_users if pr.ide == cur_id[1]][0]
        relevant_users.append(profile)

    return relevant_users


# Function to update query
def query_reformulation(target_user, topk_users, all_users):
    in_topk_sum = np.zeros(len(target_user.ratings))
    not_in_topk_sum = np.zeros(len(target_user.ratings))

    for cur_user in all_users:

        # create an array to use for query reformulation
        # the array will be zeros except the movies that the current user has rated
        reformulation_array = np.zeros(len(target_user.ratings))
        common_rated_movies = np.intersect1d(target_user.movies, cur_user.movies)

        for movie in common_rated_movies:
            target_users_rating = np.where(target_user.movies == movie)[0]
            c_users_rating = np.where(cur_user.movies == movie)[0]

            position_t = target_users_rating[0]
            position_c = c_users_rating[0]
            rating_c = cur_user.ratings[position_c]

            reformulation_array[position_t] = rating_c

        # for theta i use the relevance function with retrieval_parameter = 1
        # cur_theta = sigmoid(np.inner(target_user, cur_user))
        retrieval_parameter = np.ones(len(target_user.ratings))
        cur_theta = relevance_function(target_user, cur_user, retrieval_parameter)

        if array_in_list(cur_user, topk_users):

            term = np.multiply((1 - cur_theta), reformulation_array)
            in_topk_sum = in_topk_sum + term

        else:
            term = np.multiply(cur_theta, reformulation_array)
            not_in_topk_sum = not_in_topk_sum + term

    # calculate gradient ascent
    gradient_ascent = (1 / K) * in_topk_sum - (1 / (len(all_users) - K)) * not_in_topk_sum

    # calculate the rate the query will be updated
    rate = UTILITY_FORMULATOR_LEARNING_RATE * gradient_ascent

    # update only the values in query that have ratings
    # target_user = [target_user[i] + rate[i] if target_user[i] != 0 else 0 for i in range(0, len(target_user))]

    target_user.ratings = [target_user.ratings[i] + rate[i] for i in range(len(target_user.ratings))]

    return target_user


# Function to update retrieval model
def retrieval_model_reformulation(target_user, relevant_users, all_users, retrieval_parameter):
    relevant_sum = np.zeros(len(target_user.ratings))
    non_relevant_sum = np.zeros(len(target_user.ratings))

    for cur_user in all_users:

        # create an array to use for retrieval reformulation
        # the array will be zeros except the movies that the current user has rated
        reformulation_array = np.zeros(len(target_user.ratings))
        common_rated_movies = np.intersect1d(target_user.movies, cur_user.movies)

        for movie in common_rated_movies:
            target_users_rating = np.where(target_user.movies == movie)[0]
            c_users_rating = np.where(cur_user.movies == movie)[0]

            position_t = target_users_rating[0]
            position_c = c_users_rating[0]
            rating_t = target_user.ratings[position_t]
            rating_c = cur_user.ratings[position_c]

            reformulation_array[position_t] = rating_c*rating_t

        cur_score = relevance_function(target_user, cur_user, retrieval_parameter)

        if array_in_list(cur_user, relevant_users):
            term = np.multiply((1 - cur_score), reformulation_array)
            relevant_sum = relevant_sum + term

        else:
            term = np.multiply(cur_score, reformulation_array)
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
    # if module is 1 (Hybrid1) set K to K2
    if p.MODULE == 1:
        K = p.K2
    else:
        K = p.K

    # Set retrieval parameter
    retrieval_parameter = np.ones(len(target_user.ratings))

    # Use the top 1.000 relevant for the collaborative filtering to save time
    profiles_matrix = find_relevant_users(target_user, profiles_matrix, retrieval_parameter, 1000)

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
