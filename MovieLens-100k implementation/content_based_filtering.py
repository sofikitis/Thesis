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
    return next((True for elem in list_arrays if np.array_equal(elem, arr)), False)


# Function to calculate the relevance score of a movie given a target movie
def relevance_function(target_movie, movie, retrieval_parameter):
    omega = np.dot(np.multiply(target_movie, retrieval_parameter), movie)

    score = sigmoid(omega)

    return score


# Function to calculate utility
def calculate_utility(target_movie, topk_movies, all_movies, retrieval_parameter):
    in_topk_sum = 0
    not_in_topk_sum = 0

    for cur_movie in all_movies:

        term = relevance_function(target_movie, cur_movie, retrieval_parameter)

        if array_in_list(cur_movie, topk_movies):
            if term != 0:
                in_topk_sum += math.log(term)

        else:
            if term != 0:
                not_in_topk_sum += math.log(term)

    # calculate utility
    utility = (1 / K) * in_topk_sum - (1 / (len(all_movies) - K)) * not_in_topk_sum

    return utility


# Function to calculate the relevance of all movie profiles given the target movie
def calculate_relevance_scores(target_movie, all_movies, retrieval_parameter):
    scores = []
    i = 0

    for cur_movie in all_movies:
        cur_score = relevance_function(target_movie, cur_movie, retrieval_parameter)
        scores.append([cur_score, i])
        i += 1

    # Return sorted results to find top-k easily
    scores = sorted(scores, key=itemgetter(0), reverse=True)
    return scores


# Function to find the topK movies. Returns the movie object and a list with the movies ids and scores
def find_topk_movies(target_movie, all_movies, retrieval_parameter):
    scores = calculate_relevance_scores(target_movie, all_movies, retrieval_parameter)

    topk_movies_id = scores[0:K]
    topk_movies = []

    # Create a list with the topk (relevant) movies
    for cur_id in topk_movies_id:
        topk_movies.append(all_movies[cur_id[1], :])

    return [topk_movies_id, topk_movies]


# Function to update query
def query_reformulation(target_movie, topk_movies, all_movies):
    in_topk_sum = 0
    not_in_topk_sum = 0

    for cur_movie in all_movies:

        cur_theta = sigmoid(np.inner(target_movie, cur_movie))

        if array_in_list(cur_movie, topk_movies):

            term = (1 - cur_theta) * cur_movie
            in_topk_sum = in_topk_sum + term

        else:
            term = cur_theta * cur_movie
            not_in_topk_sum = not_in_topk_sum + term

    # calculate gradient ascent
    gradient_ascent = (1 / K) * in_topk_sum - (1 / (len(all_movies) - K)) * not_in_topk_sum

    # calculate the rate the query will be updated
    rate = UTILITY_FORMULATOR_LEARNING_RATE * gradient_ascent

    # update only the values in query that have ratings
    # target_movie = [target_movie[i] + rate[i] if target_movie[i] != 0 else 0 for i in range(0, len(target_movie))]

    target_movie = [target_movie[i] + rate[i] for i in range(len(target_movie))]

    return target_movie


# Function to update retrieval model
def retrieval_model_reformulation(target_movie, relevant_movies, all_movies, retrieval_parameter):
    relevant_sum = 0
    non_relevant_sum = 0

    for cur_movie in all_movies:

        cur_score = relevance_function(target_movie, cur_movie, retrieval_parameter)

        if array_in_list(cur_movie, relevant_movies):
            term = (1 - cur_score) * (target_movie * cur_movie)
            relevant_sum = relevant_sum + term

        else:
            term = cur_score * (target_movie * cur_movie)
            non_relevant_sum = non_relevant_sum + term

    gradient_ascent = (1 / K) * relevant_sum - (1 / (len(all_movies) - K)) * non_relevant_sum
    return retrieval_parameter + RETRIEVAL_FORMULATOR_LEARNING_RATE * gradient_ascent


# Query Reformulation convergence function
def convergenceQ(target_movie, all_movies, retrieval_parameter):
    old_util = 0

    for i in range(p.ITERATIONS):
        print(i)

        # Find the topK (relevant) movies
        topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

        # Update Query
        target_movie = query_reformulation(target_movie, topk_movies, all_movies)

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_movie, topk_movies, all_movies, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

    return topk_movies_id


def convergenceM(target_movie, all_movies, retrieval_parameter):
    old_util = 0

    for i in range(p.ITERATIONS):
        print(50 + i)

        # Find the topK (relevant) movies
        topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

        # Update retrieval parameter
        retrieval_parameter = retrieval_model_reformulation(target_movie, topk_movies, all_movies, retrieval_parameter)

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_movie, topk_movies, all_movies, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

    return topk_movies_id


# in this function both query and retrieval model are updated
def game(target_movie, all_movies, retrieval_parameter):
    old_util = 0

    for i in range(p.ITERATIONS):
        print(100 + i)

        # Find the topK (relevant) movies
        topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

        # Update Query
        target_movie = query_reformulation(target_movie, topk_movies, all_movies)

        # Update retrieval parameter
        retrieval_parameter = retrieval_model_reformulation(target_movie, topk_movies, all_movies, retrieval_parameter)

        """new_util = calculate_utility(new_target_movie, topk_movies, all_movies, retrieval_parameter)
        if new_util > old_util:
            target_movie = new_target_movie
            old_util = new_util"""

        """new_util = calculate_utility(new_target_movie, topk_movies, all_movies, new_retrieval_parameter)
        if new_util > old_util:
            retrieval_parameter = new_retrieval_parameter
            old_util = new_util"""

        if p.FIND_CONVERGENCE == 1:
            # calculate the utility of the query reformulation
            new_util = calculate_utility(target_movie, topk_movies, all_movies, retrieval_parameter)

            # Rerun game if convergence_criteria is False
            criterionA = new_util - old_util < p.CONV_CF
            criterionB = new_util > old_util
            old_util = new_util
            convergence_criteria = (criterionA and criterionB)

            if convergence_criteria:
                break

    # Find the final topK (relevant) movies
    topk_movies_id, topk_movies = find_topk_movies(target_movie, all_movies, retrieval_parameter)

    return topk_movies_id


# Function to create a user-tag profile from user's ratings
def create_user_profile(user_rating, movie_profiles):
    number_of_ratings_by_genre = np.zeros((len(movie_profiles[0])))
    sum_of_ratings_by_genre = np.zeros((len(movie_profiles[0])))
    for i in range(len(user_rating) - 1):
        movie_profile = movie_profiles[i]

        for j in range(len(movie_profile)):
            number_of_ratings_by_genre[j] += movie_profile[j]
            sum_of_ratings_by_genre[j] += user_rating[i]

    user_profile = sum_of_ratings_by_genre / number_of_ratings_by_genre

    return user_profile


def run_collaborative_filtering(user_ratings_profile, profiles_matrix):
    # Set target user and retrieval parameter
    target_user = create_user_profile(user_ratings_profile, profiles_matrix)
    n = len(target_user)
    retrieval_parameter = np.zeros(n)

    # ConvQ
    topk_movies_ConvQ = convergenceQ(target_movie=target_user, all_movies=profiles_matrix,
                                     retrieval_parameter=retrieval_parameter)

    # ConvM
    topk_movies_ConvM = convergenceM(target_movie=target_user, all_movies=profiles_matrix,
                                     retrieval_parameter=retrieval_parameter)

    # Initialize game
    topk_movies_game = game(target_movie=target_user, all_movies=profiles_matrix,
                            retrieval_parameter=retrieval_parameter)

    return [topk_movies_ConvQ, topk_movies_ConvM, topk_movies_game]
