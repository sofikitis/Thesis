import numpy as np
import pandas as pd
import json
from numpy.core.defchararray import upper
import parameters as p
import math


class UserProfile:
    def __init__(self, ide, movies, ratings):
        self.ide = ide
        self.movies = movies
        self.ratings = [float(r) for r in ratings]


# Function to import the ratings to a profiles matrix
def import_ratings_profiles(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, sep='|', header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    user_profiles_list = []
    for user, movies, ratings in zip(profiles_matrix[:, 0], profiles_matrix[:, 1], profiles_matrix[:, 2]):
        movies = [int(float(i)) for i in movies.split(',')]
        ratings = [int(float(i)) for i in ratings.split(',')]

        profile = UserProfile(user, movies, ratings)

        user_profiles_list.append(profile)

    return np.array(user_profiles_list)


# Functions to import data and the results of filterings
def import_cb_results(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()
    return profiles_matrix.astype(np.float)


def import_cf_results(file):
    profile_list = []

    f = open(file, )
    data = json.load(f)
    for i in data:
        c_user = i["User"]
        profile = UserProfile(c_user["ID"], c_user["movies"], c_user["ratings"])
        profile_list.append(profile)
        f.close()

    return np.array(profile_list)


# Function to import data and the results of h3 and h4 filterings
def import_data(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


# function to calculate the average of a user's ratings
def calculate_average_rating(user):
    sum_of_ratings = 0
    number_of_ratings = 0
    for r in user:
        if r != 0:
            sum_of_ratings += r
            number_of_ratings += 1
    avg = sum_of_ratings / number_of_ratings

    return avg


# function to calculate ratings from collaborative filtering results
def rating_profiles_to_prediction(target_user, results):
    predicted_ratings = np.zeros(len(target_user.ratings))

    for m in range(0, len(target_user.movies)):
        movie = target_user.movies[m]

        n_of_ratings = 0
        sum_of_ratings = 0
        for pr in range(0, len(results)):

            if movie in results[pr].movies:
                pos = results[pr].movies.index(movie)
                n_of_ratings = n_of_ratings + 1
                sum_of_ratings = sum_of_ratings + results[pr].ratings[pos]

        predicted_ratings[m] = sum_of_ratings / n_of_ratings if n_of_ratings != 0 else 0
    return predicted_ratings


# function to calculate ratings from content based filtering results
def cb_results_to_prediction(target_user, results, alg):
    prediction = np.zeros(len(target_user.ratings))

    weight = 1
    if alg == 'cb' or alg == 'h1':
        weight = 5

    for m in range(0, len(target_user.movies)):
        movie = target_user.movies[m]

        pos = np.where(results[:, 1] == movie)[0]

        if pos.size > 0:
            prediction[m] = results[pos, 0] * weight

    for p in range(len(prediction)):
        if math.isnan(prediction[p]):
            prediction[p] = 0

    return prediction


"""Functions to calculate metrics"""


def dcg_at_k(y_true, y_score, n):
    if len(y_true) < n:
        n = len(y_true)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:n])

    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(2, gains.size + 2))

    dcg = np.sum(gains / discounts)
    return dcg


# Function to calculate ndcg @N results
def calculate_ndcg(arrayA, arrayB, n):
    actual = dcg_at_k(arrayA, arrayB, n)
    best = dcg_at_k(arrayA, arrayA, n)
    ndcg = actual / best
    return ndcg


# Function to calculate precision @N results
def calculate_precision(arrayA, arrayB, n):
    # Sort the real ratings and keep the topN results
    real_order = np.argsort(arrayA)[::-1][:n]
    predicted_order = np.argsort(arrayB)[::-1][:n]
    common_relevant = np.intersect1d(real_order, predicted_order)

    precision = len(common_relevant) / n
    return precision


# Function to calculate Average Precision
def calculate_ap(arrayA, arrayB, n):
    ap = 0
    for i in range(1, n + 1):
        ap += calculate_precision(arrayA, arrayB, i)

    return ap / n


# Function to calculate the Reciprocal Rank
def calculate_rr(arrayA, arrayB):
    real_order = np.argsort(arrayA)[::-1]
    predicted_order = np.argsort(arrayB)[::-1]

    rank = 0
    for i in range(len(arrayA)):
        rank += 1
        if predicted_order[i] in real_order[:i]:
            break

    return 1 / rank


# Function to calculate all the metrics
def calculate_metrics(real_profile, predicted_profile):
    real_profile = np.array(real_profile)
    predicted_profile = np.array(predicted_profile)

    ndcg0 = calculate_ndcg(real_profile, predicted_profile, p.N[0])
    ndcg1 = calculate_ndcg(real_profile, predicted_profile, p.N[1])

    precision0 = calculate_precision(real_profile, predicted_profile, p.N[0])
    precision1 = calculate_precision(real_profile, predicted_profile, p.N[1])

    ap = calculate_ap(real_profile, predicted_profile, p.N[0])
    rr = calculate_rr(real_profile, predicted_profile)

    metrics = [ndcg0, ndcg1, precision0, precision1, ap, rr]

    return metrics


def get_metrics_from_results(file_dir, target_user, algorithm):
    global NOT_EVAL
    NOT_EVAL = 0

    real_ratings = target_user.ratings

    if algorithm == 'cf' or algorithm == 'h2':
        results = import_cf_results(file_dir)
        predictions = rating_profiles_to_prediction(target_user, results)

    elif algorithm == 'cb' or algorithm == 'h1':
        results = import_cb_results(file_dir)
        predictions = cb_results_to_prediction(target_user, results, 'cb')

        zeros = np.where(predictions == 0)[0]
        real_ratings = np.delete(real_ratings, zeros)
        predictions = np.delete(predictions, zeros)

    else:
        results = import_cb_results(file_dir)
        predictions = cb_results_to_prediction(target_user, results, 'h3')

    if predictions.size > 0:
        metrics = calculate_metrics(real_ratings, predictions)
    else:
        metrics = np.zeros(6)
        NOT_EVAL = NOT_EVAL + 1

    return metrics


def compute_results(rating_profiles_eval, algorithm, conv):
    metrics = []
    for j in range(0, N_USERS):

        if conv is None:
            folder = str(upper(algorithm))
            path = "data_processed/" + folder + "/" + algorithm + "_results" + str(j)

        else:
            folder = str(upper(algorithm))
            path = "data_processed/" + folder + "/" + algorithm + "_results_" + conv + str(j)

        target_user = rating_profiles_eval[j]
        cur_results = get_metrics_from_results(path, target_user, algorithm)

        metrics.append(cur_results)

    return metrics


# evaluation function for CF DF H1 and H2
def evaluate0(alg):
    print('-----', alg, '-----')
    q_results = compute_results(ml25M_profiles_eval, alg, 'q')
    m_results = compute_results(ml25M_profiles_eval, alg, 'm')
    g_results = compute_results(ml25M_profiles_eval, alg, 'g')
    all_results = [q_results, m_results, g_results]

    for cur_result in all_results:
        final_results = [0, 0, 0, 0, 0, 0]

        for c in cur_result:
            final_results = np.add(final_results, c)

        print("NDCG@10", end='\t')
        print("NDCG@30", end='\t')
        print("Precision@10", end='\t')
        print("Precision@30", end='\t')
        print("mAP", end='\t')
        print("mRR")

        print(final_results[0] / (N_USERS - NOT_EVAL), end='\t')
        print(final_results[1] / (N_USERS - NOT_EVAL), end='\t')
        print(final_results[2] / (N_USERS - NOT_EVAL), end='\t')
        print(final_results[3] / (N_USERS - NOT_EVAL), end='\t')
        print(final_results[4] / (N_USERS - NOT_EVAL), end='\t')
        print(final_results[5] / (N_USERS - NOT_EVAL), '\n')


# evaluation function for H3 H4
def evaluate1(alg):
    print('-----', alg, '-----')

    results = compute_results(ml25M_profiles_eval, alg, None)
    final_results = [0, 0, 0, 0, 0, 0]

    for c in results:
        final_results = np.add(final_results, c)

    print("NDCG@10", end='\t')
    print("NDCG@30", end='\t')
    print("Precision@10", end='\t')
    print("Precision@30", end='\t')
    print("mAP", end='\t')
    print("mRR")

    print(final_results[0] / N_USERS, end='\t')
    print(final_results[1] / N_USERS, end='\t')
    print(final_results[2] / N_USERS, end='\t')
    print(final_results[3] / N_USERS, end='\t')
    print(final_results[4] / N_USERS, end='\t')
    print(final_results[5] / N_USERS, '\n')


"""Import Data"""
ml25M_profiles_eval = import_ratings_profiles("data_processed/ml25M_users_test_eval.csv")

N_USERS = len(ml25M_profiles_eval)
NOT_EVAL = 0

evaluate0('cf')
evaluate0('cb')
evaluate0('h1')
evaluate0('h2')
evaluate1('h3')
evaluate1('h4')
