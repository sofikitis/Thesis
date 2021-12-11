import numpy as np
import pandas as pd
from scipy.stats import stats
from numpy.core.defchararray import upper
import parameters as p


# Test users evaluation history
TARGET_HISTORY = []

# Predictions Calculation Method(0-Weighted Average, 1-Average)
METHOD = 1

# Top-k users to use for the predictions
TOPK = 100


# Function to import data and the results of filterings
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
def rating_profiles_to_prediction(results):
    # Method == 0 weighted average
    if METHOD == 0:
        # find the average of target user's ratings
        target_users_avg = calculate_average_rating(TARGET_HISTORY)

        sum_of_ratings = np.zeros(len(TARGET_HISTORY))
        sum_of_correlation = 0

        for user in results:
            cur_correlation = stats.spearmanr(TARGET_HISTORY, user)[0]
            cur_avg = calculate_average_rating(user)

            sum_of_ratings = sum_of_ratings + cur_correlation * (user - cur_avg)
            sum_of_correlation += cur_correlation

        predicted_ratings = target_users_avg + sum_of_ratings / sum_of_correlation

    else:
        sum_of_ratings = np.zeros(len(TARGET_HISTORY))
        number_of_ratings = np.zeros(len(TARGET_HISTORY))

        for user in results:
            for r in range(len(user)):
                if r != 0:
                    sum_of_ratings[r] = sum_of_ratings[r] + user[r]
                    number_of_ratings[r] = number_of_ratings[r] + 1

        predicted_ratings = [s / n if n != 0 else 0 for s, n in zip(sum_of_ratings, number_of_ratings)]

    return predicted_ratings


def normalize_ratings(array):
    c_min = array.min()
    c_max = array.max()
    res = [(a - c_min) / (c_max - c_min) * 5 if a != 0 else 0 for a in array]

    return res


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

    # delete movies without ratings from the profile
    if p.DELETE_ZEROS == 1:
        unrated_users = np.where(real_profile == 0)[0]
        real_profile = np.delete(real_profile, unrated_users)
        predicted_profile = np.delete(predicted_profile, unrated_users)

    ndcg0 = calculate_ndcg(real_profile, predicted_profile, p.N[0])
    ndcg1 = calculate_ndcg(real_profile, predicted_profile, p.N[1])

    precision0 = calculate_precision(real_profile, predicted_profile, p.N[0])
    precision1 = calculate_precision(real_profile, predicted_profile, p.N[1])

    ap = calculate_ap(real_profile, predicted_profile, p.N[0])
    rr = calculate_rr(real_profile, predicted_profile)

    metrics = [ndcg0, ndcg1, precision0, precision1, ap, rr]

    return metrics


def get_metrics_from_results(file_dir, target_user):
    results = import_data(file_dir)[:TOPK]

    results = [normalize_ratings(r) for r in results]

    if len(results) > 1:
        predictions = rating_profiles_to_prediction(results)
    else:
        predictions = results[0]

    metrics = calculate_metrics(target_user, predictions)

    return metrics


def compute_results(rating_profiles_eval, algorithm, conv):
    global TARGET_HISTORY

    metrics = []
    for j in range(0, N_USERS):

        if conv is None:
            folder = str(upper(algorithm))
            path = "data_processed/" + folder + "/" + algorithm + "_results" + str(j)

        else:
            folder = str(upper(algorithm))
            path = "data_processed/" + folder + "/" + algorithm + "_results_" + conv + str(j)

        target_user = rating_profiles_eval[j]
        TARGET_HISTORY = ml100k_profiles_history[j]
        cur_results = get_metrics_from_results(path, target_user)

        metrics.append(cur_results)

    return metrics


# evaluation function for CF DF H1 and H2
def evaluate0(alg):
    print('-----', alg, '-----')
    q_results = compute_results(ml100k_profiles_eval, alg, 'q')
    m_results = compute_results(ml100k_profiles_eval, alg, 'm')
    g_results = compute_results(ml100k_profiles_eval, alg, 'g')
    all_results = [q_results, m_results, g_results]

    for cur_result in all_results:
        final_results = [0, 0, 0, 0, 0, 0]

        for c in cur_result:
            final_results = np.add(final_results, c)


        print("NDCG@10", end=' ')
        print(final_results[0] / N_USERS)
        print("NDCG@30", end=' ')
        print(final_results[1] / N_USERS)

        print("Precision@10", end=' ')
        print(final_results[2] / N_USERS)
        print("Precision@30", end=' ')
        print(final_results[3] / N_USERS)

        print("mAP", end=' ')
        print(final_results[4] / N_USERS)
        print("mRR", end=' ')
        print(final_results[5] / N_USERS, '\n')


# evaluation function for H3 H4
def evaluate1(alg):
    print('-----', alg, '-----')

    results = compute_results(ml100k_profiles_eval, alg, None)
    final_results = [0, 0, 0, 0, 0, 0]

    for c in results:
        final_results = np.add(final_results, c)


    print("NDCG@10", end=' ')
    print(final_results[0] / N_USERS)
    print("NDCG@30", end=' ')
    print(final_results[1] / N_USERS)

    print("Precision@10", end=' ')
    print(final_results[2] / N_USERS)
    print("Precision@30", end=' ')
    print(final_results[3] / N_USERS)

    print("mAP", end=' ')
    print(final_results[4] / N_USERS)
    print("mRR", end=' ')
    print(final_results[5] / N_USERS, '\n')


"""Import Data"""
ml100k_profiles_eval = import_data("data_processed/ml100k_user_profiles_test_evaluation.csv")
ml100k_profiles_history = import_data("data_processed/ml100k_user_profiles_test_history.csv")

ml100k_profiles_eval = [normalize_ratings(profile) for profile in ml100k_profiles_eval]
ml100k_profiles_history = [normalize_ratings(profile) for profile in ml100k_profiles_history]

N_USERS = len(ml100k_profiles_eval)


# evaluate0('cf')
# evaluate0('df')
# evaluate0('h1')
# evaluate0('h2')
evaluate1('h3')
# evaluate1('h4')
