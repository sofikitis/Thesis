import numpy as np
import pandas as pd
from scipy import stats
import math

# calculate ratings method (0 Weighted Average // 1 Average)
METHOD = 1


def sigmoid(x):
    try:
        # return 1 / (1 + math.exp(-x))
        return math.log((x + math.sqrt(1+pow(x, 2))))
    except OverflowError:
        if x > 0:
            return 1
        else:
            return 0


# Function to import data and the results of filterings
def import_data(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


def normalize_ratings(array):
    c_min = array.min()
    c_max = array.max()
    res = [(a - c_min) / (c_max - c_min) * 5 if a != 0 else 0 for a in array]

    return np.array(res)


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
def calculate_ratings(results, target_user):
    # Method == 0 weighted average
    if METHOD == 0:
        # find the average of target user's ratings
        target_users_avg = calculate_average_rating(target_user)

        sum_of_ratings = np.zeros(len(target_user))
        sum_of_correlation = 0

        for user in results:
            cur_correlation = stats.pearsonr(target_user, user)[0]
            cur_avg = calculate_average_rating(user)

            sum_of_ratings = sum_of_ratings + cur_correlation * (user - cur_avg)
            sum_of_correlation += cur_correlation

        predicted_ratings = target_users_avg + sum_of_ratings / sum_of_correlation

    else:
        sum_of_ratings = np.zeros(len(target_user))
        number_of_ratings = np.zeros(len(target_user))

        for user in results:
            for r in range(len(user)):
                if r != 0:
                    sum_of_ratings[r] = sum_of_ratings[r] + user[r]
                    number_of_ratings[r] = number_of_ratings[r] + 1

        predicted_ratings = [s / n if n != 0 else 0 for s, n in zip(sum_of_ratings, number_of_ratings)]

    return predicted_ratings


# function to calculate utilities
def calculate_utility(real_ratings, this_ratings, parameter):

    avg = calculate_average_rating(real_ratings)
    real_ratings = [r-avg if r != 0 else 0 for r in real_ratings]

    avg = calculate_average_rating(this_ratings)
    this_ratings = [r - avg if r != 0 else 0 for r in this_ratings]

    # utility = sigmoid(np.inner(real_ratings, this_ratings)) / parameter

    utility = stats.spearmanr(real_ratings, this_ratings)[0] / parameter

    return utility


# game to update CF and df utilities
def run_game(cf_results, df_results, target_user):

    # get the CF and df predicted ratings
    cf_ratings = calculate_ratings(cf_results, target_user)
    df_ratings = calculate_ratings(df_results, target_user)

    a, b = 0.5, 0.5
    update = 0.01

    iteration = 0
    convergence = False
    while not convergence:
        iteration += 1

        cf_utility = calculate_utility(target_user, cf_ratings, a)
        df_utility = calculate_utility(target_user, df_ratings, b)

        # calculate convergence
        convergence = - 0.01 < cf_utility - df_utility < 0.01 or a >= 1 or b >= 1 or iteration > 50
        print(cf_utility, df_utility, a, b)
        if cf_utility > df_utility:
            a = a + update
            b = b - update
        else:
            a = a - update
            b = b + update

    predicted_ratings = np.zeros(len(target_user))

    for j in range(len(target_user)):
        if cf_ratings[j] != 0 and df_ratings[j] != 0:
            predicted_ratings[j] = cf_ratings[j] * a + df_ratings[j] * b

        elif cf_ratings[j] == 0 and df_ratings[j] != 0:
            predicted_ratings[j] = df_ratings[j]

        else:
            predicted_ratings[j] = cf_ratings[j]

    return predicted_ratings


def run_hybrid3(cf_target_user, i):
    # import the collaborative filtering results
    path = "Data_Processed/CF/cf_results_g" + str(i)
    cf_results = import_data(path)

    # import the demographic filtering results
    path = "Data_Processed/DF/df_results_g" + str(i)
    df_results = import_data(path)

    cf_results = [normalize_ratings(profile) for profile in cf_results]
    df_results = [normalize_ratings(profile) for profile in df_results]
    cf_target_user = normalize_ratings(cf_target_user)

    predicted_ratings = run_game(cf_results, df_results, cf_target_user)

    return predicted_ratings

