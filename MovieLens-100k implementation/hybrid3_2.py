import numpy as np
import pandas as pd
from scipy import stats
import math

# calculate ratings method (0 Weighted Average // 1 Average)
METHOD = 1


def wave(x, y):
    w = np.inner(x, y)

    # return 1 / (1 + math.exp(-w))
    return math.log((w + math.sqrt(1 + pow(w, 2))))


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
            cur_correlation = stats.spearmanr(target_user, user)[0]
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
def calculate_utility(real_ratings, this_ratings):

    utility = stats.spearmanr(real_ratings, this_ratings)[0]    # * wave(real_ratings, this_ratings)

    return utility


def calculate_predicted_ratings(arrayA, arrayB, a, b):
    predicted_ratings = np.zeros(len(arrayA))

    # return np.multiply(arrayA, a) + np.multiply(arrayB, b)

    for j in range(len(arrayA)):
        if arrayA[j] != 0 and arrayB[j] != 0:
            predicted_ratings[j] = arrayA[j] * a + arrayB[j] * b

        elif arrayA[j] == 0 and arrayB[j] != 0:
            predicted_ratings[j] = arrayB[j]

        else:
            predicted_ratings[j] = arrayA[j]

    return predicted_ratings


# game to update CF and df utilities
def run_game(cf_results, df_results, target_user):

    # get the CF and df predicted ratings
    cf_ratings = calculate_ratings(cf_results, target_user)
    df_ratings = calculate_ratings(df_results, target_user)

    update = 0.01
    a = 1
    b = 0

    old_predicted_ratings = calculate_predicted_ratings(cf_ratings, df_ratings, a, b)

    for i in range(100):

        a -= update
        b += update

        new_predicted_ratings = calculate_predicted_ratings(cf_ratings, df_ratings, a, b)

        old_utility = calculate_utility(target_user, old_predicted_ratings)
        new_utility = calculate_utility(target_user, new_predicted_ratings)

        # calculate convergence
        convergence = old_utility > new_utility or a >= 1 or b >= 1

        if convergence:
            break

        old_predicted_ratings = new_predicted_ratings

        print(old_utility, new_utility, a, b)

    return old_predicted_ratings


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
