import numpy as np
import pandas as pd
import json
import math
from scipy import stats


class UserProfile:
    def __init__(self, ide, movies, ratings):
        self.ide = ide
        self.movies = movies
        self.ratings = [float(r) for r in ratings]


def wave(x, y):
    w = np.inner(x, y)

    # return 1 / (1 + math.exp(-w))
    return math.log((w + math.sqrt(1 + pow(w, 2))))


# Function to get unique ids of movies top-k users have rated
def get_unique_ids(cf_results, h2_results):
    ids = []

    if h2_results is not None:
        for user_cf, user_h2 in zip(cf_results, h2_results):
            ids.append(user_cf.movies)
            ids.append(user_h2.movies)
    else:
        for user_cf in cf_results:
            ids.append(user_cf.movies)

    ids = np.array(ids, dtype=object)
    ids = np.concatenate(ids, axis=0)
    unique_ids = np.unique(ids)

    return unique_ids


# Function to create users' profiles to a ratings array
def results_to_ratings(cf_results, h2_results, ids, number_of_target_user):
    ratings_cf = np.zeros((len(cf_results), len(ids)))
    ratings_h2 = np.zeros((len(h2_results), len(ids)))

    for movie in range(len(ids)):
        movie_id = ids[movie]

        for user in range(len(cf_results)):

            if movie_id in cf_results[user].movies:
                pos = cf_results[user].movies.index(movie_id)
                ratings_cf[user, movie] = cf_results[user].ratings[pos]

            if movie_id in h2_results[user].movies:
                pos = h2_results[user].movies.index(movie_id)
                ratings_h2[user, movie] = h2_results[user].ratings[pos]

    destinationCF = "Data_Processed/CF_Ratings/cf_ratings" + str(number_of_target_user)
    np.savetxt(destinationCF, ratings_cf, delimiter=",", fmt='%1.10f')

    destinationH2 = "Data_Processed/H2_Ratings/h2_ratings" + str(number_of_target_user)
    np.savetxt(destinationH2, ratings_h2, delimiter=",", fmt='%1.10f')

    return ratings_cf, ratings_h2


# Function to import or compute the user ratings of the filterings
def get_ratings(cf_results, h2_results, unique_ids, number_of_target_user):

    try:
        # i have already computed the cf and h2 ratings and saved them to files
        fileCF = "Data_Processed/CF_Ratings/cf_ratings" + str(number_of_target_user)
        ratings_cf = import_data(fileCF)

        fileH2 = "Data_Processed/H2_Ratings/h2_ratings" + str(number_of_target_user)
        ratings_h2 = import_data(fileH2)

    except FileNotFoundError:
        ratings_cf, ratings_h2 = results_to_ratings(cf_results, h2_results, unique_ids, number_of_target_user)

    return ratings_cf, ratings_h2


# Function to calculate the final predictions based on the users' ratings and the weights a, b
def get_final_prediction(ratingsA, ratingsB, ids, comb, a, b, target_user):
    predicted_ratings = np.zeros(len(ids))

    if comb == 'H2':
        predictionA = [col[np.nonzero(col)].mean() for col in ratingsA.transpose()]
        predictionB = [col[np.nonzero(col)].mean() for col in ratingsB.transpose()]

        for i in range(len(ids)):

            rating_cf = predictionA[i]
            rating_h2 = predictionB[i]

            if not math.isnan(rating_cf) and not math.isnan(rating_h2) != 0:
                rating = rating_cf * a + rating_h2 * b
            elif math.isnan(rating_cf):
                rating = rating_h2
            else:
                rating = rating_cf

            predicted_ratings[i] = rating

    elif comb == 'CB':
        predictionA = [col[np.nonzero(col)].mean() for col in ratingsA.transpose()]
        predictionB = ratingsB

        ids = list(ids)
        predicted_ratings = predictionA
        max_rating = max(target_user.ratings)

        for movie in predictionB:
            ide = movie[1]

            if ide in ids:
                pos = ids.index(ide)

                if not math.isnan(predicted_ratings[pos]) and movie[0] != 0:
                    predicted_ratings[pos] = a * predicted_ratings[pos] + b * movie[0] * max_rating

                elif math.isnan(predicted_ratings[pos]) and movie[0] != 0:
                    predicted_ratings[pos] = movie[0] * max_rating

    final_prediction = [[r, ide] for r, ide in zip(predicted_ratings, ids)]

    return final_prediction


# Function to import data
def import_data(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


# Functions to import data and the results of filterings
def import_cb_results(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file)

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


# function to calculate ratings from collaborative filtering results
def cf_results_to_prediction(target_user, results):
    predicted_ratings = []

    for m in range(0, len(target_user.movies)):
        movie = target_user.movies[m]

        n_of_ratings = 0
        sum_of_ratings = 0

        for pr in range(0, len(results)):

            if movie in results[pr].movies:
                pos = results[pr].movies.index(movie)
                n_of_ratings = n_of_ratings + 1
                sum_of_ratings = sum_of_ratings + results[pr].ratings[pos]

        r = sum_of_ratings / n_of_ratings if n_of_ratings != 0 else 0
        predicted_ratings.append(r)

    return predicted_ratings


# function to calculate ratings from content based filtering results
def cb_results_to_prediction(target_user, results):
    prediction = np.zeros(len(target_user.ratings))
    max_rating = max(target_user.ratings)
    for m in range(0, len(target_user.movies)):
        movie = target_user.movies[m]

        pos = np.where(results[:, 1] == movie)[0]
        if pos.size > 0:
            prediction[m] = results[pos, 0] * max_rating

    return prediction


def calculate_predicted_ratings(arrayA, arrayB, a, b):
    predicted_ratings = np.zeros(len(arrayA))

    # return np.multiply(arrayA, a) + np.multiply(arrayB, b)

    for j in range(len(arrayA)):
        if arrayA[j] != 0 and arrayB[j] != 0:
            predicted_ratings[j] = arrayA[j] * a + arrayB[j] * b

        elif arrayA[j] == 0:
            predicted_ratings[j] = arrayB[j]

        else:
            predicted_ratings[j] = arrayA[j]

    return predicted_ratings


# function to calculate utilities
def calculate_utility(real_ratings, this_ratings):
    # utility = stats.spearmanr(real_ratings, this_ratings)[0]
    utility = wave(real_ratings, this_ratings) * stats.spearmanr(real_ratings, this_ratings)[0]

    return utility


# game to update CF and H2 utilities
def run_game(resultsA, resultsB, target_user):
    # get the CF and H2 predicted ratings
    cf_ratings = cf_results_to_prediction(target_user, resultsA)
    # h2_ratings = cf_results_to_prediction(target_user, resultsB)
    cb_ratings = cb_results_to_prediction(target_user, resultsB)

    update = 0.01
    a, old_a = 1, 1
    b, old_b = 0, 0

    old_predicted_ratings = calculate_predicted_ratings(cf_ratings, cb_ratings, a, b)

    for i in range(100):

        a -= update
        b += update

        new_predicted_ratings = calculate_predicted_ratings(cf_ratings, cb_ratings, a, b)

        old_utility = calculate_utility(target_user.ratings, old_predicted_ratings)
        new_utility = calculate_utility(target_user.ratings, new_predicted_ratings)

        # calculate convergence
        convergence = old_utility >= new_utility or b > 1 or math.isnan(old_utility)
        print(new_utility, a, b)
        if convergence:
            return old_a, old_b

        old_predicted_ratings = new_predicted_ratings
        old_a = a
        old_b = b


def run_hybrid3(target_user, number_of_target_user):
    # Step 1: Import Content Based Filtering and Collaborative Filtering results
    # we use the results from the game algorithms
    file = "Data_Processed/CF/cf_results_g" + str(number_of_target_user)
    cf_results = import_cf_results(file)

    file = "Data_Processed/H2/h2_results_g" + str(number_of_target_user)
    h2_results = import_cf_results(file)

    file = "Data_Processed/CB/cb_results_g" + str(number_of_target_user)
    cb_results = import_cb_results(file)

    a, b = run_game(cf_results, cb_results, target_user)

    # get unique ids of movies rated by top-k users from Collaborative Filtering and Hybrid2 Filtering
    unique_ids = get_unique_ids(cf_results, h2_results)

    ratings_cf, ratings_h2 = get_ratings(cf_results, h2_results, unique_ids, number_of_target_user)

    predicted_ratings = get_final_prediction(ratings_cf, cb_results, unique_ids, 'CB', a, b, target_user)

    return predicted_ratings
