import numpy as np
import pandas as pd
import json
from genetic_algorithm import run_ga


class UserProfile:
    def __init__(self, ide, movies, ratings):
        self.ide = ide
        self.movies = np.array(movies)

        ratings = [float(r) for r in ratings]
        self.ratings = np.array(ratings)


def import_data(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


# Function to import data and the results of filterings
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


def run_hybrid4(cf_target_user, i):

    # import the collaborative filtering results
    path = "Data_Processed/CF/cf_results_g" + str(i)
    cf_results = import_cf_results(path)

    path = "Data_Processed/H2/h2_results_g" + str(i)
    h2_results = import_cf_results(path)

    # get unique ids of movies rated by top-k users from Collaborative Filtering and Hybrid2 Filtering
    unique_ids = get_unique_ids(cf_results, h2_results)

    file = "Data_Processed/CF_Ratings/cf_ratings" + str(i)
    population_cf = import_data(file)

    file = "Data_Processed/CF_Ratings/cf_ratings" + str(i)
    population_h2 = import_data(file)

    population = []
    for cf, h2 in zip(population_cf, population_h2):
        population.append(cf)
        population.append(h2)

    target_user = np.zeros(len(unique_ids))
    for movie in range(len(unique_ids)):
        movie_id = unique_ids[movie]

        if movie_id in cf_target_user.movies:
            pos = cf_target_user.movies.index(movie_id)
            target_user[movie] = cf_target_user.ratings[pos]

    # run genetic algorithm to find the predicted ratings
    predicted_ratings = run_ga(target_user, population, unique_ids)

    return predicted_ratings
