import numpy as np
import pandas as pd
import json
from numpy.core.defchararray import upper

import collaborative_filtering as cf
import content_based_filtering as cb
import hybrid1 as h1
import hybrid2 as h2
import hybrid3 as h3
import hybrid4 as h4


class UserProfile:
    def __init__(self, ide, movies, ratings):
        self.ide = ide
        self.movies = movies
        self.ratings = [float(r) for r in ratings]

    def dump(self):
        return {'User': {'ID': self.ide,
                         'movies': list(self.movies),
                         'ratings': list(self.ratings)}}


class MovieProfile:
    def __init__(self, ide, tags):
        self.ide = int(ide)
        self.tags = [float(r) for r in tags]


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


# Function to import the movie profiles
def import_movie_profiles(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, sep=',', header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    movie_profiles_list = []
    for movie in profiles_matrix[1:]:
        profile = MovieProfile(movie[0], movie[1:])

        movie_profiles_list.append(profile)

    return np.array(movie_profiles_list)


# Function to export CF results to files
def export_cf_results(results, name, i):
    folder = str(upper(name))

    destination0 = "Data_Processed/" + folder + "/" + name + "_results_q" + str(i)
    destination1 = "Data_Processed/" + folder + "/" + name + "_results_m" + str(i)
    destination2 = "Data_Processed/" + folder + "/" + name + "_results_g" + str(i)

    result0 = [obj.dump() for obj in results[0]]
    result1 = [obj.dump() for obj in results[1]]
    result2 = [obj.dump() for obj in results[2]]

    with open(destination0, 'w') as outfile:
        json.dump(result0, outfile)

    with open(destination1, 'w') as outfile:
        json.dump(result1, outfile)

    with open(destination2, 'w') as outfile:
        json.dump(result2, outfile)


# Function to export CB results to files
def export_cb_results(results, name, i):
    folder = str(upper(name))

    destination0 = "Data_Processed/" + folder + "/" + name + "_results_q" + str(i)
    destination1 = "Data_Processed/" + folder + "/" + name + "_results_m" + str(i)
    destination2 = "Data_Processed/" + folder + "/" + name + "_results_g" + str(i)

    np.savetxt(destination0, results[0], delimiter=",", fmt='%1.10f')
    np.savetxt(destination1, results[1], delimiter=",", fmt='%1.10f')
    np.savetxt(destination2, results[2], delimiter=",", fmt='%1.10f')


def collaborative_filtering_25M(target_profile, train_profiles, i):
    print("Running Collaborative Filtering")

    ml25M_cf_results = cf.run_collaborative_filtering(target_profile, train_profiles)

    export_cf_results(ml25M_cf_results, 'cf', i)


def content_based_filtering_25M(target_profile, movie_profiles, i):
    print("Running Content Based Filtering")

    ml25M_cb_results = cb.run_content_based_filtering(target_profile, movie_profiles, movie_profiles)

    export_cb_results(ml25M_cb_results, 'cb', i)


def hybrid1_25M(target_profile, movie_profiles, i):
    print("Running Hybrid1 Filtering")

    ml25M_h1_results = h1.run_hybrid1(target_profile, movie_profiles, i)

    export_cb_results(ml25M_h1_results, 'h1', i)


def hybrid2_25M(target_profile, train_profiles, i):
    print("Running Hybrid2 Filtering")

    ml25M_h2_results = h2.run_hybrid2(target_profile, train_profiles, i)

    export_cf_results(ml25M_h2_results, 'h2', i)


def hybrid3_25M(target_profile, i):
    print("Running Hybrid3 Filtering")

    ml25M_h3_results = h3.run_hybrid3(target_profile, i)

    destination = "Data_Processed/H3/h3_results" + str(i)
    np.savetxt(destination, ml25M_h3_results, delimiter=",", fmt='%1.10f')


def hybrid4_25M(target_profile, i):
    print("Running Hybrid4 Filtering")

    ml25M_h4_results = h4.run_hybrid4(target_profile, i)

    destination = "Data_Processed/H4/h4_results" + str(i)
    np.savetxt(destination, ml25M_h4_results, delimiter=",", fmt='%1.10f')


"""Import Data"""
# Import MovieLens-25M rating profiles (train and test)
ml25M_user_profiles_train = import_ratings_profiles("Data_Processed/ml25M_users_train.csv")
ml25M_user_profiles_test_history = import_ratings_profiles("Data_Processed/ml25M_users_test_history.csv")

# Import MovieLens-25M movie profiles
ml25M_movie_profiles = import_movie_profiles("Data_Processed/ml25M_movie_profiles.csv")


# normalize ratings of users
for j in ml25M_user_profiles_train:
     j.ratings = j.ratings - np.average(j.ratings) + 0.00000001

for j in ml25M_user_profiles_test_history:
    j.ratings = j.ratings - np.average(j.ratings) + 0.00000001


"""Run filterings"""
n = len(ml25M_user_profiles_test_history)

for u in range(0, n):
    print("Running for user:", u)
    c_target_profile = ml25M_user_profiles_test_history[u]

    collaborative_filtering_25M(c_target_profile, ml25M_user_profiles_train, u)
    content_based_filtering_25M(c_target_profile, ml25M_movie_profiles, u)
    hybrid1_25M(c_target_profile, ml25M_movie_profiles, u)
    hybrid2_25M(c_target_profile, ml25M_user_profiles_train, u)
    hybrid3_25M(c_target_profile, u)
    hybrid4_25M(c_target_profile, u)
