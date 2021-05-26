import numpy as np
import pandas as pd
from numpy.core.defchararray import upper

import parameters as p
import collaborative_filtering as cf
import demographic_filtering as df
import content_based_filtering as cb
import hybrid1 as h1
import hybrid2 as h2
import hybrid4 as h4
import hybrid3 as h3
import hybrid3_2 as h3_2


# Function to import the ratings to a profiles matrix
def import_ratings_profiles(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


# Function to export results to files
def export_results(results, name, i):
    folder = str(upper(name))

    destination0 = "Data_Processed/" + folder + "/" + name + "_results_q" + str(i)
    destination1 = "Data_Processed/" + folder + "/" + name + "_results_m" + str(i)
    destination2 = "Data_Processed/" + folder + "/" + name + "_results_g" + str(i)

    np.savetxt(destination0, results[0], delimiter=",", fmt='%1.1f')
    np.savetxt(destination1, results[1], delimiter=",", fmt='%1.1f')
    np.savetxt(destination2, results[2], delimiter=",", fmt='%1.1f')


# function with input the df results and output the rating profiles
def calculate_ratings_df(all_results, all_users):

    relevant_profiles = []

    for results in all_results:
        c_relevant_profiles = []

        for user in results:
            user_id = int(user[1])
            c_relevant_profiles.append(all_users[user_id])

        relevant_profiles.append(c_relevant_profiles)

    return relevant_profiles


# Function with input the cb results and output the rating profiles
def cb_results_to_profile(target_user, cb_results):
    """
    The results contain three list of the top-k movies
    For every list(ConvQ, ConvM, Game) we will create
    a ratings profile and use it to evaluate the method
    """
    predicted_ratings = []

    for result in cb_results:
        ratings = np.zeros((len(target_user)))
        i = 0
        penalty = 0
        for movie in result:
            movie_id = movie[1]
            ratings[int(movie_id)] = 5 - penalty
            i += 1
            if i % (p.K // 2) == 0:
                penalty += 0.5
        predicted_ratings.append([ratings])

    return predicted_ratings


def collaborative_filtering_100k(target_profile, train_profiles, i):
    print("Running Collaborative Filtering")
    ml100k_cf_results = cf.run_collaborative_filtering(target_profile, train_profiles)

    export_results(ml100k_cf_results, 'cf', i)


def demographic_filtering_100k(target_user_demographic, demographic_profiles, train_profiles, i):
    print("Running Demographic Filtering")

    # Run demographic filtering
    ml100k_df_results = df.run_demographic_filtering(target_user_demographic, demographic_profiles)

    ml100k_df_results = calculate_ratings_df(ml100k_df_results, train_profiles)

    export_results(ml100k_df_results, 'df', i)


def content_based_filtering_100k(test_profiles_history, movie_profiles, i):
    ml100k_cb_results = cb.run_collaborative_filtering(test_profiles_history, movie_profiles)

    ml100k_cb_results = cb_results_to_profile(test_profiles_history, ml100k_cb_results)

    export_results(ml100k_cb_results, 'cb', i)


def hybrid1_100k(target_user_cf, train_profiles, target_profile_df, demographic_profiles, i):
    print("Running Hybrid1")

    # run hybrid 1 filtering
    ml100k_h1_results = h1.run_hybrid1(target_user_cf, train_profiles, target_profile_df, demographic_profiles)

    ml100k_h1_results = calculate_ratings_df(ml100k_h1_results, train_profiles)

    export_results(ml100k_h1_results, 'h1', i)


def hybrid2_100k(target_profile_cf, train_profiles, target_profile_df, demographic_profiles, i):
    print("Running Hybrid2")

    # run hybrid 1 filtering
    ml100k_h2_results = h2.run_hybrid2(target_profile_cf, train_profiles, target_profile_df, demographic_profiles)

    export_results(ml100k_h2_results, 'h2', i)


def hybrid3_ml100k(target_profile, i):
    print("Running Hybrid3")

    destination = "Data_Processed/H3/h3_results" + str(i)

    h3_results = [h3_2.run_hybrid3(target_profile, i)]

    np.savetxt(destination, h3_results, delimiter=",", fmt='%1.10f')


def hybrid4_ml100k(target_profile, i):
    print("Running Hybrid4")

    destination = "Data_Processed/H4/h4_results" + str(i)

    h4_results = h4.run_hybrid4(target_profile, i)

    np.savetxt(destination, h4_results, delimiter=",", fmt='%1.5f')


"""Import Data"""
ml100k_profiles_train = import_ratings_profiles("data_processed/ml100k_user_profiles_train.csv")
ml100k_profiles_history = import_ratings_profiles("data_processed/ml100k_user_profiles_test_history.csv")
ml100k_profiles_eval = import_ratings_profiles("data_processed/ml100k_user_profiles_test_evaluation.csv")

ml100k_demographic_train = import_ratings_profiles("data_processed/ml100k_demographic_profiles_train.csv")
ml100k_demographic_test = import_ratings_profiles("data_processed/ml100k_demographic_profiles_test.csv")

# ml100k_movie_profiles = import_ratings_profiles("data_processed/ml100k_movie_genre_profiles.csv")

"""Run filterings"""
n = len(ml100k_profiles_history)


for u in range(0, n):
    print("Running for user:", u)

    c_target_profile = ml100k_profiles_history[u]
    c_target_profile_demographic = ml100k_demographic_test[u]

    # collaborative_filtering_100k(c_target_profile, ml100k_profiles_train, u)
    # demographic_filtering_100k(c_target_profile_demographic, ml100k_demographic_train, ml100k_profiles_train, u)
    # hybrid1_100k(c_target_profile, ml100k_profiles_train, c_target_profile_demographic, ml100k_demographic_train, u)
    # hybrid2_100k(c_target_profile, ml100k_profiles_train, c_target_profile_demographic, ml100k_demographic_train, u)
    hybrid3_ml100k(c_target_profile, u)
    # hybrid4_ml100k(c_target_profile, u)

import experiments
