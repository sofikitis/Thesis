import numpy as np
import pandas as pd

import collaborative_filtering as cf


# Functions to import data and the results of filterings
def import_cb_results(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


# function to add weight to topk movies
def clone_columns(target_user, train_profiles, topk_movies):
    for i in range(len(topk_movies)):
        # Find the top movie id
        movie_id = topk_movies[i][1]

        position = np.where(target_user.movies == movie_id)[0]

        if position.size > 0:
            position = target_user.movies.index(movie_id)
            rating = target_user.ratings[position]

            target_user.movies.append(movie_id)
            target_user.ratings = np.append(target_user.ratings, rating)

        for profile in train_profiles:
            position = np.where(profile.movies == movie_id)[0]

            if position.size > 0:
                rating = profile.ratings[position]
                profile.movies.append(movie_id)
                profile.ratings = np.append(profile.ratings, rating)

    return target_user, train_profiles


def run_hybrid2(target_user, train_profiles, number_of_target_user):

    # Step 1: Import Content Based Filtering results
    # we use the results from the game algorithm
    file = "Data_Processed/CB/cb_results_g" + str(number_of_target_user)
    cb_results = import_cb_results(file)

    target_user, train_profiles = clone_columns(target_user, train_profiles, cb_results)

    # Step 3: Run Collaborating filtering with the updated user profiles
    h2_results = cf.run_collaborative_filtering(target_user, train_profiles)

    return h2_results
