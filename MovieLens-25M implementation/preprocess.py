import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

"""General preprocessing functions"""


# Function to center a users ratings
def centering(matrix):
    centered_matrix = []

    for profile in matrix:

        number_of_ratings = 0
        ratings_sum = 0

        # Calculating average of user's ratings
        for rating in profile:
            if rating != 0:
                number_of_ratings += 1
                ratings_sum += rating

        average = ratings_sum / number_of_ratings

        # creating a profile with centered ratings
        centered_profile = []
        for rating in profile:
            if rating != 0:
                rating -= average
                centered_profile.append(rating)
            else:
                centered_profile.append(rating)

        centered_matrix.append(centered_profile)
    return centered_matrix


# Function to normalize a users ratings
def normalize(profile):
    normalized_profile = []
    for rating in profile:
        if rating != 0:
            rating = (rating - np.min(profile)) / (np.max(profile) - np.min(profile))
            normalized_profile.append(rating)
        else:
            normalized_profile.append(rating)

    return normalized_profile


# Function to split data to training and testing
def splitting(matrix):
    matrix = np.array(matrix)
    matrix = np.transpose(matrix)

    matrix_train, matrix_test = train_test_split(matrix, test_size=0.25, shuffle=True)

    matrix_train = np.transpose(matrix_train)
    matrix_test = np.transpose(matrix_test)

    return [matrix_train, matrix_test]


# Function to pre process the profiles matrix
def pre_process(matrix):
    matrix_centered = centering(matrix)

    # cur_profile_normalized = normalize(cur_profile_centered)

    [matrix_train, matrix_test] = splitting(matrix_centered)

    return [matrix_train, matrix_test]


"""Functions to import data and create profiles"""


# Function to import the ratings to a profiles matrix
def import_ratings_25M(read_file, write_file):

    with open(write_file, "ab") as file:

        # Import ratings csv with pandas
        raw_data = pd.read_csv(read_file)

        # Raw data to pandas DataFrame
        data_frame = pd.DataFrame(raw_data)

        # Find the unique user and movie ids in the data
        users = data_frame.userId.unique()
        movies = data_frame.movieId.unique()
        movies.sort()

        first_line = np.zeros((1, len(movies)))
        first_line[0, :] = movies
        np.savetxt(file, first_line, delimiter=",", fmt='%i')

        for cur_user in users:

            cur_user_profile = np.zeros((1, len(movies)))

            cur_indexes = np.where(data_frame.userId == cur_user)[0]

            data_frame_slice = data_frame.iloc[cur_indexes, :]

            for row in data_frame_slice.itertuples():
                cur_movie = row[2]
                value = row[3]

                # Find the line with the corresponding movie Id
                y = np.where(movies == cur_movie)
                cur_user_profile[0, y] = value

            np.savetxt(file, cur_user_profile, delimiter=",", fmt='%i')

    return None


# Function to import the ratings to an inverted index
def import_25M_ratings_to_inv(read_file, write_file):

    # Import ratings csv with pandas
    raw_data = pd.read_csv(read_file)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(raw_data)

    # Find the unique user and movie ids in the data
    users = data_frame.userId.unique()

    # Create dictionary
    entries = {}

    for user in users:

        # Find rating for this movie
        rows = data_frame.loc[data_frame['userId'] == user]
        rows = rows.to_numpy()

        movies = []
        ratings = []
        for row in rows:
            movies.append(row[1])
            ratings.append(row[2])

        entries[user] = [movies, ratings]

    with open(write_file, "w") as file:
        for key in entries.keys():
            file.write("%s,%s\n" % (key, entries[key]))


# Function to import the tags to a movie profile matrix
def import_tags(file):
    raw_data = pd.read_csv(file)

    data_frame = pd.DataFrame(raw_data)

    movies = data_frame.movieId.unique()
    tags = data_frame.tagId.unique()

    profiles_matrix = np.zeros((len(movies) + 1, len(tags) + 1))

    for i in range(0, len(movies)):
        profiles_matrix[i + 1, 0] = movies[i]

    for i in range(0, len(tags)):
        profiles_matrix[0, i + 1] = tags[i]

    for row in data_frame.itertuples():

        cur_movie = int(row[1])
        cur_tag = int(row[2])
        value = row[3]

        # Find the column and line with the corresponding movie and tag Id
        x = np.where(profiles_matrix[:, 0] == cur_movie)[0]
        y = np.where(profiles_matrix[0, :] == cur_tag)[0]

        profiles_matrix[x, y] = value

    # delete first column and first line of the matrix, these are the movie and tag ids
    # profiles_matrix = np.delete(profiles_matrix, 0, axis=0)
    profiles_matrix = np.delete(profiles_matrix, 0, axis=1)

    return profiles_matrix


# Function to export profiles matrix to csv for future use
def export_data_to_csv(matrix, name):
    np.savetxt(name, matrix, delimiter=",", fmt='%1.5f')


# Function to split user profiles to train and test(history and evaluation) !! can't run from here
"""
def test_train_split(file):
    ml25M_profiles = import_ratings_profiles("Data_Processed/ml25M_user_inverted_index.csv")

    ml25M_profiles_train, ml25M_profiles_test = train_test_split(ml25M_profiles, test_size=0.33)

    with open("Data_Processed/ml25M_users_train.csv", "w") as file:
        for user in ml25M_profiles_train:
            file.write("%s,%s,%s\n" % (user[0], user[1], user[2]))

    ml25M_profiles_test_history = []
    ml25M_profiles_test_eval = []
    for profile in ml25M_profiles_test:
        c_user = profile[0]
        c_rated_movies = profile[1]
        c_ratings = profile[2]

        rm_history, rm_eval, r_history, r_eval = train_test_split(c_rated_movies, c_ratings, test_size=0.25)

        c_history = [c_user, rm_history, r_history]
        c_eval = [c_user, rm_eval, r_eval]

        ml25M_profiles_test_history.append(c_history)
        ml25M_profiles_test_eval.append(c_eval)

    with open("Data_Processed/ml25M_users_test_history.csv", "w") as file:
        for user in ml25M_profiles_test_history:
            file.write("%s,%s,%s\n" % (user[0], user[1], user[2]))

    with open("Data_Processed/ml25M_users_test_eval.csv", "w") as file:
        for user in ml25M_profiles_test_eval:
            file.write("%s,%s,%s\n" % (user[0], user[1], user[2]))

"""

# MovieLens 25M
''' 
# Import MovieLens-25M ratings
import_ratings_25M("Data/MovieLens-25M/ratings.csv", "Data/Processed/ml25M_user_profiles.csv")

# Import MovieLens-25M tags
ml25M_movie_profiles = import_tags("Data/MovieLens-25M/genome-scores.csv")
# Export MovieLens-25M movie profiles to csv
export_data_to_csv(ml25M_movie_profiles, "Data/Processed/ml25M_movie_profiles")
'''

# MovieLens 25M ratings to inverted index
import_25M_ratings_to_inv("Data_Sets/ratings.csv", "Data_Processed/ml25M_user_inverted_index.csv")
