import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Function to import the ratings file to a user profiles matrix
def import_user_ratings_100k(file):
    # Import ratings csv with pandas
    raw_data = pd.read_csv(file, sep='\t')

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(raw_data)

    # Find the unique user and movie ids in the data
    users = data_frame.userId.unique()
    movies = data_frame.movieId.unique()
    users.sort()
    movies.sort()

    # Create a matrix which will contain the profiles
    profiles_matrix = np.zeros((len(users) + 1, len(movies) + 1))

    # Initialize the first column and line with the id of users and movies (will delete later)
    for i in range(0, len(users)):
        profiles_matrix[i + 1, 0] = users[i]

    for i in range(0, len(movies)):
        profiles_matrix[0, i + 1] = movies[i]

    for row in data_frame.itertuples():
        cur_user = int(row[1])
        cur_movie = int(row[2])
        value = row[3]

        # Find the column and line with the corresponding movie and tag Id
        x = np.where(profiles_matrix[:, 0] == cur_user)[0]
        y = np.where(profiles_matrix[0, :] == cur_movie)[0]

        profiles_matrix[x, y] = value

    # delete first column and first line of the matrix, these are the movie and tag ids
    profiles_matrix = np.delete(profiles_matrix, 0, axis=0)
    profiles_matrix = np.delete(profiles_matrix, 0, axis=1)

    return profiles_matrix


# function to import demographic data of users
def import_demographic_data(file):
    # Import ratings csv with pandas
    raw_data = pd.read_csv(file, sep='|')

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(raw_data)

    occupations = data_frame.occupation.unique()
    occupations_to_number = np.arange(len(occupations))

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    # occupation to occupation id
    for i in range(len(occupations)):
        positions = np.where(profiles_matrix[:, 3] == occupations[i])[0]

        for j in positions:
            profiles_matrix[j, 3] = occupations_to_number[i]

    # gender to number
    for user in profiles_matrix:
        if user[2] == 'F':
            user[2] = 1
        else:
            user[2] = 0

    zip_codes = data_frame.zip.unique()
    zip_codes_to_number = np.arange(len(zip_codes))

    # zip_code to zip_code id
    for i in range(len(zip_codes)):
        positions = np.where(profiles_matrix == zip_codes[i])[0]

        for j in positions:
            profiles_matrix[j, 4] = zip_codes_to_number[i]

    profiles_matrix = profiles_matrix.astype(float)
    profiles_matrix = np.delete(profiles_matrix, 0, 1)

    for col in range(len(profiles_matrix[0])):
        profiles_matrix[:, col] = preprocessing.scale(profiles_matrix[:, col])

    return profiles_matrix


# Function to export profiles matrix to csv for future use
def export_data_to_csv(matrix, name):
    np.savetxt(name, matrix, delimiter=",", fmt='%1.1f')


# Function to keep 75% of ratings as history and 25% for evaluation
def history_eval_split(matrix_test):
    # For every user in the test set we keep 75% of ratings as history and use the 25% to evaluate
    rows_in_matrix_test = len(matrix_test)
    columns_in_matrix_test = len(matrix_test[0])

    matrix_test_history = np.zeros((rows_in_matrix_test, columns_in_matrix_test))
    matrix_test_evaluation = np.zeros((rows_in_matrix_test, columns_in_matrix_test))

    for i in range(rows_in_matrix_test):
        rated_movies = 0
        for j in range(columns_in_matrix_test):

            # Every three non zero ratings we keep the rating in evaluation matrix
            cur_rating = matrix_test[i, j]
            if cur_rating != 0:
                if rated_movies == 3:
                    matrix_test_evaluation[i, j] = cur_rating
                    rated_movies = 0
                else:
                    matrix_test_history[i, j] = cur_rating
                    rated_movies += 1
    return matrix_test_history, matrix_test_evaluation


# Function to keep 75% of ratings as history and 25% for evaluation
def history_eval_split_single(matrix_test):
    # For every user in the test set we keep 75% of ratings as history and use the 25% to evaluate
    columns_in_matrix_test = len(matrix_test)

    matrix_test_history = np.zeros(columns_in_matrix_test)
    matrix_test_evaluation = np.zeros(columns_in_matrix_test)

    rated_movies = 0
    for j in range(columns_in_matrix_test):

        # Every three non zero ratings we keep the rating in evaluation matrix
        cur_rating = matrix_test[j]
        if cur_rating != 0:
            if rated_movies == 3:
                matrix_test_evaluation[j] = cur_rating
                rated_movies = 0
            else:
                matrix_test_history[j] = cur_rating
                rated_movies += 1
    return matrix_test_history, matrix_test_evaluation


# Function to split user profiles to train and test sets
def preprocess_user_profiles(matrix):
    matrix_train, matrix_test = train_test_split(matrix, test_size=0.25, shuffle=True)

    matrix_test_history, matrix_test_evaluation = history_eval_split(matrix_test)

    return matrix_train, matrix_test_history, matrix_test_evaluation


def find_id_of_user(t_user, all_users):
    user_id = 0
    for cur_user in all_users:
        if (cur_user == t_user).all():
            break
        else:
            user_id += 1

    return user_id


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

    return np.array(centered_matrix)


# Import MovieLens-100k ratings
ml100k_ratings_profiles = import_user_ratings_100k("dataset/u.data")
ml100k_ratings_profiles = centering(ml100k_ratings_profiles)

# Import MovieLens-100k demographic data
ml100k_demographic_profiles = import_demographic_data("dataset/u.user")

# Train - Test split the data
cf_profiles_train, cf_profiles_test, df_profiles_train, df_profiles_test = \
    train_test_split(ml100k_ratings_profiles, ml100k_demographic_profiles, test_size=0.25, shuffle=True)


# History - Evaluation split the ml100k test profiles
rating_profiles_history, rating_profiles_eval = history_eval_split(cf_profiles_test)

# Save train and test sets to csv
export_data_to_csv(cf_profiles_train, "data_processed/ml100k_user_profiles_train.csv")
export_data_to_csv(rating_profiles_history, "data_processed/ml100k_user_profiles_test_history.csv")
export_data_to_csv(rating_profiles_eval, "data_processed/ml100k_user_profiles_test_evaluation.csv")

export_data_to_csv(df_profiles_train, "data_processed/ml100k_demographic_profiles_train.csv")
export_data_to_csv(df_profiles_test, "data_processed/ml100k_demographic_profiles_test.csv")
