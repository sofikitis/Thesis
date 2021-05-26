import numpy as np
import pandas as pd
from scipy import stats

import collaborative_filtering as cf
import demographic_filtering as df
from genetic_algorithm import run_ga


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

    return res


def run_hybrid4(cf_target_user, i):

    # import the collaborative filtering results
    path = "Data_Processed/CF/cf_results_g" + str(i)
    cf_results = import_data(path)

    # import the demographic filtering results
    path = "Data_Processed/DF/df_results_g" + str(i)
    df_results = import_data(path)

    population = []
    for i, j in zip(df_results, cf_results):
        population.append(i)
        population.append(j)

    # run genetic algorithm to find the predicted ratings
    predicted_ratings = run_ga(cf_target_user, population)

    return predicted_ratings
