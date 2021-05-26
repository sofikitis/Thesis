import numpy as np
from collaborative_filtering import run_collaborative_filtering
from demographic_filtering import run_demographic_filtering
import parameters as p
import pandas as pd


# Function to import data and the results of filterings
def import_data(file):
    # Import profiles csv with pandas
    profiles = pd.read_csv(file, header=None, index_col=False)

    # Raw data to pandas DataFrame
    data_frame = pd.DataFrame(profiles)

    # Convert data frame to numpy array
    profiles_matrix = data_frame.to_numpy()

    return profiles_matrix.astype(np.float)


def run_hybrid2(target_user_cf, train_profiles, target_user_df, demographic_profiles):
    # update the module
    p.MODULE = 2

    # Step 1: Get the results from Demographic Filtering (Game)
    df_results = run_demographic_filtering(target_user_df, demographic_profiles)
    topk_users = df_results[2]

    relevant_profiles = []
    for user in topk_users:
        user_id = user[1]
        relevant_profiles.append(train_profiles[user_id])

    relevant_profiles = np.array(relevant_profiles)
    print(len(relevant_profiles))
    # Step 3: Run content based filtering these users
    cf_results = run_collaborative_filtering(target_user_cf, relevant_profiles)

    return cf_results
