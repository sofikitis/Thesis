import numpy as np
from collaborative_filtering import run_collaborative_filtering
from demographic_filtering import run_demographic_filtering
import parameters as p


def find_id_of_user(user,  all_users):

    user_id = 0
    for cur_user in all_users:
        if (cur_user == user).all():
            break
        else:
            user_id += 1

    return user_id


def run_hybrid1(target_user_cf, train_profiles, target_user_df, demographic_profiles):
    # update the module
    p.MODULE = 1

    # Step 1: Get the results from Collaborative Filtering (Game)
    cf_results = run_collaborative_filtering(target_user_cf, train_profiles)
    topk_users = cf_results[2]

    # find the demographic profiles of the top-k users
    topk_users_id = []
    for user in topk_users:
        user_id = find_id_of_user(user, train_profiles)
        topk_users_id.append(user_id)

    relevant_demographic_profiles = []
    for user in topk_users_id:
        relevant_demographic_profiles.append(demographic_profiles[user])

    relevant_demographic_profiles = np.array(relevant_demographic_profiles)

    # Step 3: Run content based filtering with  these users
    df_results = run_demographic_filtering(target_user_df, relevant_demographic_profiles)

    return df_results
