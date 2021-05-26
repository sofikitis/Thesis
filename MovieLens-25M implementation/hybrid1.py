import numpy as np
import json

import content_based_filtering as cb
import parameters as p


class UserProfile:
    def __init__(self, ide, movies, ratings):
        self.ide = ide
        self.movies = movies
        self.ratings = [float(r) for r in ratings]


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


def run_hybrid1(target_user, movie_profiles, number_of_target_user):
    # Step 1: Import results from collaborative Filtering
    # we use the results from the game algorithm
    file = "Data_Processed/CF/cf_results_g" + str(number_of_target_user)
    cf_results = import_cf_results(file)

    # Step 2: Find the top rated movies of each user in the top-k
    high_rated_movies_ids = []
    for user in cf_results:

        # get the sorted indices of the top movies
        top_ratings = np.argsort(user.ratings)[::-1][:p.HIGH_MOVIES]

        for ide in top_ratings:
            if ide < len(user.movies):                                                 # """prepei na to allaksw"""
                high_rated_movies_ids.append(user.movies[ide])

    high_rated_movies = []
    for movie_id in high_rated_movies_ids:
        for movie in movie_profiles:
            if movie.ide == movie_id:
                high_rated_movies.append(movie)
                break

    high_rated_movies = np.array(high_rated_movies)

    # Step 3: Run content based filtering these movies
    h1_results = cb.run_content_based_filtering(target_user, movie_profiles, high_rated_movies)

    return h1_results
