"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np

# Importing data
# This is a dataset with all the movies and their similaries
# Takes long to preprocess. The user will query the csv file to interact with
# the app. The files has 4 columns:
# 1. 'title' - of the movie being queried
# 2. 'recommendations' - the recommended movies for the movie being queried
# 3. 'similarity' - the similarity coefficient based on cosine similarity
# 4. 'rank' - the ranking of the recommended movies
# Note that each movie has 10 recommended movies in the file

similarities_data = pd.read_csv('predict_deliverables/data/similarities_data.csv')

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # initialising list of recommended movies
    recommended_movies = list()

    # iterating through user inputed movie list to get the recommendations
    for i, movie_title in enumerate(movie_list):

        # filtering similaries_data for a given movie's data
        sim_data = similarities_data[similarities_data['title'] == movie_title]

        # getting only the recommendations column (collects 10 movies)
        similar_movies = sim_data['recommendations'].tolist()

        # Now, for eact of the collected movie, append to recommended movies
        for movie in similar_movies:
            recommended_movies.append(movie)

    # By the end of the loop, if the user queried 3 movies, we have 30
    # recommended movies. 10 for each movie quiried. We randomly collect
    # ton_n movies.
    pd.Series(recommended_movies).sample(top_n).tolist()

    # finally, return the list of top_n recommended movies
    return recommended_movies
