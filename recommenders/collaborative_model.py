# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
dataset = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)


def get_ids(movie_list):
    movies = pd.DataFrame(movie_list, columns=['title'])
    df = pd.merge(movies, movies_df, how='inner', on='title')
    movie_ids = list(df['movieId'])
    return movie_ids


def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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

    #indices = pd.Series(movies_df['title'])
    movie_ids = get_ids(movie_list)
    train = ratings_df
    empty = pd.DataFrame()
    for i in movie_ids:
        ds = train[train['movieId']==i]
        empty = pd.concat([empty, ds])
    best_rating = empty[empty['rating']>=3]
    count_ratings = best_rating.groupby('userId').count()
    sorted_df = count_ratings.sort_values('movieId', ascending=False)
    user_id = sorted_df.index[0]
    
    metric = 'cosine'
    
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
    model_knn.fit(dataset)

    distances, indices = model_knn.kneighbors(dataset.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue;
    train = train.astype({"movieId": str})
    Movie_user = train.groupby(by = 'userId')['movieId'].apply(lambda x:','.join(x))
    b = indices.squeeze().tolist()
    d = Movie_user[Movie_user.index.isin(b)]
    l = ','.join(d.values)
    Movie_seen_by_similar_users = l.split(',')
    Movies_under_consideration = list(map(int, Movie_seen_by_similar_users))
    df = pd.DataFrame({'movieId':Movies_under_consideration})
    top_10_recommendation = df[0:top_n+1]
    Movie_Name = top_10_recommendation.merge(movies_df, how='inner', on='movieId')
    recommended_movies = Movie_Name.title.values.tolist()
               
    return recommended_movies
