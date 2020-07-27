"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import turicreate

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
titles = pd.read_csv('resources/data/movies.csv')
ratings_train = pd.read_csv('resources/data/ratings.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","About", "Trending", "Recommender"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                        st.title("We think you'll like:")
                        for i,j in enumerate(top_recommendations):
                            st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              #We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                        st.title("We think you'll like:")
                        for i,j in enumerate(top_recommendations):
                            st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "About":
        st.title("About")
        image = {"play":"https://3.bp.blogspot.com/-7Spg1mVpPm8/WRMKj5pUN0I/AAAAAAACklU/Ct1vOhZ7gtk06OXtdbCfGElR0jmExy1oQCLcB/s1600/movie_recommend.gif"}
        st.video(image["play"])

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == "Trending":
        movie_name = st.text_area("Enter Movie Title", "Type Here")
        if st.button("Search"):
            movies = titles[titles['title'].str.contains(movie_name, case=False, regex=False)]
            movie_titles = movies['title']
            for i,j in enumerate(movie_titles):
                st.subheader(str(i+1)+'. '+j)
            #st.success(movie_titles)
        
        st.subheader("Popular Movies")
        #st.image('resources/imgs/best.png',use_column_width=True)
        st.image('resources/imgs/five_star.jpg',use_column_width=True)
        ratings_train.pop('timestamp')
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(ratings_train, test_size = 0.25)
        train_df = turicreate.load_sframe(train_data)
        test_df = turicreate.load_sframe(test_data)
        popularity_model = turicreate.popularity_recommender.create(train_df, user_id='userId', item_id='movieId', target='rating')
        popular = popularity_model.recommend(users = [133, 3567], k = 5)
        df = popular.to_dataframe()
        data = pd.merge(df, titles, how='inner', on='movieId')
        films = data['title'].unique()
        for i,j in enumerate(films):
            st.subheader(str(i+1)+'. '+j)
        #st.write(films)
        st.subheader('New Release')
        video = {'avengers':'https://www.youtube.com/watch?v=rYC7Dpe-4mU'}
        st.video(video['avengers'])

    if page_selection == "Recommender":
        st.image('resources/imgs/movie.jpg',use_column_width=True)
        sys = st.radio("New user Register Profile",
                       ('Sign In',
                        'Register'))
        if sys == 'Sign In':
            userId = st.text_area("Enter User ID", "Type Here")
            if st.button('Recommend'):
                def collab(userId,top_n=10):
                    dataset = ratings_train.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
                    train = ratings_train
                    metric = 'cosine'
                    user_id = int(userId)
    
                    similarities=[]
                    indices=[]
                    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute') 
                    model_knn.fit(dataset)

                    distances, indices = model_knn.kneighbors(dataset.iloc[user_id, :].values.reshape(1, -1), n_neighbors = 20)
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
                    top_10_recommendation = df[0:top_n]
                    Movie_Name = top_10_recommendation.merge(titles, how='inner', on='movieId')
                    recommended_movies = Movie_Name.title.values.tolist()
               
                    return recommended_movies   
                recommended_movie = collab(userId, top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(recommended_movie):
                    st.subheader(str(i+1)+'. '+j)
        if sys == 'Register':
            st.title("Discover your movie in a few clicks")
            st.subheader("Enter your three favorite movies")
            movie1 = st.text_area("Enter First Preference", "Type Here")
            movie2 = st.text_area("Enter Second Preference", "Type Here")
            movie3 = st.text_area("Enter Third Preference", "Type Here")
            favorites = [movie1, movie2, movie3]
            if st.button('Recommend'):
                top_recommendations = content_model(movie_list=favorites,
                                                            top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)  
            
if __name__ == '__main__':
    main()
