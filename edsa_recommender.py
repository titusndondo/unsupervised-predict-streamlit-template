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
                              We'll need to fix it!")


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
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

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
        train_data, test_data = train_test_split(ratings_train, test_size = 0.25)
        train_data = turicreate.load_sframe(train_data)
        test_data = turicreate.load_sframe(test_data)
        popularity_model = turicreate.popularity_recommender.create(train_data, user_id='userId', item_id='movieId', target='rating')
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
        userId = st.text_area("Enter User ID", "Type Here")
        if st.button("Sign In"):
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(ratings_train[['userId','movieId', 'rating']], reader)
            from surprise.model_selection import train_test_split
            trainset, testset = train_test_split(data, test_size=.25)
            model = SVD()
            svd_rec = model.fit(trainset)
            person_of_int = ratings_train[ratings_train['userId']==userId]
            person = person_of_int.drop('timestamp', axis=1)
            recommended = svd_rec.predict(person)
            st.title("We think you'll like:")
            for i,j in enumerate(recommended):
                st.subheader(str(i+1)+'. '+j)            
            
            
if __name__ == '__main__':
    main()
