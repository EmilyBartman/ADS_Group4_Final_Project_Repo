import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

print(os.listdir(os.curdir))

def main():
    d = pd.read_csv('imdb_top_1000.csv')
    d=d[np.isfinite(pd.to_numeric(d.Released_Year, errors="coerce"))]
    d = d[['Released_Year', 'Runtime', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'IMDB_Rating']]
    d["Runtime"] = d.Runtime.replace({'min':''},regex=True)

    df = d.copy()

    #getting the dropdown values
    df = df.dropna()
    release_year = df['Released_Year'].unique()
    release_year = np.sort(release_year)
    genres = df.Genre.replace({', ':','},regex=True)
    genres = genres.str.split(',').explode('Genre')
    genres = np.unique(genres)
    director = df["Director"].unique()
    director = np.sort(director)
    stars = df["Star1"].append(df["Star2"])
    stars = stars.append(df["Star3"])
    stars = stars.append(df["Star4"]).unique()
    stars = np.sort(stars)

    runtime = 0
    star1 = []
    star2 = []
    star3 = []
    star4 = []
    tobedropped = []

    st.title('Top 1000 IMDb Movies & TV Shows')

    st.markdown('Our problem is that movie ratings are too retroactive. We will create a dependable movie ratings prediction model to set movie makers’ and audience members\' expectations upon the release of a new film, before the critics.')
    st.markdown('This application will be useful for two primary reasons: \n\n\t\t(1) rating expectations impact film financing, and \n\t(2) rating expectations impact audience willingness to attend.')
    st.markdown('While researching current movie rating applications, most showed current ratings like RottenTomatoes, IMDb, and Metacritic created by viewers, but do not show predictions of movie ratings created by prediction models. However, there were many articles about utilizing prediction models with no application being created to interact with and allow usage of the models.')

    st.header('Data Statistics')
    st.write(df.describe())

    st.header('Data Head')
    st.write(df.head())

    release_selection = st.selectbox("Select the release year:", release_year)
    runtime_selection = st.number_input("Enter the duration in minutes:", runtime)
    genre_selection = st.multiselect("Select the genres:", genres, placeholder="eg Action, Adventure")
    director_selection = st.selectbox("Select the director:", director)
    star_selection = st.multiselect("Select the top 4 stars of the film:", stars, placeholder="Select no more than 4 stars")
    # Submit button
    if st.button("Submit"):
        if len(star_selection) == 4:
            star1 = star_selection[0]
            star2 = star_selection[1]
            star3 = star_selection[2]
            star4 = star_selection[3]
        elif len(star_selection) == 3:
            star1 = star_selection[0]
            star2 = star_selection[1]
            star3 = star_selection[2]
            star4 = ''
        elif len(star_selection) == 2:
            star1 = star_selection[0]
            star2 = star_selection[1]
            star3 = ''
            star4 = ''
        elif len(star_selection) == 1:
            star1 = star_selection[0]
            star2 = ''
            star3 = ''
            star4 = ''
        genre_input = ' '.join(genre_selection)

        prediction_data = [release_selection, runtime_selection, genre_input, director_selection, star1, star2, star3, star4, tobedropped]
        prediction_data = np.array(prediction_data)
        prediction_df = pd.DataFrame([prediction_data], columns=d.columns)
        d = pd.concat([d, prediction_df])

        d["Genre"] = d["Genre"].astype('category')
        d["Genre"] = d["Genre"].cat.codes
        d["Director"] = d["Director"].astype('category')
        d["Director"] = d["Director"].cat.codes
        d["Star1"] = d["Star1"].astype('category')
        d["Star1"] = d["Star1"].cat.codes
        d["Star2"] = d["Star2"].astype('category')
        d["Star2"] = d["Star2"].cat.codes
        d["Star3"] = d["Star3"].astype('category')
        d["Star3"] = d["Star3"].cat.codes
        d["Star4"] = d["Star4"].astype('category')
        d["Star4"] = d["Star4"].cat.codes

        prediction_data = d.iloc[-1].copy()
        prediction_data = prediction_data.drop('IMDB_Rating')
        prediction_data = prediction_data.values.reshape(1, -1)
        d = d.drop(d.index[-1])

        print(prediction_data, "\n", d)

        training_columns = ['Director', 'Genre', 'Released_Year', 'Runtime', 'Star1', 'Star2', 'Star3', 'Star4']
        y = d['IMDB_Rating']
        x = d[training_columns]

        model = LinearRegression()

        # Train the model on the training set
        model.fit(x, y)
        print(prediction_data)
        predictions = model.predict(prediction_data)

        st.success(predictions)

if __name__ == "__main__":
    main()
