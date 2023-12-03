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
    genres = df.Genre.replace({', ':','},regex=True)
    genres = genres.str.split(',').explode('Genre')
    genres = np.unique(genres)
    director = df["Director"].unique()
    director = np.sort(director)

    stars = pd.Series(pd.concat([df['Star1'], df['Star2'], df['Star3'], df['Star4']]).unique())
    stars = np.sort(stars)

    runtime = 0
    star1 = []
    star2 = []
    star3 = []
    star4 = []
    tobedropped = []

    st.title('Movie Rating Predictor Application')
    
    st.markdown('Welcome to the Movie Rating Predictor Application: the proactive approach to predicting your movies’ audience ratings before the critics!')
    
    st.markdown('The problem is that movie ratings are too retroactive. We will create a dependable movie ratings prediction model to set movie makers’ and audience members\' expectations upon the release of a new film, before the critics.')
    st.markdown('This application will be useful for two primary reasons: \n\n\t\t(1) rating expectations impact film financing, and \n\t(2) rating expectations impact audience willingness to attend.')
    st.markdown('While researching current movie rating applications, most showed current ratings like RottenTomatoes, IMDb, and Metacritic created by viewers, but do not show predictions of movie ratings created by prediction models. However, there were many articles about utilizing prediction models with no application being created to interact with and allow usage of the models. That is where we come in!')

    st.header('Model Data Source & Head')
    st.markdown('Name: IMDB Movies Dataset')
    st.markdown('Owner: HARSHIT SHANKHDHAR ')
    st.markdown('Sourced: IMDb ')
    st.markdown('Stored: Kaggle.com ')
    st.markdown('Date: Feb 01, 2021') 
    st.markdown('URL: https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows')
    st.write(df.head())


    st.header('Data Statistics')
    st.write(df.describe())
    st.write(df.plot.hexbin(x='Runtime', y='IMDB_Rating', gridsize=15)
    st.markdown('Based on a comparative analysis using SSE and MSE scores to decide on the backend predictive model, we have employed a Linear Regression model trained on the IMDB dataset on the backend of our application to predict your film's rating. To review our findings and see all of the work mentioned above, visit https://colab.research.google.com/drive/1eXeMPPkGnWUJ5szIomcrde-Ouv7XAdOr?usp=sharing')

    st.header('Instructions and Disclaimer')  
    st.write('Please input the following information below and we will use our trained Linear Regression model to predict your film’s rating.')
    st.write('Note: The model was trained off free publicly available data scraped from IMDB in 2021 and rating predictions may slightly differ from critic and audience ratings. This tool is only meant to be a free tool to assist in setting expectations and does not guarantee anything.') 

    release_selection = st.number_input("Select the release year:", step=1, min_value=1920, max_value=2050)
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
        elif len(star_selection) == 0:
            star1 = ''
            star2 = ''
            star3 = ''
            star4 = ''
        genre_input = ' '.join(genre_selection)

        prediction_data = [release_selection, runtime_selection, genre_input, director_selection, star1, star2, star3, star4, tobedropped]
        prediction_data = np.array(prediction_data, dtype=object)
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

        training_columns = ['Released_Year', 'Runtime', 'Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']
        y = d['IMDB_Rating']
        x = d[training_columns]
        sc = StandardScaler()
        x = sc.fit_transform(x)
        prediction_data = sc.transform(prediction_data)

        model = LinearRegression()

        # Train the model on the training set
        print(type(prediction_data))

        model.fit(x, y)
        predictions = model.predict(prediction_data)

        st.success(predictions.round(2))

if __name__ == "__main__":
    main()
