import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

print(os.listdir(os.curdir))

df = pd.read_csv('imdb_top_1000.csv')

df= df.drop('Certificate',axis=1)
df = df.dropna()
#We removed the Certificate column because it had unreliable data as well as missing values that overly-pruned our data.
#We are choosing not to impute the missing values for the other two columns "Meta_score" and "Gross" out of an abundance of caution to prevent as little bias as possible.

df['Released_Year'] = df['Released_Year'].apply(lambda x: 1995 if x == 'PG' else int(x))
df['Gross'] = df['Gross'].apply(lambda x: x.replace(',','') if ',' in x else x).astype(int)
df['Runtime'] = df['Runtime'].apply(lambda x: x.replace(' min','') if ' min' in x else x).astype(int)

# Create and clean inputs
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

#build the regression model
training_columns = ['Director', 'Genre', 'Released_Year', 'Runtime', 'Star1', 'Star2', 'Star3', 'Star4']
training_data = df[training_columns]

def model_run(predictionData):
    print("prediction data:",type(predictionData))
    return predictionData


def main():
    runtime = 0
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
    genre_selection = st.multiselect("Select the genre:", genres, placeholder="eg Action, Adventure")
    director_selection = st.selectbox("Select the director:", director)
    star_selection = st.multiselect("Select the top 4 stars of the film:", stars, placeholder="Select no more than 4 stars")
    prediction_data = [release_selection, runtime_selection, genre_selection, director_selection, star_selection]
    
    # Submit button
    if st.button("Submit"):
        result = model_run(prediction_data)

        # Display the result
        st.success(result)
if __name__ == "__main__":
    main()
