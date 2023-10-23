import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import os

df = pd.read_csv('imdb_top_1000.csv')
print(df.shape)

print('Null Count: ',df.isnull().sum())
df= df.drop('Certificate',axis=1)
#We removed the Certificate column because it had unreliable data as well as missing values that overly-pruned our data.
#We are choosing not to impute the missing values for the other two columns "Meta_score" and "Gross" out of an abundance of caution to prevent as little bias as possible.

df['Released_Year'] = df['Released_Year'].apply(lambda x: 1995 if x == 'PG' else int(x))

df = df.dropna()
print(df.shape)

df['Gross'] = df['Gross'].apply(lambda x: x.replace(',','') if ',' in x else x).astype(int)
df['Runtime'] = df['Runtime'].apply(lambda x: x.replace(' min','') if ' min' in x else x).astype(int)
#print(df.dtypes)



#*************************************
#********* Streamlit Section *********
#*************************************
st.title('Top 1000 IMDb Movies & TV Shows')

st.markdown('Our problem is that movie ratings are too retroactive. We will create a dependable movie ratings prediction model to set movie makers’ and audience members\' expectations upon the release of a new film, before the critics.')
st.markdown('This application will be useful for two primary reasons: \n\n\t\t(1) rating expectations impact film financing, and \n\t(2) rating expectations impact audience willingness to attend.')
st.markdown('While researching current movie rating applications, most showed current ratings like RottenTomatoes, IMDb, and Metacritic created by viewers, but do not show predictions of movie ratings created by prediction models. However, there were many articles about utilizing prediction models with no application being created to interact with and allow usage of the models.')

st.header('Data Statistics')
st.write(df.describe())

st.header('Data Head')
st.write(df.head())
