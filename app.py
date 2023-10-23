import streamlit as st
import pandas as pd
import numpy as np
import os

df = pd.read_csv('imdb_top_1000.csv')
print(df.shape)

df['Released_Year'] = df['Released_Year'].apply(lambda x: 1995 if x == 'PG' else int(x))

df = df.dropna()
print(df.shape)

df['Gross'] = df['Gross'].apply(lambda x: x.replace(',','') if ',' in x else x).astype(int)
df['Runtime'] = df['Runtime'].apply(lambda x: x.replace(' min','') if ' min' in x else x).astype(int)

print(df.dtypes)
