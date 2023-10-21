import streamlit as st
import pandas as pd
import numpy as np
import os

print(os.listdir(os.curdir))

print("Hello world!")

df = pd.read_csv('imdb_top_1000.csv')

print(df) 
#test