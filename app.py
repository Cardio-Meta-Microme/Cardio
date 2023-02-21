"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

st.write(" # Displaying a Dataframe")

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.write(" # Adding some interactivity")

def generate_table():
    df = pd.DataFrame({
    'first column': np.random.rand(5),
    'second column': np.random.rand(5)
    })

    df

if st.button(label="Generate DF"):
    generate_table()



