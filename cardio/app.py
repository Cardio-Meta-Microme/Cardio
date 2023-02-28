"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
from preprocessing_scripts import get_dataframes

# The write function is a handy magic that will interpret input and display it
# The object is displayed in whatever streamlit thinks is a reasonable way.
st.write(" # Displaying a Dataframe")

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

st.write(" # Adding some interactivity")

def generate_table():
    st.session_state['df'] = pd.DataFrame({
        'first column': np.random.rand(5),
        'second column': np.random.rand(5)
        })
    st.session_state['cols'] = 3


col1, col2 = st.columns(2)

if 'cols' not in st.session_state:
    st.session_state['cols'] = 3

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame({
        'first column': np.random.rand(5),
        'second column': np.random.rand(5)
        })

with col1:
    if st.button(label="Clear DF"):
        generate_table()

with col2:
    if st.button(label="Add Column"):
        name = "column" + str(st.session_state.cols)
        st.session_state.df[name] = np.random.rand(5)
        st.session_state.cols += 1

st.session_state.df

if st.button(label = "Fetch Raw Data"):
    st.write("Fetching data!")
    st.session_state["raw_data"] = get_dataframes.get_df()
    st.write("Data fetched!")
    st.write(st.session_state.raw_data[0].head())



