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

