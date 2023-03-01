"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

# Read in data from the Google Sheet.
# Getting data from a google sheet: https://docs.streamlit.io/knowledge-base/tutorials/databases/public-gsheet
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url)

metacard_drug = load_data(st.secrets["metacard_drug_public_gsheets_url"])
metacard_kegg = load_data(st.secrets["metacard_kegg_public_gsheets_url"])
metacard_metadata = load_data(st.secrets["metacard_metadata_public_gsheets_url"])
metacard_microbiome = load_data(st.secrets["metacard_microbiome_public_gsheets_url"])
metacard_serum = load_data(st.secrets["metacard_serum_public_gsheets_url"])
metacard_taxonomy = load_data(st.secrets["metacard_taxonomy_public_gsheets_url"])
metacard_urine = load_data(st.secrets["metacard_urine_public_gsheets_url"])

if st.button(label = "Fetch Raw Data"):
    st.write(metacard_kegg.head(50))

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





