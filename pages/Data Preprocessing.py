"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np
from cardio.preprocessing_scripts import norm_filt
from cardio.vis import make_vis
import seaborn as sns
import matplotlib.pyplot as plt

# Read in data from the Google Sheet.
# Getting data from a google sheet: https://docs.streamlit.io/knowledge-base/tutorials/databases/public-gsheet
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data()
def load_data(sheets_url):
    """
    Takes a secret url, for a public google spreadsheet and formats it to download.

    Parameter
    ---------
    sheets_url: String that specifies 

    Returns
    -------
    A pandas DataFrame
    """
    # Test that the URLs are correct
    assert "/edit#gid=" in sheets_url, "URL specified is not a public google sheet. Please check permissions."

    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    # st.write(f"attempting to access {csv_url}")
    return pd.read_csv(csv_url)

# Loading data in from the google sheet URLs
with st.spinner('Wait for it...'):
    metacard_drug = load_data(st.secrets["metacard_drug_public_gsheets_url"])
    metacard_kegg = load_data(st.secrets["metacard_kegg_public_gsheets_url"])
    metacard_metadata = load_data(st.secrets["metacard_metadata_public_gsheets_url"])
    metacard_microbiome = load_data(st.secrets["metacard_microbiome_public_gsheets_url"])
    metacard_serum = load_data(st.secrets["metacard_serum_public_gsheets_url"])
    metacard_taxonomy = load_data(st.secrets["metacard_taxonomy_public_gsheets_url"])
    metacard_urine = load_data(st.secrets["metacard_urine_public_gsheets_url"])
    st.write("Success!")

# Displaying a view of the data.
datasets = {"metacard_drug": metacard_drug, 
            "metacard_kegg":metacard_kegg, 
            "metacard_metadata": metacard_metadata,
            "metacard_microbiome": metacard_microbiome,
            "metacard_serum": metacard_serum,
            "metacard_taxonomy": metacard_taxonomy, 
            "metacard_urine": metacard_urine}

if 'datasets' not in st.session_state:
    st.session_state['datasets'] = datasets

abundance = norm_filt.count_to_abundance(metacard_microbiome)

processed_datasets = {"abundance" : abundance}

with st.expander(label = "Count to Abundance", expanded=False):
    st.write("## Raw Microbiome")
    st.dataframe(metacard_microbiome.head(50))
    st.write("## Processed Microbiome")
    st.dataframe(abundance)

metamicro = norm_filt.combine_metamicro(metacard_metadata, metacard_serum, abundance)

# Filter sparse for metabolites and species seperately then merge
# Add a slider widget that can change how much to filter. Report on number species removed.

 # this is where we could remove X-metabolites
metamicro_filt = norm_filt.filter_sparse(metamicro, metamicro.columns[2:], percent=0.1)

processed_datasets['metamicro_filt'] = metamicro_filt

if 'processed_datasets' not in st.session_state:
    st.session_state['processed_datasets'] = processed_datasets
    st.session_state['metamicro'] = metamicro

with st.expander(label="Combine Datasets", expanded=False):
    st.dataframe(metamicro)
    

