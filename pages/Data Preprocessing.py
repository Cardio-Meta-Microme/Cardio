"""
# Data Preprocessing Page
In which we show the transformations that took place between the raw data and the 
final data that were fed into the model.


"""

import streamlit as st
import pandas as pd
import numpy as np
from cardio.preprocessing_scripts import norm_filt
from cardio.vis import make_vis
from cardio.preprocessing_scripts import trimdata
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read in data from the Google Sheet.
# Getting data from a google sheet: https://docs.streamlit.io/knowledge-base/tutorials/databases/public-gsheet
@st.cache_data()
def load_data(sheets_url, sheets = False):
    """
    Takes a secret url, for a public google spreadsheet and formats it to download.

    Parameter
    ---------
    sheets_url: String that specifies 

    Returns
    -------
    A pandas DataFrame
    """
    # Check whether you download locally or doesnload from a google sheet
    wdir = os.getcwd()
    if sheets:
        # Test that the URLs are correct
        assert "/edit#gid=" in sheets_url, "URL specified is not a public google sheet. Please check permissions."

        csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
        # st.write(f"attempting to access {csv_url}")
        data = pd.read_csv(csv_url)
    else:
        data = pd.read_excel(wdir + "/data" + sheets_url)
    return data

metacard_drug = load_data("/metacard_drug.xlsx")
metacard_kegg = load_data("/metacard_kegg.xlsx")
metacard_metadata = load_data("/metacard_metadata.xlsx")
metacard_microbiome = load_data("/metacard_microbiome.xlsx")
metacard_serum = load_data("/metacard_serum.xlsx")
metacard_taxonomy = load_data("/metacard_taxonomy.xlsx")
metacard_urine = load_data("/metacard_urine.xlsx")


# Loading data in from the google sheet URLs.
# ONLY USE IF YOU DON'T HAVE THE LOCAL FILES.
if st.button(label="Fetch Data (DO NOT PRESS ON UNIVERSITY WIFI)"):
    with st.spinner('Wait for it...'):
        metacard_drug = load_data(sheets_url = st.secrets["metacard_drug_public_gsheets_url"], sheets=True)
        metacard_kegg = load_data(sheets_url = st.secrets["metacard_kegg_public_gsheets_url"], sheets=True)
        metacard_metadata = load_data(sheets_url = st.secrets["metacard_metadata_public_gsheets_url"], sheets=True)
        metacard_microbiome = load_data(sheets_url = st.secrets["metacard_microbiome_public_gsheets_url"], sheets=True)
        metacard_serum = load_data(sheets_url = st.secrets["metacard_serum_public_gsheets_url"], sheets=True)
        metacard_taxonomy = load_data(sheets_url = st.secrets["metacard_taxonomy_public_gsheets_url"], sheets=True)
        metacard_urine = load_data(sheets_url = st.secrets["metacard_urine_public_gsheets_url"], sheets=True)
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

# Calling the new preprocessing
abundance_new = trimdata.preprocess()

st.markdown("""
# GOAL: Classifier Model for Heart Disease Risk from Microbiome and Metabolome

There are a few things that we need to do before we can train a model that will try to classify a person at risk for Ischemic Heart
Disease (IHD).

Firstly, microbiome data typically come processed as raw read counts (actually there are some steps before this but our dataset already had those 
transformations done).



## Preprocessing Steps


""")

with st.expander(label = "See Preprocessing", expanded=False):
    st.write("## Raw Microbiome")
    st.dataframe(metacard_microbiome.head(50))
    st.write("## Processed Microbiome")
    st.dataframe(abundance_new[0])
    columns_original = len(metacard_microbiome.columns.values) + len(metacard_serum.columns.values)
    columns_processed = len(abundance_new[0].columns.values)
    st.write(f"The unprocessed dataframe had {columns_original} columns, the processed dataframe has {columns_processed} columns")
    fig, ax = plt.subplots()
    sns.histplot(data=abundance_new[0], x = 'shannon', ax = ax, hue="Gender")
    ax.set_title("Shannon Diversity of Sample")
    st.pyplot(fig)

if 'processed_datasets' not in st.session_state:
    st.session_state['metamicro_filt'] = abundance_new
    

