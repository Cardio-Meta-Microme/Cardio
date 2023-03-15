"""
# Data Preprocessing Page
In which we show the transformations that took place between the raw data and the 
final data that were fed into the model.


"""

import streamlit as st
import pandas as pd
import numpy as np
from cardio.vis import make_vis
from cardio.preprocessing_scripts import trimdata
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read in data from the Google Sheet.
# Getting data from a google sheet: https://docs.streamlit.io/knowledge-base/tutorials/databases/public-gsheet
@st.cache_data()
def load_data(path, sheets = False):
    """
    Takes a secret url, for a public google spreadsheet and formats it to download.

        Parameters:
            path(string): A string that is either the google sheet URL, or the local path to the csv datafiles.
            sheets(boolean): Flag that indicates whether you are pulling from google sheets or from local. Default False.
        Returns:
            data(pandas dataframe): 
    """
    # Check whether you download locally or doesnload from a google sheet
    wdir = os.getcwd()
    if sheets:
        # Test that the URLs are correct
        assert "/edit#gid=" in path, "URL specified is not a public google sheet. Please check permissions."

        csv_url = path.replace("/edit#gid=", "/export?format=csv&gid=")
        # st.write(f"attempting to access {csv_url}")
        data = pd.read_csv(csv_url)
    else:
        data = pd.read_excel(wdir + "/data" + path)
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
        metacard_drug = load_data(path = st.secrets["metacard_drug_public_gsheets_url"], sheets=True)
        metacard_kegg = load_data(path = st.secrets["metacard_kegg_public_gsheets_url"], sheets=True)
        metacard_metadata = load_data(path = st.secrets["metacard_metadata_public_gsheets_url"], sheets=True)
        metacard_microbiome = load_data(path = st.secrets["metacard_microbiome_public_gsheets_url"], sheets=True)
        metacard_serum = load_data(path = st.secrets["metacard_serum_public_gsheets_url"], sheets=True)
        metacard_taxonomy = load_data(path = st.secrets["metacard_taxonomy_public_gsheets_url"], sheets=True)
        metacard_urine = load_data(path = st.secrets["metacard_urine_public_gsheets_url"], sheets=True)
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

""")

with st.expander(label = "See Preprocessing", expanded=False):
    st.write("## Raw Microbiome")
    st.dataframe(metacard_microbiome.head(50))
    st.markdown("""
    ## Preprocessing Steps
    - combine metadata, microbiome, and metabolome into one dataframe with patients as indices
    - drop patients missing over 1000 features from model
    - calculate each patient’s shannon diversity from microbe counts
    - centered log ratio (CLR) transform counts to relative abundance:
        - we have to do this because compositional data is constrained by total 
        - image address: Aitchison_triadlogratio.jpg
        - formula:  $ clr(x) =  \ln\left[\\frac{x_1}{g_m(x)},\ldots,\\frac{x_D}{g_m(x)}\\right] $ where $ g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D} $ is the geometric mean of x 
    - filter sparse features separately for microbiome/metabolome
        - sparse defined as having more than a certain number of NAs
    """)
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

if st.button(label="download data"):
    abundance_new[0].to_csv("processed_data.csv")
    

