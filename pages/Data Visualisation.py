import streamlit as st
import pandas as pd
import numpy as np
from cardio.preprocessing_scripts import norm_filt
from cardio.vis import make_vis
import seaborn as sns
import matplotlib.pyplot as plt

# Checking if the data are still loaded using the session state API
if 'datasets' in st.session_state:
    st.write('The data is still loaded!')

percent = st.slider('What prevalence threshold do you want', 0.0, 1.0, 0.25, 0.05)
st.session_state['metamicro_filt'] = norm_filt.filter_sparse(st.session_state['metamicro'], st.session_state['metamicro'].columns, percent=percent)
st.write(st.session_state['metamicro_filt'])

labels = st.session_state['metamicro_filt'].columns.values
fig = make_vis.plot_general_dist_altair(st.session_state['metamicro_filt'])
st.write("## General Distributions")
st.altair_chart(fig)

fig_micro = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'].drop('Gender', axis=1), 'Bacteria')
fig_metabolite = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'].drop('Gender', axis=1), 'Metabolite')

st.write("## New Figure (Lit 😎 🔥🔥🔥🔥)")
st.altair_chart(fig_micro)
st.altair_chart(fig_metabolite)