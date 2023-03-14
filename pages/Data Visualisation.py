import streamlit as st
import pandas as pd
import numpy as np
from cardio.vis import make_vis
import seaborn as sns
import matplotlib.pyplot as plt

# Checking if the data are still loaded using the session state API
if 'datasets' in st.session_state:
    st.write('The data is still loaded!')

#percent = st.slider('What prevalence threshold do you want', 0.0, 1.0, 0.25, 0.05)
#st.session_state['metamicro_filt'] = norm_filt.filter_sparse(st.session_state['metamicro'], st.session_state['metamicro'].columns, percent=percent)
st.write(st.session_state['metamicro_filt'][0])

labels = st.session_state['metamicro_filt'][0].columns.values
fig = make_vis.plot_general_dist_altair(st.session_state['metamicro_filt'][0])
st.write("## General Distributions")
st.altair_chart(fig, use_container_width=True)

all_microbiome_columns = ["ID", "Status", "Age", "BMI", "Gender", "shannon"]
all_microbiome_columns.extend(st.session_state['metamicro_filt'][1])
all_metabolome_columns = ["ID", "Status", "Age", "BMI", "Gender", "shannon"]
all_metabolome_columns.extend(st.session_state['metamicro_filt'][2])

st.dataframe(st.session_state['metamicro_filt'][0][all_microbiome_columns])

fig_micro = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'][0][all_microbiome_columns], 'Bacteria')
fig_metabolite = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'][0][all_metabolome_columns], 'Metabolite')

st.write("## New Figure (Lit 😎 🔥🔥🔥🔥)")
st.altair_chart(fig_micro, use_container_width=True)
st.altair_chart(fig_metabolite, use_container_width=True)

fig_umap = make_vis.plot_UMAP(st.session_state['metamicro_filt'][0], "default", True)

st.write("## UMAP")
st.altair_chart(fig_umap, use_container_width=True)

fig_cluster_age_BMI = make_vis.cluster_age_bmi(st.session_state['metamicro_filt'][0])

st.write("## Scatter Plot of Age and BMI")
st.altair_chart(fig_cluster_age_BMI, use_container_width=True)
