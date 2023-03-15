from cardio.vis import make_vis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Checking if the data are still loaded using the session state API
if 'datasets' in st.session_state:
    st.write('The data is still loaded!')

st.markdown("""
## Understanding the Data

We've built some interactive plots to explore the distribution of our patients training data.
""")

st.write(st.session_state['metamicro_filt'][0])

labels = st.session_state['metamicro_filt'][0].columns.values
fig_boxplot, fig_gender = make_vis.plot_general_dist_altair(st.session_state['metamicro_filt'][0])
st.markdown("""
### General Distributions

As a sanity check there are some obvious differences between the BMI of our different
cohort groups.

#### 

Unmedicated metabolocially matched controls have a higher BMI, are younger 

""")
st.altair_chart(fig_boxplot)

st.write("Gender")

col = st.columns(5)

col[0].altair_chart(fig_gender, use_container_width=True)

st.dataframe(st.session_state['metamicro_filt'][0])

fig_micro = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'][0], 'Bacteria', st.session_state['metamicro_filt'][1], st.session_state['metamicro_filt'][2])
fig_metabolite = make_vis.plot_micro_abundance(st.session_state['metamicro_filt'][0], 'Metabolite', st.session_state['metamicro_filt'][1], st.session_state['metamicro_filt'][2])

st.write("## New Figure (Lit 😎 🔥🔥🔥🔥)")
st.altair_chart(fig_micro, use_container_width=True)
st.altair_chart(fig_metabolite, use_container_width=True)

fig_umap = make_vis.plot_UMAP(st.session_state['metamicro_filt'][0], "default", True, st.session_state['metamicro_filt'][1], st.session_state['metamicro_filt'][2])

st.write("## UMAP")
st.altair_chart(fig_umap, use_container_width=True)

fig_cluster_age_BMI = make_vis.cluster_age_bmi(st.session_state['metamicro_filt'][0])

st.write("## Scatter Plot of Age and BMI")
st.altair_chart(fig_cluster_age_BMI, use_container_width=True)
