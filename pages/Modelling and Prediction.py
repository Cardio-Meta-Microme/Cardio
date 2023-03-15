import streamlit as st
import pandas as pd
import numpy as np
from cardio.vis import make_vis
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.write("## The Final Model")

wdir = os.getcwd()

st.write(f"Retrieving from directory: {wdir}/cardio/")

model = pd.read_pickle(wdir + "/cardio/Trained_Production_RF_Classifier_230314.pkl")

model_features = pd.read_pickle(wdir + "/cardio/Trained_Production_RF_Classifier_features_230314.pkl")

st.write(type(model_features[:10].tolist()))

st.pyplot(make_vis.feature_histograms(st.session_state['metamicro_filt'][0], st.session_state['metamicro_filt'][0].iloc[10], model_features[:20].tolist()), )