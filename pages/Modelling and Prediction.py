from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
from cardio.vis import make_vis
import seaborn as sns
import matplotlib.pyplot as plt
import os

from cardio.model import making_model

from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.write("## The Final Model")

if st.button(label="download data"):
    st.session_state['metamicro_filt'][0].iloc[0:5].to_csv("example_data.csv")

wdir = os.getcwd()

st.write(f"Retrieving from directory: {wdir}/cardio/")

#model = pd.read_pickle(wdir + "/cardio/Trained_Production_RF_Classifier_230314.pkl")
model_path = wdir + "/cardio/model/Trained_Production_RF_Classifier_230314.pkl"
features_path = wdir + "/cardio/model/Trained_Production_RF_Classifier_features_230314.pkl"
na_fill_path = wdir + "/cardio/model/na_fill_values.pkl"
model = making_model.RF_Classifier(model_path,features_path,na_fill_path)

model_features = pd.read_pickle(wdir + "/cardio/model/Trained_Production_RF_Classifier_features_230314.pkl")

#st.write(type(model_features[:10].tolist()))
st.write(model)

if st.button(label= "download some sample"):
    st.session_state['metamicro_filt'][0].sample(1).to_csv("example_data3.csv")

st.markdown("## Patient Input")

uploaded_file = st.file_uploader(label="Upload a Single Row to Predict", type="csv", accept_multiple_files=False)

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file, index_col=0)
    st.write(user_data)

    prediction_model = model.classify(user_data)

    st.write(prediction_model)
    
    st.markdown("""
    ### Patient Data in Sample Distribution

    Below is a visualisation of the patients data in the distribution of the 10 most informative
    features for our model.

    """)

    st.pyplot(make_vis.feature_histograms(st.session_state['metamicro_filt'][0], user_data.iloc[0] , model_features[:10].tolist()))



