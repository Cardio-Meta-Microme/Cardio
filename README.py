"""
# Landing Page
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("README - Metabolander")

st.markdown("""## Cardio
Tool for Analysis of meta microbiomes and metabolomes.

=======

## Components


## Patient status abbreviations:
- IHD: ischemic heart disease patients
- HC: healthy controls
- MMC: metabolically matched controls
- UMMC unmedicated metabolically matched controls
- ACS: acute coronary syndrome
- CIHD: chronic IHD
- HF: heart failure due to IHD

### Dependencies
- numpy
- scipy
- pandas
- matplotlib
- openpyxl

## TODO

- Revamp README page
- Make all of the data loading and processing occur silently in the README
- Add extra descriptions to the Data Preprocessing
    - mathematical description of processing steps
    - extra transformations (e.g. how the distributions of features change)
- Make the model training and model prediction pages

""")

