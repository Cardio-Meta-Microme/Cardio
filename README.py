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
- openpyxl""")

