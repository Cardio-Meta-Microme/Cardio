# README Metabolander
Tool for Analysis of meta microbiomes and metabolomes.

[![Python Tests](https://github.com/Cardio-Meta-Microme/Cardio/actions/workflows/pytest_pip.yml/badge.svg)](https://github.com/Cardio-Meta-Microme/Cardio/actions/workflows/pytest_pip.yml)

## Running the App

The app is easy to run. 

1. Install the required packages from requirements.txt
2. Navigate to the Cardio directory (base directory) in command line or IDE
3. Run the command `streamlit run README.py'
4. Open the server instance on your desired browser.

**Note: Navigating through the pages in a non-linear way will not work. In future we would silently load all data on the README page. However, if you do navigate in a non-linear way, simply return to the Data Preprocessing page and everythign will be fine.**

## Components

![Component Model](https://github.com/Cardio-Meta-Microme/Cardio/tree/main/assets/component_diagram.png)

### Preprocessing Scripts

Data preprocessing inputs:

- log-transformed metabolites
- microbiome read-normalized counts
- patient health metrics (Age, BMI, sex, ID, health status)


Processing steps:

- read CSV files from data folder
- combine metadata, microbiome, and metabolome into one dataframe with patients as indices
- drop patients missing over 1000 features from model
- calculate each patient’s shannon diversity from microbe counts
- centered log ratio (CLR) transform counts to relative abundance:
    - we have to do this because compositional data is constrained by total 
    - image address: Aitchison_triadlogratio.jpg
    - formula:  $ clr(x) =  \ln\left[\frac{x_1}{g_m(x)}, \ldots, \frac{x_D}{g_m(x)}\right] $ where $ g_m(x) = (\prod\limits_{i=1}^{D} x_i)^{1/D} $ is the geometric mean of x 
- filter sparse features separately for microbiome/metabolome
    - sparse defined as having more than a certain number of NAs


Before ML feature selection and training: 

    impute NAs as lowest value in distribution -1

### Data Visualization

There are a several useful visualizations in this component. 

Firstly we visualise the general distributions of our cohort using Altair, this includes a boxplot of the BMI, age,
and shannon diversity of the sample.

Next we plot the the relative abundance of bacterial species among all four of the cohorts. These display the bacterial species or metabolites that are significantly changed from the healthy cohort. This change is deemed significant by a [Benjamini-Hochberg](https://link.springer.com/referenceworkentry/10.1007/978-1-4419-9863-7_1215) test.

Finally there are two Uniform Manifold Approximation Projections (UMAPs) which reduce the dimensionality

### Modelling



### User Input Predictions

### Patient status abbreviations:
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
