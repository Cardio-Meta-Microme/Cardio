![example event parameter](https://github.com/Cardio-Meta-Microme/Cardio/tree/main/.github/workflows/main.yml/badge.svg?event=push)

# Cardio
Tool for Analysis of meta microbiomes and metabolomes.


=======

## Components

![cox interaction model](https://github.com/Cardio-Meta-Microme/Cardio/tree/main/assets/cox_interaction_model.png)

#### Preprocessing Scripts
 - 'trimdata' is the main function. Can be called using the 'trimdata.preprocess()' method.
    - This takes a pandas dataframe (read internally) and removes columns that have fewer than a threshold value (default is columns with fewer than 20% prevalence).
    - It transforms the raw microbiome counts into abundance.
    - It then calculates the Shannon diversity on the microbiome
    - The function returns the transformed microbiome abundance and metabolome data in a merged pandas dataframe.
#### Data Visualization

There are a several useful visualizations in this component. 

Firstly we visualise the general distributions of our cohort using Altair, this includes a boxplot of the BMI, age,
and shannon diversity of the sample.

Next we plot the the relative abundance of bacterial species among all four of the cohorts. These display the bacterial species or metabolites that are significantly changed from the healthy cohort. This change is deemed significant by a [Benjamini-Hochberg](https://link.springer.com/referenceworkentry/10.1007/978-1-4419-9863-7_1215) test.

Finally there are two Uniform Manifold Approximation Projections (UMAPs) which reduce the dimensionality

#### Modelling

#### User Input Predictions

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
