import pandas as pd
import os

def get_df():
    """
    Returns the dataframe files. In the future add a path parameter.

    Returns a pandas dataframes.
    """
    # For this to work in general don't use relative paths. Get cwd.
    path = os.getcwd()

    patient_general = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST9', header=1)
    microbiome_counts = pd.read_excel( path +'/data/raw_data.xlsx', sheet_name='ST10', header=1)
    gmm_kegg_abundances = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST11', header=1)
    serum_metabolites = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST12', header=1)

    return patient_general, microbiome_counts, gmm_kegg_abundances, serum_metabolites