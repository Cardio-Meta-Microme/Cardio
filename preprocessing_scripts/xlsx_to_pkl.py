import pandas as pd
import os

"""
This script reads the excel spreadsheet and writes the two tables that we are interested in, 
    ST9: general patient information
    ST10: patient microbiome counts
    ST11: patient GMM and KEGG abundances
    ST12: patient log-transformed serum metabolites
to pandas dataframe pickles. The merged dataframe is also 
"""
# For this to work in general don't use relative paths. Get cwd.
path = os.getcwd()

patient_general = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST9', header=1)
microbiome_counts = pd.read_excel( path +'/data/raw_data.xlsx', sheet_name='ST10', header=1)
gmm_kegg_abundances = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST11', header=1)
serum_metabolites = pd.read_excel(path + '/data/raw_data.xlsx', sheet_name='ST12', header=1)


patient_general.to_pickle(path + '/data/patient_general_metadata.pkl')
microbiome_counts.to_pickle(path + '/data/microbiome_counts.pkl')
gmm_kegg_abundances.to_pickle(path + '/data/gmm_kegg_abundances.pkl')
serum_metabolites.to_pickle(path + '/data/serum_metabolites.pkl')

all_data = patient_general.merge(microbiome_counts).merge(gmm_kegg_abundances).merge(serum_metabolites)
all_data.to_pickle(path + '/data/merged_dataframe.pkl')