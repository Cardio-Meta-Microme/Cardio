import pandas as pd
import numpy as np
import scipy.stats

"""
This script reads csv files, calculates microbe relative abundance, and
filters out rare features (present in > some % samples).
"""


def count_to_abundance(df):
    """
    Transform microbiome sequencing count data to relative abundance with 
    Center-log ratio (CLR) transformation.
    """

    counts = df.drop(columns=['MGS count', 'Gene count', 'Microbial load'])
    counts.set_index(['ID', 'Status'], inplace=True)

    counts += 1.0
    counts['gmean'] = counts.apply(scipy.stats.mstats.gmean, axis=1, nan_policy='omit')

    abund = np.log(counts.divide(counts['gmean'], axis=0))
    abund.drop(columns='gmean', inplace=True)
    abund.reset_index(inplace=True)
    return abund


def filter_sparse(df, feature_cols, percent):
    '''
    filters sparse features (microbes or metabolites) so that we only consider
    those present in at least (percent) % of samples
    '''
    sparse = df[feature_cols].isnull().sum() / df.shape[0] > (1.0 - percent)
    feature_cols = sparse[~sparse].index
    df.drop(columns=sparse[sparse].index, inplace=True)
    return df


def combine_metamicro(metadata, metabs, micros):
    """
    
    """
    ids = metadata[['ID', 'Status', 'Age (years)', 'BMI (kg/m²)', 'Gender']]
    ids_metab = ids.merge(metabs, how='inner', on=['ID', 'Status'])
    ids_metab_micro = micros.merge(ids_metab, how='inner', on=['ID', 'Status'])
    ids_metab_micro.set_index(['ID', 'Status'], inplace=True)
    ids_metab_micro.drop_duplicates(keep='first', inplace=True)
    ids_metab_micro.dropna(axis=0, how='all', inplace=True)
    ids_metab_micro.reset_index(inplace=True)
    return ids_metab_micro

def run_example():
    # import csv files from current directory - we can change this
    metadata = pd.read_csv('metacard_metadata.csv')
    microbiome = pd.read_csv('metacard_microbiome.csv')
    metabolome = pd.read_csv('metacard_serum.csv')

    # transform and filter microbiome dataframe
    abundance = count_to_abundance(microbiome)

    # merge into one large dataframe by patient ID, filter sparse features
    metamicro = combine_metamicro(metadata, metabolome, abundance)
    metamicro_filt = filter_sparse(metamicro, metamicro.columns[2:], percent=0.25)
    # this is where we could remove X-metabolites

    # send final df to pickle
    metamicro_filt.to_pickle('./metamicro_processed.pkl')