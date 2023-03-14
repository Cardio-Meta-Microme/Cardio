"""
This script reads csv files, calculates microbe relative abundance and alpha
diversity, filters out rare features (present in > some % samples).
"""

import pandas as pd
import numpy as np
import scipy.stats
import skbio.diversity
import os
import json

def read_csvs():
    """
    Read raw data csv files and merge into one large dataframe
    """
    # Get the current working directory to use absolute paths.
    wdir = os.getcwd()

    metadata = pd.read_csv(wdir + "/data" + '/metacard_metadata.csv')
    metadata = metadata[['ID', 'Status', 'Age (years)', 'BMI (kg/m²)', 'Gender']]
    metadata.columns = ['ID', 'Status', 'Age', 'BMI', 'Gender']
    metadata_columns = metadata.columns

    microbiome = pd.read_csv(wdir + "/data" + '/metacard_microbiome.csv')
    microbiome_columns = microbiome.columns

    metabolome = pd.read_csv(wdir + "/data" + '/metacard_serum.csv')
    metabolome_columns = metabolome.columns

    all_joined = metadata.merge(microbiome, how='inner',
                                on=['ID', 'Status']).merge(metabolome,
                                                           how='inner',
                                                           on=['ID', 'Status'])

    return all_joined, metadata_columns, microbiome_columns, metabolome_columns


def basic_filtering(df, microbiome_columns, metabolome_columns):
    """
    Drop patients missing most measurements, replace zeros with NaN, get
    separate dfs for microbiome/metabolome
    """
    # drop patients missing measurements for over 1000 features
    df[df == 0] = float('nan')
    drop_patients = pd.isna(df).sum(axis=1) < 1000
    df = df.loc[drop_patients]
    
    # get separate dataframes for micro/metabo
    df_meta = df.loc[:, metabolome_columns]
    df_micro = df.loc[:, microbiome_columns]

    return df, df_meta, df_micro


def calculate_shannon_diversity(data):
    """
    Calculates shannon index for each patient
    
    Parameters
    ----------
    data, pandas df where each row is a patient and .X is their microbiome
    abundance data

    Returns
    -------
    shannon_diversity, pandas df where each index is a patient and the
    column is their shannon diversity value
    """
    shannon_diversity = pd.DataFrame()
    for ind in data.index:
        shannon_diversity.loc[ind,'shannon'] = skbio.diversity.alpha.shannon(data.loc[ind])
    return shannon_diversity


def count_to_abundance(df):
    """
    Transform microbiome sequencing count data to relative abundance with 
    Center-log ratio (CLR) transformation.
    """

    counts = df.drop(columns=['MGS count', 'Gene count', 'Microbial load'])
    counts.set_index(['ID', 'Status'], inplace=True)
    counts['gmean'] = counts.apply(scipy.stats.mstats.gmean,
                                   axis=1, nan_policy='omit')

    abund = np.log(counts.divide(counts['gmean'], axis=0))
    abund.drop(columns='gmean', inplace=True)
    abund.reset_index(inplace=True)
    return abund


def sparse_filt(df, remove_str, na_lim):
    """
    Filtering out sparse and uncharacterized features in a data subset
    -----------------------------------------------
    Inputs: dataframe, identifier of unchar. columns (str), number of
    acceptable nas in col (int)
    -----------------------------------------------
    Outputs: trimmed dataframe
    """
    keep_columns = pd.isna(df).sum(axis=0) < na_lim
    df = df.loc[:, keep_columns]
    keep_columns = [col for col in df.columns if remove_str not in col]
    df = df.loc[:, keep_columns]
    return df


def preprocess():
    """
    Main preprocessing function that runs all others. Saves pkl files in
    root/data
    No inputs
    Outputs: processed dataframe, microbe columns, metabolite columns
    """

    raw_data, metadata_cols, microbiome_cols, metabolome_cols = read_csvs()
    basicfilt_data, df_meta, df_micro = basic_filtering(raw_data, 
                                                        microbiome_cols,
                                                        metabolome_cols)
    abundance = count_to_abundance(df_micro)

    # filtering based off of feature sparsity
    df_meta_filt = sparse_filt(df_meta, remove_str='X-', na_lim=20)
    df_micro_filt = sparse_filt(abundance, remove_str='unclassified',
                                na_lim=500)

    # calculate shannon diversity from raw microbiome counts
    df_divers = calculate_shannon_diversity(df_micro.drop
                                            (columns=['MGS count',
                                                      'Gene count',
                                                      'Microbial load', 'ID',
                                                      'Status']).fillna(0))

    # merge together metadata, filtered metabolites, and filtered abundances
    df_basic = basicfilt_data[metadata_cols].merge(df_divers, how='inner',
                                                   right_index=True,
                                                   left_index=True)
    df_metabs = df_basic.merge(df_meta_filt, how='inner',
                               on=['ID', 'Status'])
    df_metamicro = df_metabs.merge(df_micro_filt, how='inner', 
                                   on=['ID', 'Status'])

    # drop duplicate patients
    df_metamicro.set_index(['ID', 'Status'], inplace=True)
    df_metamicro.drop_duplicates(keep='first', inplace=True)
    df_metamicro.reset_index(inplace=True)

    # update microbiome and metabolome columns
    micro_cols = list(df_metamicro.columns[df_metamicro.columns.isin(microbiome_cols[2:])])
    metabo_cols = list(df_metamicro.columns[df_metamicro.columns.isin(metabolome_cols[2:])])

    # output micro/metabo columns to json
    json.dump(micro_cols, open('./data/microbiome_columns.json', 'w'))
    json.dump(metabo_cols, open('./data/metabolite_columns.json', 'w'))
    return df_metamicro

if __name__ == "__main__":
    preprocess()