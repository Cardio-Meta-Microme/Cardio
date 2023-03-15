"""
This script reads csv files, calculates microbe relative abundance and alpha
diversity, filters out rare features (present in > some % samples).
"""

import pandas as pd
import numpy as np
import scipy.stats
import skbio.diversity
import os
import math
import warnings
import pickle
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def read_csvs():
    """
    Read raw data csv files from Cardio/data and merge into one large dataframe

        Parameters:
            None
        Returns:
            all_joined(Pandas dataframe): dataframe containing metadata,
            microbiome, and metabolome for each patient
            metadata_columns(column index object): column names of metadata
            microbiome_columns(column index object): column names of microbes
            metabolome_columns(column index object): column names of
            metabolites
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

        Parameters:
            df(Pandas dataframe): entire dataset
            microbiome_columns(column index object):
            metabolome_columns(column index object):

        Returns:
            df(Pandas dataframe): entire dataset with null patients dropped
            and zeros as NaNs
            df_meta(Pandas dataframe): dataframe of metabolite values (already
            log transformed)
            df_micro(Pandas dataframe): dataframe of microbiome counts
    """
    # drop patients missing measurements for over 1000 features
    df[df == 0] = float('nan')
    drop_patients = pd.isna(df).sum(axis=1) < 1000
    df = df.loc[drop_patients]

    if df.shape[0] == 0:
        raise ValueError('Empty dataframe: all patients removed due to lack \
                         of features.')
    else:
        pass

    # get separate dataframes for micro/metabo
    df_meta = df.loc[:, metabolome_columns]
    df_micro = df.loc[:, microbiome_columns]

    return df, df_meta, df_micro


def calculate_shannon_diversity(data):
    """
    Calculates shannon (alpha) diversity index for each patient

        Parameters:
            data(Pandas dataframe): microbiome df where each row is a patient
            and .X is their microbiome count data

        Returns:
            shannon_diversity(Pandas df): 1-column df where each index is a
            patient and the column is their shannon diversity value
    """

    shannon_diversity = pd.DataFrame()
    null_shannon = []
    for ind in data.index:
        shannon_diversity.loc[ind,'shannon'] = skbio.diversity.alpha.shannon(data.loc[ind])
        if math.isnan(shannon_diversity.loc[ind,'shannon']):
            null_shannon.append(ind)
        else:
            pass
    if len(null_shannon) > 0:
        print('Shannon diversity could not be calculated for the following indices: ',
              null_shannon)
    return shannon_diversity


def count_to_abundance(df):
    """
    Transform microbiome sequencing count data to relative abundance with
    Center-log ratio (CLR) transformation.

        Parameters:
            df(Pandas dataframe): microbiome counts per species with metadata
            columns MGS count, Gene count, Microbial load, ID, and Status
        Returns:
            abund(Pandas dataframe): microbe center-log ratio abundances per
            species with metadata columns ID and Status
    """

    counts = df.drop(columns=['MGS count', 'Gene count', 'Microbial load'])
    counts.set_index(['ID', 'Status'], inplace=True)

    # assert that all count columns are dtype int
    assert counts.iloc[:, 0].dtype == 'float64'

    counts['gmean'] = counts.apply(scipy.stats.mstats.gmean,
                                   axis=1, nan_policy='omit')


    abund = np.log(counts.divide(counts['gmean'], axis=0))
    abund.drop(columns='gmean', inplace=True)
    abund.reset_index(inplace=True)
    return abund


def sparse_filt(df, remove_str, na_lim):
    """
    Filtering out sparse and uncharacterized features in a data subset

        Parameters:
            df(Pandas dataframe): dataframe with feature columns and patients
            as indices
            remove_str(string): identifier of uncharacterized columns (e.g.
            unknown metabolites or species) - either 'X-' or 'unclassified'
            here
            na_lim(int): number of acceptable nas in feature column

        Returns:
            df(Pandas dataframe): trimmed dataframe
    """
    keep_columns = pd.isna(df).sum(axis=0) < na_lim
    df = df.loc[:, keep_columns]
    keep_columns = [col for col in df.columns if remove_str not in col]
    df = df.loc[:, keep_columns]

    if df.shape[1] == 0:
        print('N/A limit (max. acceptable NAs per column): ', na_lim)
        raise ValueError('Oops! You have removed all features from your dataframe! \
                        This likely means all of your columns surpassed the NA limit \
                        above.')
    else:
        pass
    return df


def preprocess():
    """
    Main preprocessing function that runs all others.

        Parameters: None

        Returns:
            df_metamicro(Pandas dataframe): processed dataframe
            micro_cols(list): feature columns of microbial species
            metabo_cols(list): feature coumns of metabolites
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

    fill_values = df_metamicro[df_metamicro.columns[5:]].min(axis=0) - 1
    filled = df_metamicro[df_metamicro.columns[5:]].fillna(fill_values, axis=0)

    pickle.dump(fill_values, open('cardio/model/na_fill_values.pkl', 'wb'))

    df_metamicro = pd.concat((df_metamicro[df_metamicro.columns[:5]], filled), axis=1)

    return df_metamicro, micro_cols, metabo_cols

