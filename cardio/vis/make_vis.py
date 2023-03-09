import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.cluster
import sklearn.inspection
import sklearn.impute
import sklearn.manifold

from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from vega_datasets import data
from umap import UMAP

SMALL_SIZE = 10
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def process_for_visualization(data):

    #mapping sample names
    sample_name = {'HC275': 'Healthy\n n=275',
     'MMC269': 'Metabolically\n Matched\n n=269',
     'IHD372': 'Heart Disease\n n=372',
     'UMCC222': 'NoTx. Metabolically\n Matched\n n=222'}


    #mapping ID to these sample names
    data['sample_group'] = data['Status'].apply(lambda x: sample_name[x])

    general_data = data[['ID', 'Status', 'Age (years)', 'BMI (kg/m²)', 'Gender', 'sample_group']]

    return general_data


def plot_general_dist(df):
    """
    Make boxplots overlayed with scatter to show distribution of column names in
    Parameters
    ----------
    df, pandas df with cohort info with colnames sample_group, BMI (kg/m²), Age (years), Gender
    Returns
    -------
    fig, matplotlib figure object
    """

    #processing data for visualization
    df = process_for_visualization(df)

    for col in ['sample_group', 'BMI (kg/m²)', 'Age (years)', 'Gender']:
        assert col in df.columns

    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(20, 10)

    sns.boxplot(x='sample_group', y='BMI (kg/m²)', data=df, ax=axs[0])
    axs[0].set(xlabel='', title='BMI (kg/m²) by Cohort')

    sns.boxplot(x='sample_group', y='Age (years)', data=df, ax=axs[1])
    axs[1].set(xlabel='', title='Age by Cohort')

    gender_counts = df.groupby(['sample_group','Gender']).count()
    gender_counts.reset_index(inplace=True)

    sns.barplot(x='sample_group', y='Age (years)',hue='Gender', data=gender_counts, ax=axs[2])
    axs[2].set(xlabel='', title='Gender Count by Cohort')

    return fig

def mk_dict(colnames, df):
    """
    Makes a dictionary of mean normalized read counts for either bacterial species or metabolites for a dataframe
    
        Parameters:
            colnames(list): names of bacterial species, matches the column names in the df
            df(Pandas dataframe): subgroup of whole dataset for only one disease condition
        Returns:
            micro_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key)
    """
    micro_dict = {}
    if df['Status'].eq('HC275').any():
        for point in df[colnames].columns:
            micro_dict[point] = df[point].mean()
    else:
        for point in df[colnames].columns:
            micro_dict[point] = [df[point].mean()]
    
    return micro_dict

def mk_control_df(micro_dict, valtype):
    """
    Makes a sorted dataframe of mean normalized read counts for either bacterial species or metabolites ONLY for the healthy control disease subtype
    
        Parameters:
            micro_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key)
            valtype(string): either "Bacteria" or "Metabolites"
        Returns:
            micro_dict(Pandas dataframe): sorted dataframe of normalized read counts for either bacterial species or metabolites
    """
    micro_dict = dict(sorted(micro_dict.items(), key=lambda x:x[1]))
    micro_type = list(micro_dict.keys())
    type_counts = list(micro_dict.values())

    micro_dict = pd.DataFrame(micro_dict, index=[0]).T.reset_index()
    micro_dict.rename(columns={'index': valtype, 0: 'Normalized Read Count'}, inplace=True)
    
    return micro_dict

def mk_df(micro_dict, control_dict, valtype):
    """
    Makes a dataframe of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype
    
        Parameters:
            micro_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key)
            control_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key) for healthy control subtype
            valtype(string): either "Bacteria" or "Metabolites"
        Returns:
            micro_df(Pandas dataframe): sorted dataframe of normalized read counts for either bacterial species or metabolites
    """
    micro_dict = dict(sorted(micro_dict.items(), key=lambda kv: control_dict[kv[0]]))
    micro_type = list(micro_dict.keys())
    type_counts = list(micro_dict.values())
    
    for species in micro_dict.keys():
        if np.sign(micro_dict[species]) == np.sign(control_dict[species]):
            micro_dict[species].append("No Significant Change")
        else:
            micro_dict[species].append("Significant Change")
    
    micro_df = pd.DataFrame(micro_dict).T.reset_index()
    micro_df.rename(columns={'index': valtype, 0: 'Normalized Read Count', 1: 'Change from Healthy Control'}, inplace=True)
    
    return micro_df

def mk_chart(HCDF, IHDDF, MMCDF, UMCCDF, valtype):
    """
    Makes a chart of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype for all four subgroups
    
        Parameters:
            HCDF(Pandas dataframe): subset of dataframe information for healthy controls
            IHDDF(Pandas dataframe): subset of dataframe information for IHD group
            MMCDF(Pandas dataframe): subset of dataframe information for MMC group
            UMCCDF(Pandas dataframe): subset of dataframe information for UMCC
            valtype(string): either "Bacteria" or "Metabolites"
        Returns:
            chart(Altair chart): chart of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype for all four subgroups
    """
    domain = ['No Significant Change', 'Significant Change']
    range_ = ['grey', 'red']
    
    chartHC = alt.Chart(HCDF, title='Healthy Controls').mark_bar(size=1).encode(
    alt.X('Normalized Read Count'),
    alt.Y(valtype, sort=None, axis=alt.Axis(labels=False, tickSize=0)),
    color=alt.value('grey'),
    tooltip=[valtype, 'Normalized Read Count']
    ).properties(
    width=300,
    height=300
    ).interactive() 
    
    chartUMCC = alt.Chart(UMCCDF, title='Untreated Metabolically Matched Controls').mark_bar(size=1).encode(
    alt.X('Normalized Read Count'),
    alt.Y(valtype, sort=None, axis=alt.Axis(labels=False, tickSize=0)),
    color=alt.Color('Change from Healthy Control', scale=alt.Scale(domain=domain, range=range_)),
    tooltip=[valtype, 'Normalized Read Count']
    ).properties(
    width=300,
    height=300
    ).interactive()
    
    chartIHD = alt.Chart(IHDDF, title='Ischemic Heart Disease').mark_bar(size=1).encode(
    alt.X('Normalized Read Count'),
    alt.Y(valtype, sort=None, axis=alt.Axis(labels=False, tickSize=0)),
    color=alt.Color('Change from Healthy Control', scale=alt.Scale(domain=domain, range=range_)),
    tooltip=[valtype, 'Normalized Read Count']
    ).properties(
    width=300,
    height=300
    ).interactive() 
    
    chartMMC = alt.Chart(MMCDF, title='Metabolically Matched Controls').mark_bar(size=1).encode(
    alt.X('Normalized Read Count'),
    alt.Y(valtype, sort=None, axis=alt.Axis(labels=False, tickSize=0)),
    color=alt.Color('Change from Healthy Control', scale=alt.Scale(domain=domain, range=range_)),
    tooltip=[valtype, 'Normalized Read Count']
    ).properties(
    width=300,
    height=300
    ).interactive()
    
    chart = alt.vconcat(alt.hconcat(chartHC, chartIHD),alt.hconcat(chartMMC, chartUMCC))
    
    return chart

def plot_micro_abundance(df, microtype):
    """
    Wrapper function to generate a chart of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype for all four subgroups
    
        Parameters:
            df(Pandas dataframe): entire dataframe
            microtype(string): either "Bacteria" or "Metabolites"
        Returns:
            chart(Altair chart): chart of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype for all four subgroups
    """
    if microtype == "Bacteria":
        mtype = bacteria
    elif microtype == "Metabolite":
        mtype = metabolites
    
    df_HC275 = df[df["Status"] == "HC275"]
    df_MMC269 = df[df["Status"] == "MMC269"]
    df_IHD372 = df[df["Status"] == "IHD372"]
    df_UMCC222 = df[df["Status"] == "UMCC222"]
    
    hcdict = mk_dict(mtype, df_HC275)
    hcdf = mk_control_df(hcdict, microtype)

    mmcdict = mk_dict(mtype, df_MMC269)
    mmcdf = mk_df(mmcdict, hcdict, microtype)

    ihddict = mk_dict(mtype, df_IHD372)
    ihddf = mk_df(ihddict, hcdict, microtype)

    umccdict = mk_dict(mtype, df_UMCC222)
    umccdf = mk_df(umccdict, hcdict, microtype)

    chart = mk_chart(hcdf, ihddf, mmcdf, umccdf, microtype)
    
    return chart
