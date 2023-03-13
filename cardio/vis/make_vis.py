import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import altair as alt
import statsmodels

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.cluster
import sklearn.inspection
import sklearn.impute
import sklearn.manifold

# Packages for visualisation
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from vega_datasets import data
from umap import UMAP
from statsmodels.stats.weightstats import ztest
from statsmodels.stats.multitest import multipletests

#Global variable for matplotlib fonts
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
plt.rc("figure", labelsize=BIGGER_SIZE)
plt.style.use("dark_background")

def process_for_visualization(df):
    """
    Processed data for visualization by changing chort names to more memorable titles
    Parameters
    ---------
    data, pandas df with columsn ID, Status, Age (years), BMI (kg/m²), Gender
    Returns
    -------
    general_data, pandas df with columns mentioned above and new cols sample_group, and sample_group_breaks
    """
    #check that the columns we need are present
    for col in ['Status', 'BMI (kg/m²)', 'Age (years)', 'Gender']:
        assert col in df.columns, f"Column {col} is not in the dataframe"

    #mapping sample names
    sample_name_w_breaks = {'HC275': 'Healthy\n n=275',
     'MMC269': 'Metabolically\n Matched\n n=269',
     'IHD372': 'Heart Disease\n n=372',
     'UMCC222': 'NoTx.\n Metabolically\n Matched\n n=222'}

    sample_name = {'HC275': 'Healthy n=275',
     'MMC269': 'Metabolically Matched n=269',
     'IHD372': 'Heart Disease n=372',
     'UMCC222': 'NoTx. Metabolically Matched n=222'}


    #mapping ID to these sample names
    df['sample_group_breaks'] = df['Status'].apply(lambda x: sample_name_w_breaks[x])
    df['sample_group'] = df['Status'].apply(lambda x: sample_name[x])

    general_data = df[['ID', 'Status', 'Age (years)', 'BMI (kg/m²)', 'Gender', 'sample_group', 'sample_group_breaks']]

    return general_data


def plot_general_dist_altair(df):
    """
    Make boxplots showing general characteristics of each cohort we are trying to classify
    Parameters
    ----------
    df, pandas df with cohort info with colnames sample_group, sample_group_breaks, BMI (kg/m²),
    Age (years), Gender
    Returns
    -------
    fig, altair figure object
    """

    #processing data for visualization
    df = process_for_visualization(df)

    #check that the columns we need are present
    for col in ['Status', 'BMI (kg/m²)', 'Age (years)', 'Gender']:
        assert col in df.columns

    #creating first boxplot of bmi using altair
    chart1 = alt.Chart(df).mark_boxplot().encode(
        alt.X('sample_group', title='', axis=alt.Axis(labels=False)),
        alt.Y('BMI (kg/m²)'),
        alt.Color('sample_group', legend=alt.Legend(title='Patient Group'))
    )

    #creating second boxplot of Age using altair
    chart2 = alt.Chart(df).mark_boxplot().encode(
        alt.X('sample_group', title='', axis=alt.Axis(labels=False)),
        alt.Y('Age (years)'),
        alt.Color('sample_group', legend=alt.Legend(title='Patient Group'))
    )
    #concatenating the charts using altair
    chart1_2 = alt.hconcat(chart1, chart2)

    #creating a df that groups by disease classification and counts the number of Males and Females
    gender_counts = df.groupby(['sample_group_breaks','Gender']).count()
    gender_counts.reset_index(inplace=True)
    #This just makes the title wrapping for the altair bar chart easier
    gender_counts['sample_group_breaks'] = gender_counts.sample_group_breaks.str.split('\n')

    #using altair to make a bar chart by gender for each cohort
    chart3 = alt.Chart(gender_counts).mark_bar().encode(
        alt.X('Gender:N', title=''),
        alt.Y('ID:Q', title='Patient Count'),
        alt.Color('Gender:N'),
        alt.Column('sample_group_breaks:N', header=alt.Header(labelAlign='center'), title='')
    ).properties(
        width=50)

    #horizontally concatenate each figure keeping the legends / colors independent
    fig = alt.hconcat(chart1_2, chart3).resolve_scale(color='independent')

    return fig

def ttest(hc, subgroup, microtype):
    """
    Evaluates a t test for each feature of the dataframe
    
        Parameters:
            hc(Pandas dataframe): subgroup of whole dataset for only the healthy control group
            subgroup(Pandas dataframe): subgroup of whole dataset for only one disease condition
            microtype(list): names of bacterial or metabolite species, matches the column names in the df
        Returns:
            pvals(list): p values of each t test
    """
    pvals = []
    for species in microtype:
        t, pval = statsmodels.stats.weightstats.ztest(hc[species].dropna(), subgroup[species].dropna())
        pvals.append(pval)
    return pvals

def multtest(hc, subgroup, microtype):
    """
    Evaluates a Benjamini/Hochberg test for each p value of the t test of each feature of the dataframe
    
        Parameters:
            hc(Pandas dataframe): subgroup of whole dataset for only the healthy control group
            subgroup(Pandas dataframe): subgroup of whole dataset for only one disease condition
            microtype(list): names of bacterial or metabolite species, matches the column names in the df
        Returns:
            siglist(list): boolean values for each feature if it has significantly deviated from the healthy control group
    """
    tlist = ttest(hc, subgroup, microtype)
    sigbool = statsmodels.stats.multitest.multipletests(tlist, method="fdr_bh")[0]
    siglist = dict(map(lambda i,j : (i,j) , microtype, sigbool))
    
    return siglist

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

def mk_df(micro_dict, control_dict, valtype, siglist):
    """
    Makes a dataframe of mean normalized read counts for either bacterial species or metabolites sorted based off the healthy control subtype

        Parameters:
            micro_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key)
            control_dict(dict): mean normalized read counts (values) for bacterial species or metabolites (key) for healthy control subtype
            valtype(string): either "Bacteria" or "Metabolites"
	    siglist(list): boolean values for each feature if significantly deviated from healthy
	Returns:
            micro_df(Pandas dataframe): sorted dataframe of normalized read counts for either bacterial species or metabolites
    """
    micro_dict = dict(sorted(micro_dict.items(), key=lambda kv: control_dict[kv[0]]))
    micro_type = list(micro_dict.keys())
    type_counts = list(micro_dict.values())

    for species in micro_dict.keys():
        if siglist[species]:
            micro_dict[species].append("Significant Change")
        else:
            micro_dict[species].append("No Significant Change")

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

    bacteria = [column for column in df.columns if 'CAG' in column and 'unclassified' not in column]
    metabolites = list(df.columns[339:1551])

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

    mmcsiglist = multtest(df_HC275, df_MMC269, mtype)
    mmcdict = mk_dict(mtype, df_MMC269)
    mmcdf = mk_df(mmcdict, hcdict, microtype, mmcsiglist)

    ihdsiglist = multtest(df_HC275, df_IHD372, mtype)
    ihddict = mk_dict(mtype, df_IHD372)
    ihddf = mk_df(ihddict, hcdict, microtype, ihdsiglist)

    umccsiglist = multtest(df_HC275, df_UMCC222, mtype)
    umccdict = mk_dict(mtype, df_UMCC222)
    umccdf = mk_df(umccdict, hcdict, microtype, umccsiglist)

    chart = mk_chart(hcdf, ihddf, mmcdf, umccdf, microtype)

    return chart

def high_var(df):
    """
    Returns 500 most variable features in dataframe

        Parameters:
            df(Pandas dataframe): entire dataframe
        Returns:
            hv_data(Pandas dataframe): only includes the 500 most variable features
    """
    high_var = df.var(axis=0).sort_values(ascending=False)[:500].index
    hv_data = df[high_var]

    return hv_data

def plot_UMAP(df, columns, hivar):
    """
    Creates a UMAP

        Parameters:
            df(Pandas dataframe): entire dataframe
            columns(str): either "all" or "default", specifies if you want to include age and BMI or not
            hivar(bool): True or False, specifies if you want to only include the high variable
        Returns:
            chart(Altair chart): clusters patients
    """
    bacteria = [column for column in df.columns if 'CAG' in column and 'unclassified' not in column]
    metabolites = list(df.columns[339:1551])
    default_modeling_columns = bacteria + metabolites
    all_modeling_columns = bacteria + metabolites + ['Age (years)', 'BMI (kg/m²)']

    # specify the dataframe at either all_modeling_columns or default_modeling_columns, depending on inclusion of age and BMI
    if columns == "all":
        X = df[all_modeling_columns]
    elif columns == "default":
        X = df[default_modeling_columns]

    if hivar:
        X = high_var(X)
    else:
        pass

    # replace NaN values with zero (do we want to impute with mean???)
    for col in X.columns:
        X[col] = np.where(X[col].isna(), 0, X[col])

    # Dimensionality
    reducer = UMAP()
    X = reducer.fit_transform(X)

    principal_df = pd.DataFrame(data=X, columns=['component_one', 'component_two'])
    final_df = pd.concat([principal_df, df[['Status']]], axis=1)

    chart = alt.Chart(final_df).mark_circle(size=10).encode(
                alt.X('component_one'),
                alt.Y('component_two'),
                color='Status',
                tooltip=['Status']
            ).interactive()

    return chart

def cluster_age_bmi(df):
    """
    Creates a scatter plot of age and BMI

        Parameters:
            df(Pandas dataframe): entire dataframe
        Returns:
            chart(Altair chart): clusters patients based off their age and BMI
    """
    chart = alt.Chart(df).mark_circle(size=15).encode(
                alt.X('Age (years)'),
                alt.Y('BMI (kg/m²)'),
                color='Status',
                tooltip=['Status']
            ).interactive()
    return chart

def feature_histograms(df, patient_data, features):
    """
    Takes in training data and patient data and plots patient values against the training data with IHD for
    certain list of predictive features
    Parameters
    ----------
    df, pandas df with the training data with columns 'Status'
    patient data, pandas series where each row is a feature to plot against training data
    features, list of features, should be columns in both patient data and df
    Returns
    ------
    fig, altair plot
    """

    #filtering data for IHD only
    df = df[df['Status'] == 'IHD372']
    #subsetting to features we're interested in
    df = df[features]
    patient_data = patient_data[features]

    #determining how big a plotting grid to make
    if len(features) % 5 == 0:
        rows = len(features) / 5
    else:
        rows = math.ceil(len(features) / 5)

    #creating said grid
    fig, axs = plt.subplots(int(rows), 5)
    fig.set_size_inches(20, 10)

    #so only the axis with plots show
    for ax in axs:
        for col in ax:
            col.set_axis_off()

    counter = 0
    #for loop creating plot for each feature
    for i, feature in enumerate(features):
        #resetting counter after each row is full
        if counter % 5 == 0:
            counter = 0
        else:
            pass
        #making kdeplot
        sns.kdeplot(df[feature], legend=False, fill=True, color='lightgray', ax=axs[i//5,counter])
        #adding vertical line for patient data
        axs[i//5,counter].axvline(patient_data.loc[feature], color='r')
        axs[i//5,counter].set_axis_on()
        axs[i//5,counter].set_ylabel('')
        counter += 1

    plt.suptitle('Predictive Features')
    fig.supylabel('Density', x=0)

    plt.tight_layout()
    return fig
