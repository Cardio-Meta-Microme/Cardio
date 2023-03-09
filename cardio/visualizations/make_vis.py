import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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