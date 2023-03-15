#this will test visualization module
"""importing python modules and modules from make_vis.py"""
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import altair as alt
import unittest
import math
import collections
import os

#adding the current path for imports
sys.path.append('../../cardio')

from vis import make_vis
from preprocessing_scripts import trimdata


#Reading testing data
TEST_DATA = pd.read_csv('../../data/processed_data.csv')
DATA_DIR = '../../data'
MICROBIOME = pd.read_csv(DATA_DIR + '/metacard_microbiome.csv')
MICROBIOME_COL = MICROBIOME.columns

METABOLOME = pd.read_csv(DATA_DIR + '/metacard_serum.csv')
METABOLOME_COL = METABOLOME.columns


class ProcessForVisualization(unittest.TestCase):

    """Class for testing process_for_visualization"""

    def test_col_names(self):
        """Test whether col names are in the test_df"""
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        try:
            make_vis.process_for_visualization(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_study_groups(self):
        """Test whether the right patient groups are there"""
        test_df = TEST_DATA.copy()
        test_df['Status'] = np.where(test_df['Status'] == 'HC275', 'HC276', test_df['Status'])
        try:
            make_vis.process_for_visualization(test_df)
            self.assertTrue(False)
        except KeyError:
            self.assertTrue(True)

    def test_output(self):
        """Test the output has the right columns"""
        test_df = TEST_DATA.copy()
        test_output = make_vis.process_for_visualization(test_df)

        col_names = ['ID', 'Status', 'Age', 'BMI', 'Gender', 'shannon',\
                    'sample_group', 'sample_group_breaks']

        for col in col_names:
            assert col in test_output.columns


class PlotGeneralDistAltair(unittest.TestCase):

    """Class for testing altair plot"""

    def test_col_names(self):
        """Test whether right col names are in input dataframe"""
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        try:
            make_vis.plot_general_dist_altair(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_input_type(self):
        """Test whether the input is a pandas df"""
        test_df = TEST_DATA.copy()
        test_col = test_df['Status']
        try:
            make_vis.plot_general_dist_altair(test_col)
        except AssertionError:
            self.assertTrue(True)


class FeatureHistograms(unittest.TestCase):

    """Class for testing feature histograms"""

    def test_status(self):
        """test whether status is in columns"""
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        rand_features = list(test_df.columns.to_series().sample(20).index)
        try:
            make_vis.feature_histograms(test_df, test_df.iloc[0], rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_patient_features_in_test_col(self):
        """test whether features are present in the training data column"""
        test_df = TEST_DATA.copy()
        rand_features = list(test_df.columns.to_series().sample(20).index)
        patient = test_df.iloc[0]
        test_df = test_df.drop([rand_features[0]], axis=1)

        try:
            make_vis.feature_histograms(test_df, patient, rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_patient_features_in_patient_col(self):
        """test whether features are in patient col"""
        test_df = TEST_DATA.copy()
        rand_features = list(test_df.columns.to_series().sample(20).index)
        patient = test_df.iloc[0]
        patient = patient.drop([rand_features[0]])

        try:
            make_vis.feature_histograms(test_df, patient, rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

class TestTTest(unittest.TestCase):

    def test_length_match(self):
        """Tests that the list returned in t test is the same length as the list of features in input"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(math.isclose(len(microcols),len(make_vis.ttest(df_HC275, df_MMC269, microcols))), "Length of p values does not match length of input columns")


class TestMultTest(unittest.TestCase):

    def test_length_match(self):
        """Tests that the list returned in multtest is the same length as the list of features in input"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(math.isclose(len(microcols),len(make_vis.multtest(df_HC275, df_MMC269, microcols))), "Length of p values does not match length of input columns")

    def test_key_match(self):
        """Tests that multtest is using the same indices as the input list"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(set(microcols) == set(make_vis.multtest(df_HC275, df_MMC269, microcols).keys()), "The keys of p value list from multtest do not match original keys")


class TestMkDict(unittest.TestCase):

    def test_length_match(self):
        """Tests that returned dictionary has same width as the length of the columns provided"""
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(math.isclose(len(microcols),len(make_vis.mk_dict(microcols, df))), "Length of dictionary does not match length of input columns")

    def test_key_match(self):
        """Tests that the input columns match the dictionary made my mk_dict"""
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(set(microcols) == set(make_vis.mk_dict(microcols, df).keys()), "The features provided do not match the dictionary keys")


class TestMkControlDf(unittest.TestCase):

    def test_sorted(self):
        """Tests that the mk_control_df correctly sorts the dictionary by the values"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        cdf = make_vis.mk_control_df(make_vis.mk_dict(microcols, df_HC275), "Bacteria")
        cdfs = sorted(cdf['Normalized Read Count'])
        self.assertTrue((cdf['Normalized Read Count'] == cdfs).all(), "Values are not sorted correctly")

    def test_key_match(self):
        """Tests that mk_control_df uses same indices as the list provided"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        self.assertTrue(math.isclose(len(microcols),len(make_vis.mk_control_df(make_vis.mk_dict(microcols, df_HC275), "Bacteria"))), "Length of sorted df does not match length of input columns")


class TestMkDf(unittest.TestCase):

    def test_sorted(self):
        """Tests that mk_df is ordering the clinical subgroup by the same order used in the healthy control group"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]

        hcdict = make_vis.mk_dict(microcols, df_HC275)

        mmcsiglist = make_vis.multtest(df_HC275, df_MMC269, microcols)
        mmcdict = make_vis.mk_dict(microcols, df_MMC269)

        self.assertTrue((mmcdict.keys() == hcdict.keys(), "Subgroup is not sorted according to healthy group"))

    def test_key_match(self):
        """Tests that mk_df is assigning significance based on p-values computed earlier"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]

        hcdict = make_vis.mk_dict(microcols, df_HC275)

        mmcsiglist = make_vis.multtest(df_HC275, df_MMC269, microcols)
        mmcdict = make_vis.mk_dict(microcols, df_MMC269)
        mmcdf = make_vis.mk_df(mmcdict, hcdict, "Bacteria", mmcsiglist)
        self.assertTrue((mmcdf.columns == ['Bacteria', 'Normalized Read Count', 'Change from Healthy Control']).all(), "Columns of df incorrect")


class TestMkChart(unittest.TestCase):

    def test_sorted(self):
        """Tests that mk_chart is creating an altair stacked chart"""
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        df_IHD372 = df[df["Status"] == "IHD372"]
        df_UMCC222 = df[df["Status"] == "UMCC222"]

        hcdict = make_vis.mk_dict(microcols, df_HC275)
        hcdf = make_vis.mk_control_df(hcdict, "Bacteria")

        mmcsiglist = make_vis.multtest(df_HC275, df_MMC269, microcols)
        mmcdict = make_vis.mk_dict(microcols, df_MMC269)
        mmcdf = make_vis.mk_df(mmcdict, hcdict, "Bacteria", mmcsiglist)

        ihdsiglist = make_vis.multtest(df_HC275, df_IHD372, microcols)
        ihddict = make_vis.mk_dict(microcols, df_IHD372)
        ihddf = make_vis.mk_df(ihddict, hcdict, "Bacteria", ihdsiglist)

        umccsiglist = make_vis.multtest(df_HC275, df_UMCC222, microcols)
        umccdict = make_vis.mk_dict(microcols, df_UMCC222)
        umccdf = make_vis.mk_df(umccdict, hcdict, "Bacteria", umccsiglist)

        self.assertTrue(isinstance(make_vis.mk_chart(hcdf, ihddf, mmcdf, umccdf, "Bacteria"), alt.vegalite.v4.api.VConcatChart), "Chart type is incorrect")


class TestPlotMicroAbundance(unittest.TestCase):

    def test_microtype_input(self):
        """Tests that user is entering microtype specification as a string"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, 3, microcols, metabcols)
            self.assertTrue(False, "Allowing non-string microtypes")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_bacteria_type_input(self):
        """Tests that user is entering microcols specification as a list"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", 3, metabcols)
            self.assertTrue(False, "Allowing non-list bacteria columns")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_metabolite_type_input(self):
        """Tests that user is entering metabcols specification as a list"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", microcols, 5)
            self.assertTrue(False, "Allowing non-list metabolites columns")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_bacteria_input(self):
        """Tests that user is specifying micrcols which exist in the entered dataframe"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", ['thing1', 'thing2'], metabcols)
            self.assertTrue(False, "Allowing bacteria columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)

    def test_metabolite_input(self):
        """Tests that user is specifying metabcols which exist in the entered dataframe"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Metabolite", microcols, ['thing1', 'thing2'])
            self.assertTrue(False, "Allowing metabolite columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)

class TestHighVar(unittest.TestCase):

    def test_shape(self):
        """Tests that high_var is returning 500 most variable features"""
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(math.isclose(make_vis.high_var(df).shape[1],500), "Not returning 500 features")

class TestPlotUMAP(unittest.TestCase):

    def test_column_input(self):
        """Tests that user is entering columns specification as a string"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, 2, False, microcols, metabcols)
            self.assertTrue(False, "Allowing non-string columns")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_hivar_input(self):
        """Tests that user is entering hivar specification as a boolean"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", 3, microcols, metabcols)
            self.assertTrue(False, "Allowing non-bool hivar")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_bacteria_type_input(self):
        """Tests that user is entering micrcols specification as a list"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, 1, metabcols)
            self.assertTrue(False, "Allowing non-list bacteria columns")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_metabolite_type_input(self):
        """Tests that user is entering metabcols specification as a list"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, microcols, 8)
            self.assertTrue(False, "Allowing non-list metabolites columns")
        except (TypeError) as err:
            self.assertTrue(True)

    def test_bacteria_input(self):
        """Tests that user is specifying microcols which exist in the entered dataframe"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True,  ['thing1', 'thing2'], metabcols)
            self.assertTrue(False, "Allowing bacteria columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)

    def test_metabolite_input(self):
        """Tests that user is specifying metabcols which exist in the entered dataframe"""
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, microcols, ['thing1', 'thing2'])
            self.assertTrue(False, "Allowing metabolite columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)


class TestClusterAgeBMI(unittest.TestCase):

    def test_output_type(self):
        """Tests that cluster_age_bmi is returning an altair plot"""
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(type(make_vis.cluster_age_bmi(df)) == alt.vegalite.v4.api.Chart, "Returned type is not an altair plot")
