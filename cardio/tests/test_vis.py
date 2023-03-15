#this will test visualization module
"""importing python modules and modules from make_vis.py"""
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import statsmodels
import unittest
import math
import collections

import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.gaussian_process
import sklearn.cluster
import sklearn.inspection
import sklearn.impute
import sklearn.manifold

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
from umap import UMAP
from cardio.vis import make_vis
from vis.make_vis import process_for_visualization, plot_general_dist_altair, \
			feature_histograms

collections.Callable = collections.abc.Callable


#adding the current path for imports
sys.path.append('../../cardio')

#Reading testing data
TEST_DATA = pd.read_csv('../../data/processed_data.csv')
print(TEST_DATA)

class ProcessForVisualization(unittest.TestCase):

    """Class for testing process_for_visualization"""

    def test_col_names(self):
        """Test whether col names are in the test_df"""
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        try:
            process_for_visualization(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_study_groups(self):
        """Test whether the right patient groups are there"""
        test_df = TEST_DATA.copy()
        test_df['Status'] = np.where(test_df['Status'] == 'HC275', 'HC276', test_df['Status'])
        try:
            process_for_visualization(test_df)
            self.assertTrue(False)
        except KeyError:
            self.assertTrue(True)

    def test_output(self):
        """Test the output has the right columns"""
        test_df = TEST_DATA.copy()
        test_output = process_for_visualization(test_df)

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
            plot_general_dist_altair(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_input_type(self):
        """Test whether the input is a pandas df"""
        test_df = TEST_DATA.copy()
        test_col = test_df['Status']
        try:
            plot_general_dist_altair(test_col)
        except AssertionError:
            self.assertTrue(True)


class FeatureHistograms(unittest.TestCase):

    """Class for testing feature histograms"""

    def test_status(self):
        """test whether status is in columns"""
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        rand_features = list(test_df.columns.to_series().sample(20).index)
        try:
            feature_histograms(test_df, test_df.iloc[0], rand_features)
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
            feature_histograms(test_df, patient, rand_features)
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
            feature_histograms(test_df, patient, rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

class TestTTest(unittest.TestCase):
        
    def test_length_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(math.isclose(len(microcols),len(make_vis.ttest(df_HC275, df_MMC269, microcols))), "Length of p values does not match length of input columns")
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestTTest)
_ = unittest.TextTestRunner().run(suite)

class TestMultTest(unittest.TestCase):
        
    def test_length_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(math.isclose(len(microcols),len(make_vis.multtest(df_HC275, df_MMC269, microcols))), "Length of p values does not match length of input columns")
        
    def test_key_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        self.assertTrue(set(microcols) == set(make_vis.multtest(df_HC275, df_MMC269, microcols).keys()))
    
suite = unittest.TestLoader().loadTestsFromTestCase(TestMultTest)
_ = unittest.TextTestRunner().run(suite)

class TestMkDict(unittest.TestCase):
        
    def test_length_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(math.isclose(len(microcols),len(make_vis.mk_dict(microcols, df))), "Length of dictionary does not match length of input columns")
        
    def test_key_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(set(microcols) == set(make_vis.mk_dict(microcols, df).keys()))
    
suite = unittest.TestLoader().loadTestsFromTestCase(TestMkDict)
_ = unittest.TextTestRunner().run(suite)

class TestMkControlDf(unittest.TestCase):
        
    def test_sorted(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        cdf = make_vis.mk_control_df(make_vis.mk_dict(microcols, df_HC275), "Bacteria")
        cdfs = sorted(cdf['Normalized Read Count'])
        self.assertTrue((cdf['Normalized Read Count'] == cdfs).all(), "Values are not sorted correctly")
        
    def test_key_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(math.isclose(len(microcols),len(make_vis.mk_control_df(make_vis.mk_dict(microcols, df_HC275), "Bacteria"))), "Length of sorted df does not match length of input columns")
    
suite = unittest.TestLoader().loadTestsFromTestCase(TestMkControlDf)
_ = unittest.TextTestRunner().run(suite)

class TestMkDf(unittest.TestCase):
        
    def test_sorted(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        
        hcdict = make_vis.mk_dict(microcols, df_HC275)
        
        mmcsiglist = make_vis.multtest(df_HC275, df_MMC269, microcols)
        mmcdict = make_vis.mk_dict(microcols, df_MMC269)
        
        self.assertTrue((mmcdict.keys() == hcdict.keys(), "Subgroup is not sorted according to healthy group"))
        
    def test_key_match(self):
        df, microcols, metabcols = trimdata.preprocess()
        df_HC275 = df[df["Status"] == "HC275"]
        df_MMC269 = df[df["Status"] == "MMC269"]
        
        hcdict = make_vis.mk_dict(microcols, df_HC275)
        
        mmcsiglist = make_vis.multtest(df_HC275, df_MMC269, microcols)
        mmcdict = make_vis.mk_dict(microcols, df_MMC269)
        mmcdf = make_vis.mk_df(mmcdict, hcdict, "Bacteria", mmcsiglist)
        self.assertTrue((mmcdf.columns == ['Bacteria', 'Normalized Read Count', 'Change from Healthy Control']).all(), "Columns of df incorrect")
             
suite = unittest.TestLoader().loadTestsFromTestCase(TestMkDf)
_ = unittest.TextTestRunner().run(suite)

class TestMkChart(unittest.TestCase):
        
    def test_sorted(self):
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
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestMkChart)
_ = unittest.TextTestRunner().run(suite)

class TestPlotMicroAbundance(unittest.TestCase):
        
    def test_microtype_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, 3, microcols, metabcols)
            self.assertTrue(False, "Allowing non-string microtypes")
        except (TypeError) as err:
            self.assertTrue(True)
            
    def test_bacteria_type_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", 3, metabcols)
            self.assertTrue(False, "Allowing non-list bacteria columns")
        except (TypeError) as err:
            self.assertTrue(True)
        
    def test_metabolite_type_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", microcols, 5)
            self.assertTrue(False, "Allowing non-list metabolites columns")
        except (TypeError) as err:
            self.assertTrue(True)
            
    def test_bacteria_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Bacteria", ['thing1', 'thing2'], metabcols)
            self.assertTrue(False, "Allowing bacteria columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)
    
    def test_metabolite_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_micro_abundance(df, "Metabolite", microcols, ['thing1', 'thing2'])
            self.assertTrue(False, "Allowing metabolite columns which do not exist in the df")
        except (KeyError) as err:
            self.assertTrue(True)
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotMicroAbundance)
_ = unittest.TextTestRunner().run(suite)

class TestHighVar(unittest.TestCase):
        
    def test_shape(self):
        df, microcols, metabcols = trimdata.preprocess()
        self.assertTrue(math.isclose(make_vis.high_var(df).shape[1],500), "Not returning 500 features")
    
suite = unittest.TestLoader().loadTestsFromTestCase(TestHighVar)
_ = unittest.TextTestRunner().run(suite)

class TestPlotUMAP(unittest.TestCase):
        
    def test_column_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, 2, False, microcols, metabcols)
            self.assertTrue(False, "Allowing non-string columns")
        except (TypeError) as err:
            self.assertTrue(True)
            
    def test_hivar_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", 3, microcols, metabcols)
            self.assertTrue(False, "Allowing non-bool hivar")
        except (TypeError) as err:
            self.assertTrue(True)
            
    def test_bacteria_type_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, 1, metabcols)
            self.assertTrue(False, "Allowing non-list bacteria columns")
        except (TypeError) as err:
            self.assertTrue(True)
        
    def test_metabolite_type_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, microcols, 8)
            self.assertTrue(False, "Allowing non-list metabolites columns")
        except (TypeError) as err:
            self.assertTrue(True)
            
    def test_bacteria_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True,  ['thing1', 'thing2'], metabcols)
            self.assertTrue(False, "Allowing bacteria columns which do not exist in the df")
        except (TypeError) as err:
            self.assertTrue(True)
    
    def test_metabolite_input(self):
        df, microcols, metabcols = trimdata.preprocess()
        try:
            make_vis.plot_UMAP(df, "all", True, microcols, ['thing1', 'thing2'])
            self.assertTrue(False, "Allowing metabolite columns which do not exist in the df")
        except (TypeError) as err:
            self.assertTrue(True)
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotUMAP)
_ = unittest.TextTestRunner().run(suite)
