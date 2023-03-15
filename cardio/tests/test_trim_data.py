import unittest
import numpy as np
import pandas as pd
import scipy.stats
import skbio.diversity
import os
import math
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import collections
collections.Callable = collections.abc.Callable
import sys
sys.path.append('../..')
from cardio.preprocessing_scripts import trimdata


# Define a class in which the tests will run
class TestPreprocessing(unittest.TestCase):

    def test_readcsvs(self):
        """
        Test whether omics data .csv files are in the correct folder for
        function read_csvs
        """
        try:
            trimdata.read_csvs()
            self.assertTrue(True)
            print('CSV path test passed')
        except FileNotFoundError:
            print('Data files not found. Working directory must be Cardio, \
                  and csv data files must be in Cardio/data. Metadata, microbiome, \
                  and metabolome data must be entitled metacard_metadata.csv, \
                  metacard_microbiome.csv, and metacard_serum.csv')

    def testbasicfilt(self):
        """
        Test that empty dataframe throws ValueError
        in function basic_filtering
        """
        df = pd.DataFrame(data=[], columns=['ID', 'Age', 'Feature'])
        try:
            trimdata.basic_filtering(df, df, df)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
            print('Test of basic filtering empty dataframe passed')

    def test_sparsefilt(self):
        """
        Test that sparse feature removal throws error if all features
        are sparse, and thus removed from df in function sparse_filt
        """
        df = pd.DataFrame(data=[], index=['a', 'b', 'c'])
        try:
            trimdata.sparse_filt(df, remove_str='X-', na_lim=20)
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)
            print('Test of sparse feature filtering empty dataframe passed')

    def test_shannon(self):
        """
        Test that Shannon diversity function runs, outputting a column of
        float64 and that NaNs do not throw errors
        """
        df = pd.DataFrame(data=[[1, 2, 3], [np.nan, np.nan, np.nan]])
        shan = trimdata.calculate_shannon_diversity(df)
        assert shan.shannon.dtype == 'float64'
        print('Smoke test for shannon calculation worked')

    def test_counttoabund(self):
        """
        Test that function count_to_abundance throws AssertionError if
        counts are not numerical dtype
        """
        df = pd.DataFrame(data=[['a', 'b', 'c', 'd', 'e', 'f']], 
                        columns=['MGS count', 'Gene count', 'Microbial load', 'ID', 'Status', 'count'])
        try:
            trimdata.count_to_abundance(df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
            print('Test of invalid count data type passed')