#this will test visualization module
import sys
import unittest
import pandas as pd
sys.path.append('../../cardio')

from vis.make_vis import process_for_visualization, plot_general_dist_altair, \
                        feature_histograms

#PENDING
TEST_DATA = pd.read_csv('../../data/preprocessed.csv')

class ProcessForVisualization(unittest.TestCase):

    def test_col_names(self):
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        try:
            process_for_visualization(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_study_groups(self):
        test_df = TEST_DATA.copy()
        test_df['Status'] = np.where(test_df['Status'] == 'HC275', 'HC276', test_df['Status'])
        try:
            process_for_visualization(test_df)
            self.assertTrue(False)
        except KeyError:
            self.assertTrue(True)

    def test_output(self):
        test_df = TEST_DATA.copy()
        test_output = process_for_visualization(test_df)

        col_names = ['ID', 'Status', 'Age (years)', 'BMI (kg/m²)', 'Gender', 'sample_group', 'sample_group_breaks']

        for col in col_names:
            assert col in test_output.columns


class PlotGeneralDistAltair(unittest.TestCase):

    def test_col_names(self):
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        try:
            plot_general_dist_altair(test_df)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_input_type(self):
        test_df = TEST_DATA.copy()
        test_col = test_df['Status']
        try:
            plot_general_dist_altair(test_col)
        except AssertionError:
            self.assertTrue(True)


class FeatureHistograms(unittest.TestCase):

    def test_status(self):
        test_df = TEST_DATA.copy().drop(['Status'], axis=1)
        rand_features = list(test_df.columns.to_series().sample(20).index)
        try:
            feature_histograms(test_df, test_df.iloc[0], rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_patient_features_in_test_col(self):
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
        test_df = TEST_DATA.copy()
        rand_features = list(test_df.columns.to_series().sample(20).index)
        patient = test_df.iloc[0]
        patient = patient.drop([rand_features[0]])

        try:
            feature_histograms(test_df, patient, rand_features)
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)
