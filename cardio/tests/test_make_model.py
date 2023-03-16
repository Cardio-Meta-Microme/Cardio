import sys
import unittest
sys.path.append('../..')

import numpy as np

from cardio.model.making_model import load_data, make_x_y, compute_metrics, univariate_ftest_feature_subset
from cardio.model.making_model import split_data, evaluate_model, reverse_selection_feature_subset
from cardio.model.making_model import hyperparam_optimize_n_trees, RF_Classifier
from sklearn.ensemble import RandomForestClassifier

DATAPATH = 'data/cleaned_data.pkl'
MODELPATH = 'cardio/model/Trained_Production_RF_Classifier_230314.pkl'
COLUMNPATH = 'cardio/model/Trained_Production_RF_Classifier_features_230314.pkl'
NAPATH = 'cardio/model/na_fill_values.pkl'

DF = load_data(DATAPATH)


class TestModelingFunctions(unittest.TestCase):
    """
    Class for testing modeling functions
    """
    def test_load_data(self):
        """Test load_data function"""
        df = load_data(DATAPATH)
        assert 'Status' in df.columns

    def test_make_x_y(self):
        """test make_X_Y function"""
        X, Y, X_cols = make_x_y(DF.copy())
        assert len(X) == len(Y)

    def test_compute_metrics(self):
        """test compute_metrics function"""
        acc, prec, rec = compute_metrics(np.array([0, 0, 1]), np.array([0, 0, 1]))
        assert acc == 1
        assert prec == 1
        assert rec == 1

    def test_univariate_ftest(self):
        """test univariate_ftest_feature_subset function"""
        X, Y, X_cols = make_x_y(DF.copy())
        keep_columns = univariate_ftest_feature_subset(X, Y)
        assert sum(keep_columns) < len(keep_columns)
        assert len(keep_columns) == len(X_cols)

    def test_split_data(self):
        """test split_data function"""
        X, Y, X_cols = make_x_y(DF.copy())
        X_train, Y_train, X_valid, Y_valid, X_train_valid, Y_train_valid, X_test, Y_test = split_data(X, Y)
        assert len(X_train) == len(Y_train)
        assert len(X_train) + len(X_valid) == len(X_train_valid)

    def test_evaluate_model(self):
        """test evalute_model function"""
        X, Y, X_cols = make_x_y(DF.copy())
        keep_columns = univariate_ftest_feature_subset(X, Y)
        X = X[:, keep_columns]
        model = RandomForestClassifier(100).fit(X, Y)
        acc, prec, rec = evaluate_model(model, X, Y)
        assert acc > 0.5

    def test_reverse_selection_feature_subset(self):
        """test reverse_selection_feature_subset function"""
        X, Y, X_cols = make_x_y(DF.copy())
        keep_columns = univariate_ftest_feature_subset(X, Y)
        X = X[:, keep_columns[:100]]
        support = reverse_selection_feature_subset(X, Y)
        assert len(support) > 0
        assert sum(support) <= len(support)

    def test_hyperparm_optimize(self):
        """test hyperparam_optimize_n_trees function"""
        X, Y, X_cols = make_x_y(DF.copy())
        X_train, Y_train, X_valid, Y_valid, X_train_valid, Y_train_valid, X_test, Y_test = split_data(X, Y)
        n_trees = [3, 25]
        best = hyperparam_optimize_n_trees(X_train, Y_train, X_valid, Y_valid, n_trees)
        assert best == 25

suite = unittest.TestLoader().loadTestsFromTestCase(TestModelingFunctions)
_ = unittest.TextTestRunner().run(suite)


class Test_RF_Classifier_Class(unittest.TestCase):
    """Class for testing the RF_Classifier Class"""
    
    def test_init(self):
        """Tests initialization of the RF_Classifier class"""
        model = RF_Classifier(MODELPATH, COLUMNPATH, NAPATH)
        assert True
    
    def test_classify(self):
        """test classification using the RF_classifier class"""
        model = RF_Classifier(MODELPATH, COLUMNPATH, NAPATH)
        predictions = model.classify(DF.iloc[:10])
        assert len(predictions) == 10

suite = unittest.TestLoader().loadTestsFromTestCase(Test_RF_Classifier_Class)
_ = unittest.TextTestRunner().run(suite)




    




        
