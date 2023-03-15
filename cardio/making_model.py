import pandas as pd
from sklearn.feature_selection import f_classif, GenericUnivariateSelect, RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

def load_data(path):
    """
    Load data from a pickle file and clean it by dropping any rows with NaN values and the Gender column.

    Parameters:
    -----------
    path : str
        Path to the pickle file containing the data.

    Returns:
    --------
    df : pandas.DataFrame
        Cleaned DataFrame of the data.
    """

    df = pd.read_pickle(path).copy()
    df = df.dropna(axis=0)
    del df['Gender']
    return df

def make_X_Y(df):
    """
    Create numpy arrays X and Y from a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.

    Returns:
    --------
    X : numpy.ndarray
        The input DataFrame's columns from 2 (inclusive) to second-to-last (exclusive).
    Y : numpy.ndarray
        A binary array with 1 if the row's Status is IHD372 or CIHD158, 0 otherwise.
    X_cols : numpy.ndarray
        The column labels of X.
    """

    df['IHD'] = ((df.Status == 'IHD372') ^ (df.Status == 'CIHD158')).astype(int)
    X_cols = df.columns[2:-1]
    Y_cols = ['IHD']
    
    X = df.loc[:, X_cols].values
    Y = df.loc[:, Y_cols].values.flatten()
    
    return X, Y, X_cols

def compute_metrics(true, pred):
    """
    Compute accuracy, precision and recall scores from true and predicted labels.

    Parameters:
    -----------
    true : numpy.ndarray
        The true labels.
    pred : numpy.ndarray
        The predicted labels.

    Returns:
    --------
    accuracy : float
        The accuracy score.
    precision : float
        The precision score.
    recall : float
        The recall score.
    """

    accuracy, precision, recall = round(accuracy_score(true, pred), 2),\
                                  round(precision_score(true, pred), 2),\
                                  round(recall_score(true, pred), 2)
            
    print('accuracy: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))
    return accuracy, precision, recall

def univariate_ftest_feature_subset(X, Y):
    """
    Select a subset of features from X using an univariate F-test.

    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix.
    Y : numpy.ndarray
        The target vector.

    Returns:
    --------
    keep_columns : numpy.ndarray
        A boolean array with True values for columns with F-test p-values less than 0.025.
    """

    print('selecting subset of features using univariate f-test')
    selection = GenericUnivariateSelect(f_classif).fit(X, Y)
    keep_columns = selection.pvalues_ < 0.025
    print('keeping {}/{} features with f-statistic p-values < 0.025'.format(sum(keep_columns), len(keep_columns)))
    return keep_columns
    # X = X[:, keep_columns]
    # X_columns = np.array(X_cols)[keep_columns]

def split_data(X, Y):
    """
    Split the data into training, validation and testing sets.

    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix.
    Y : numpy.ndarray
        The target vector.

    Returns:
    --------
    X_train : numpy.ndarray
        The training set features.
    Y_train : numpy.ndarray
        The training set targets.
    X_valid : numpy.ndarray
        The validation set features.
    Y_valid : numpy.ndarray
        The validation set targets.
    X_train_valid : numpy.ndarray
        The combined training and validation set features.
    Y_train_valid : numpy.ndarray
        The combined training and validation set targets.
    X_test : numpy.ndarray
        The test set features.
    Y_test : numpy.ndarray
        The test set targets.
    """

    print('splitting data into train, valid and test')
    X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(X, Y, test_size=0.05)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size=0.1)
    print('train size: {}, valid size: {}, test size: {}'.format(len(X_train), len(X_valid), len(X_test)))

    return X_train, Y_train, X_valid, Y_valid, X_train_valid, Y_train_valid, X_test, Y_test

def decision_tree(X, Y):
    """
    Train a decision tree classifier on X and Y.

    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix.
    Y : numpy.ndarray
        The target vector.

    Returns:
    --------
    basic : sklearn.tree.DecisionTreeClassifier
        A trained decision tree classifier.
    """

    basic = tree.DecisionTreeClassifier()
    basic.fit(X, Y)
    return basic

def RF(X, Y, n_trees):
    """
    Train a random forest classifier on X and Y.

    Parameters:
    -----------
    X : numpy.ndarray
        The feature matrix.
    Y : numpy.ndarray
        The target vector.
    n_trees : int
        The number of decision trees to use in the random forest.

    Returns:
    --------
    clf : sklearn.ensemble.RandomForestClassifier
        A trained random forest classifier.
    """

    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X, Y)
    return clf

def evaluate_model(model, X_eval, Y_eval):
    """Evaluates the performance of a given model on a given evaluation set and prints the accuracy, precision and recall 
    scores.

    Parameters:
    -----------
    model : sklearn estimator
        A trained estimator that will be used to make predictions on the evaluation set.
    X_eval : numpy array
        The feature matrix of the evaluation set.
    Y_eval : numpy array
        The target labels of the evaluation set.

    Returns:
    --------
    None
    """

    Yhat_eval = model.predict(X_eval)
    compute_metrics(Y_eval, Yhat_eval)

def reverse_selection_feature_subset(X, Y):
    """Selects a subset of features by recursively removing features with the least absolute correlation with the target
    variable. 

    Parameters:
    -----------
    X : numpy array
        A matrix of shape (n_samples, n_features) containing the input data.
    Y : numpy array
        A vector of shape (n_samples,) containing the target labels.

    Returns:
    --------
    keep_columns : numpy array
        A boolean mask that indicates which columns of X were selected to be kept.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = linear_model.LogisticRegression(max_iter=500, solver='liblinear')
    cv = StratifiedKFold(5)
    rfecv = RFECV(estimator=clf, step=25, cv=cv, scoring="accuracy", min_features_to_select=1)
    rfecv.fit(X_scaled, Y)
    return rfecv.support_

def hyperparam_optimize_n_trees(X_train, Y_train, X_valid, Y_valid, n_trees):
    """Optimizes the number of trees in a random forest classifier using the validation set. 

    Parameters:
    -----------
    X_train : numpy array
        The feature matrix of the training set.
    Y_train : numpy array
        The target labels of the training set.
    X_valid : numpy array
        The feature matrix of the validation set.
    Y_valid : numpy array
        The target labels of the validation set.
    n_trees : list
        A list of integers representing the number of trees to try.

    Returns:
    --------
    best_n_trees : int
        The number of trees that resulted in the highest accuracy on the validation set.
    """

    result_estimator = []
    for n_estimator in n_trees:
        clf = RF(X_train, Y_train, n_estimator)
        Yhat_valid = clf.predict(X_valid)
        print('Performance on validation set of Random Forest with {} trees'.format(n_estimator))
        accuracy, _, _ = compute_metrics(Y_valid, Yhat_valid)
        result_estimator.append((accuracy, n_estimator))

    print('N_Trees - Accuracy')
    for result, estimator in result_estimator:
        print(estimator, result)

    best = max(result_estimator)
    print('Best n_trees: {}'.format(best[1]))
    return best[1]


class RF_Classifier():
    def __init__(self, model_path, columns_path):
        """Initializes a RF_Classifier object.

        Parameters:
        -----------
        model_path : str
            The path to the saved model.
        columns_path : str
            The path to the saved list of column names.

        Returns:
        --------
        None
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.columns = pickle.load(open(columns_path, 'rb'))

    def classify(self, df):
        """Makes predictions on a given dataframe using the trained random forest model.

        Parameters:
        -----------
        df : pandas dataframe
            The dataframe to make predictions on.

        Returns:
        --------
        predictions : numpy array
            The predicted target labels for the input dataframe.
        """
        X = df.loc[:, self.columns].values
        predictions = self.model.predict(X)
        return predictions
    

if __name__ == '__main__':
    path = '../data/cleaned_data.pkl'
    print('loading data')
    df = load_data(path)

    X, Y, X_columns = make_X_Y(df)

    print('target is presence of IHD, codes IHD372 and CIHD158')
    print('dataset contains {} patients with IHD, {} patients without, {} patients total'.format(
        Y.sum(), len(Y) - Y.sum(), len(Y)))

    print('training data initially contains {} patients and {} features'.format(X.shape[0], X.shape[1]))

    keep_columns = univariate_ftest_feature_subset(X, Y)

    X = X[:, keep_columns]
    X_columns = np.array(X_columns)[keep_columns]

    X_train, Y_train, X_valid, Y_valid, X_train_valid, Y_train_valid, X_test, Y_test = split_data(X, Y)

    basic = decision_tree(X_train, Y_train)
    print('Performance on training set of basic decision tree classifier')
    evaluate_model(basic, X_train, Y_train)

    print('Performance on validation set of basic decision tree classifier')
    evaluate_model(basic, X_valid, Y_valid)

    print('running RFECV with 5-fold CV, step size of 25, using a logistic regression model as base')
    support = reverse_selection_feature_subset(X_train_valid, Y_train_valid)

    print('RFECV terminated finding {}/{} features useful'.format(sum(support), len(support)))

    X_train_subset = X_train[:, support]
    X_valid_subset = X_valid[:, support]
    X_train_valid_subset = X_train_valid[:, support]
    X_test_subset = X_test[:, support]
    X_subset = X[:, support]
    X_columns = X_columns[support]

    print('Now using random forest with a few numbers of estimators and using the features found by the RFECV')
    trees = [100, 250, 500, 1000, 5000, 10000]

    best_n_trees = hyperparam_optimize_n_trees(X_train_subset, Y_train, X_valid_subset, Y_valid, trees)

    print('Retraining random forest on entire train-valid set and testing on test set')

    clf = RF(X_train_valid_subset, Y_train_valid, best_n_trees)
    print('Performance on test set:')
    evaluate_model(clf, X_test_subset, Y_test)

    print('Retraining classifier on entire dataset and saving to pickle')
    clf = RF(X_subset, Y, best_n_trees)
    pickle.dump(clf, open('Trained_Production_RF_Classifier_230314.pkl', 'wb'))

    pickle.dump(X_columns, open('Trained_Production_RF_Classifier_features_230314.pkl', 'wb'))
