import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import GenericUnivariateSelect, RFECV, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """
    Load data from a pickle file and clean it by dropping any patient
    with NaN values and the Gender column.

        Parameters
            path : str
                Path to the pickle file containing the data.

        Returns
            data_frame : pandas.DataFrame
                Cleaned DataFrame of the data.
    """

    data_frame = pd.read_pickle(path).copy()
    data_frame = data_frame.dropna(axis=0)
    del data_frame['Gender']
    return data_frame

def make_x_y(data_frame):
    """
    Create numpy arrays X and Y from a pandas DataFrame.

        Parameters
            data_frame : pandas.DataFrame
                The input DataFrame.

        Returns
            X : numpy.ndarray
                The input DataFrame's columns from 2 (inclusive)
                to second-to-last (exclusive).
            Y : numpy.ndarray
                A binary array with 1 if the patient's Status is
                IHD372 or CIHD158, 0 otherwise.
            x_cols : numpy.ndarray
                The column labels of X.
    """

    data_frame['IHD'] = ((data_frame.Status == 'IHD372') ^ (data_frame.Status == 'CIHD158')).astype(int)
    x_cols = data_frame.columns[2:-1]
    y_cols = ['IHD']

    X = data_frame.loc[:, x_cols].values
    Y = data_frame.loc[:, y_cols].values.flatten()

    return X, Y, x_cols

def compute_metrics(true, pred):
    """
    Compute accuracy, precision and recall scores from true and
    predicted labels.

        Parameters
            true : numpy.ndarray
                The true labels.
            pred : numpy.ndarray
                The predicted labels.

        Returns
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
    Select a subset of features using a univariate F-test.

        Parameters
            X : numpy.ndarray
                The feature matrix.
            Y : numpy.ndarray
                The target vector.

        Returns
            keep_columns : numpy.ndarray
                A boolean array with True values for columns with
                F-test p-values less than 0.025.
    """

    print('selecting subset of features using univariate f-test')
    selection = GenericUnivariateSelect(f_classif).fit(X, Y)
    keep_columns = selection.pvalues_ < 0.025
    print('keeping {}/{} features with f-statistic p-values < 0.025'.format(
        sum(keep_columns), len(keep_columns)))
    return keep_columns
    # X = X[:, keep_columns]
    # x_columns = np.array(x_cols)[keep_columns]

def split_data(X, Y):
    """
    Split the data into training, validation and testing sets.

        Parameters
            X : numpy.ndarray
                The feature matrix.
            Y : numpy.ndarray
                The target vector.

        Returns
            x_train : numpy.ndarray
                The training set features.
            y_train : numpy.ndarray
                The training set targets.
            x_valid : numpy.ndarray
                The validation set features.
            y_valid : numpy.ndarray
                The validation set targets.
            x_train_valid : numpy.ndarray
                The combined training and validation set features.
            y_train_valid : numpy.ndarray
                The combined training and validation set targets.
            x_test : numpy.ndarray
                The test set features.
            y_test : numpy.ndarray
                The test set targets.
    """

    print('splitting data into train, valid and test')
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(X, Y, test_size=0.05)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.1)
    print('train size: {}, valid size: {}, test size: {}'.format(
        len(x_train), len(x_valid), len(x_test)))

    return x_train, y_train, x_valid, y_valid, x_train_valid, y_train_valid, x_test, y_test

def evaluate_model(model, x_eval, y_eval):
    """Evaluates the performance of a given model on a given evaluation set and
       prints the accuracy, precision and recall scores.

        Parameters
            model : sklearn estimator
                A trained estimator that will be used to make predictions on the evaluation set.
            x_eval : numpy array
                The feature matrix of the evaluation set.
            y_eval : numpy array
                The target labels of the evaluation set.

        Returns
            accuracy : float
                The accuracy score.
            precision : float
                The precision score.
            recall : float
                The recall score.
    """

    Yhat_eval = model.predict(x_eval)
    return compute_metrics(y_eval, Yhat_eval)

def reverse_selection_feature_subset(X, Y):
    """Selects a subset of features by recursively removing features
       using the RFECV scikitlearn class. 

        Parameters
            X : numpy array
                A matrix of shape (n_samples, n_features) containing the input data.
            Y : numpy array
                A vector of shape (n_samples,) containing the target labels.

        Returns
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

def hyperparam_optimize_n_trees(x_train, y_train, x_valid, y_valid, n_trees):
    """Optimizes the number of trees in a random forest classifier using the validation set. 

        Parameters
            x_train : numpy array
                The feature matrix of the training set.
            y_train : numpy array
                The target labels of the training set.
            x_valid : numpy array
                The feature matrix of the validation set.
            y_valid : numpy array
                The target labels of the validation set.
            n_trees : list
                A list of integers representing the number of trees to try.

        Returns
            best_n_trees : int
                The number of trees that resulted in the highest accuracy on the validation set.
    """

    result_estimator = []
    for n_estimator in n_trees:
        clf = RandomForestClassifier(n_estimator).fit(x_train, y_train)
        Yhat_valid = clf.predict(x_valid)
        print('Performance on validation set of Random Forest with {} trees'.format(n_estimator))
        accuracy, _, _ = compute_metrics(y_valid, Yhat_valid)
        result_estimator.append((accuracy, n_estimator))

    print('N_Trees - Accuracy')
    for result, estimator in result_estimator:
        print(estimator, result)

    best = max(result_estimator)
    print('Best n_trees: {}'.format(best[1]))
    return best[1]


def plot_precision_recall_curve(y_true, y_proba, filepath):
    """Plots and saves to file a precision-recall curve.

        Parameters:
            y_true (array-like): True binary labels.
            y_proba (array-like): Predicted probabilities.
            filepath (str): File path to save the plot.

        Returns:
            None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    sns.set(style="darkgrid")
    plt.style.use("dark_background")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.savefig(filepath)


def plot_confusion_matrix(y_true, y_pred, filepath):
    """Plots and saves to file a confusion matrix.

        Parameters:
            y_true (array-like): True binary labels.
            y_pred (array-like): Predicted binary labels.
            filepath (str): File path to save the plot.

        Returns:
            None
    """
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    _, ax = plt.subplots(figsize=(8, 6))
    sns.set(style="darkgrid")
    plt.style.use("dark_background")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(filepath)



class RF_Classifier():
    def __init__(self, model_path, columns_path, fill_nas_path):
        """Initializes a RF_Classifier object.

            Parameters:
                model_path : str
                    The path to the saved model.
                columns_path : str
                    The path to the saved list of column names.

            Returns:
                None
        """
        self.model = pickle.load(open(model_path, 'rb'))
        self.columns = pickle.load(open(columns_path, 'rb'))
        self.na_fill = pickle.load(open(fill_nas_path, 'rb'))

    def classify(self, data_frame):
        """Makes predictions on a given dataframe using the trained random forest model.

            Parameters:
                data_frame : pandas dataframe
                    The dataframe to make predictions on.

            Returns:
                predictions : numpy array
                    The predicted target labels for the input dataframe.
        """
        X = data_frame[self.columns].fillna(self.na_fill, axis=0).values
        predictions = self.model.predict(X)
        return predictions


if __name__ == '__main__':
    path = '../../data/cleaned_data.pkl'
    print('loading data')
    data_frame = load_data(path)

    X, Y, x_columns = make_x_y(data_frame)

    print('target is presence of IHD, codes IHD372 and CIHD158')
    print('dataset contains {} patients with IHD, {} patients without, {} patients total'.format(
        Y.sum(), len(Y) - Y.sum(), len(Y)))

    print('training data initially contains {} patients and {} features'.format(X.shape[0], X.shape[1]))

    keep_columns = univariate_ftest_feature_subset(X, Y)

    X = X[:, keep_columns]
    x_columns = np.array(x_columns)[keep_columns]

    x_train, y_train, x_valid, y_valid, x_train_valid, y_train_valid, x_test, y_test = split_data(X, Y)

    basic = tree.DecisionTreeClassifier().fit(x_train, y_train)
    print('Performance on training set of basic decision tree classifier')
    evaluate_model(basic, x_train, y_train)

    print('Performance on validation set of basic decision tree classifier')
    evaluate_model(basic, x_valid, y_valid)

    print('running RFECV with 5-fold CV, step size of 25, using a logistic regression model as base')
    support = reverse_selection_feature_subset(x_train_valid, y_train_valid)

    print('RFECV terminated finding {}/{} features useful'.format(sum(support), len(support)))

    x_train_subset = x_train[:, support]
    x_valid_subset = x_valid[:, support]
    x_train_valid_subset = x_train_valid[:, support]
    x_test_subset = x_test[:, support]
    X_subset = X[:, support]
    x_columns = x_columns[support]

    print('Now using random forest with a few numbers of estimators and using the features found by the RFECV')
    trees = [100, 250, 500, 1000, 5000, 10000]

    best_n_trees = hyperparam_optimize_n_trees(x_train_subset, y_train, x_valid_subset, y_valid, trees)

    print('Retraining random forest on entire train-valid set and testing on test set')

    clf = RandomForestClassifier(best_n_trees).fit(x_train_valid_subset, y_train_valid)
    print('Performance on test set:')
    evaluate_model(clf, x_test_subset, y_test)

    print('plotting test set performance')
    probabilities = clf.predict_proba(x_test_subset)[:, 1]
    predictions = clf.predict(x_test_subset)
    plot_precision_recall_curve(y_test, probabilities, 'precision_recall_curve.png')
    plot_confusion_matrix(y_test, predictions, 'confusion_matrix.png')

    print('computing feature importance on the test set')
    result = permutation_importance(clf, x_test_subset, y_test, n_repeats=100, n_jobs=6)
    importances = result.importances_mean
    features_sorted = x_columns[np.argsort(importances)[::-1]]
    importances_sorted = importances[np.argsort(importances)[::-1]]
    pickle.dump([features_sorted, importances_sorted], open('features_importances_sorted.pkl', 'wb'))
    print('top 10 most important features:')
    print('feature - importance')
    for feature, importance in zip(features_sorted[:10], importances_sorted[:10]):
        print(feature, '-', importance)


    print('Retraining classifier on entire dataset and saving to pickle')
    clf = RandomForestClassifier(best_n_trees).fit(X_subset, Y)
    pickle.dump(clf, open('Trained_Production_RF_Classifier_230314.pkl', 'wb'))
    pickle.dump(x_columns, open('Trained_Production_RF_Classifier_features_230314.pkl', 'wb'))
