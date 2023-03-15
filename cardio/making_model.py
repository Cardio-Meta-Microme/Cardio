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
    df = pd.read_pickle(path).copy()
    df = df.dropna(axis=0)
    del df['Gender']
    return df

def make_X_Y(df):
    df['IHD'] = ((df.Status == 'IHD372') ^ (df.Status == 'CIHD158')).astype(int)
    X_cols = df.columns[2:-1]
    Y_cols = ['IHD']
    
    X = df.loc[:, X_cols].values
    Y = df.loc[:, Y_cols].values.flatten()
    
    return X, Y, X_cols

def compute_metrics(true, pred):
    accuracy, precision, recall = round(accuracy_score(true, pred), 2),\
                                  round(precision_score(true, pred), 2),\
                                  round(recall_score(true, pred), 2)
            
    print('accuracy: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))
    return accuracy, precision, recall

def univariate_ftest_feature_subset(X, Y):
    print('selecting subset of features using univariate f-test')
    selection = GenericUnivariateSelect(f_classif).fit(X, Y)
    keep_columns = selection.pvalues_ < 0.025
    print('keeping {}/{} features with f-statistic p-values < 0.025'.format(sum(keep_columns), len(keep_columns)))
    return keep_columns
    # X = X[:, keep_columns]
    # X_columns = np.array(X_cols)[keep_columns]

def split_data(X, Y):
    print('splitting data into train, valid and test')
    X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(X, Y, test_size=0.05)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size=0.1)
    print('train size: {}, valid size: {}, test size: {}'.format(len(X_train), len(X_valid), len(X_test)))

    return X_train, Y_train, X_valid, Y_valid, X_train_valid, Y_train_valid, X_test, Y_test

def decision_tree(X, Y):
    basic = tree.DecisionTreeClassifier()
    basic.fit(X, Y)
    return basic

def RF(X, Y, n_trees):
    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X, Y)
    return clf

def evaluate_model(model, X_eval, Y_eval):
    Yhat_eval = model.predict(X_eval)
    compute_metrics(Y_eval, Yhat_eval)

def reverse_selection_feature_subset(X, Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = linear_model.LogisticRegression(max_iter=500, solver='liblinear')
    cv = StratifiedKFold(5)
    rfecv = RFECV(estimator=clf, step=25, cv=cv, scoring="accuracy", min_features_to_select=1)
    rfecv.fit(X_scaled, Y)
    return rfecv.support_

def hyperparam_optimize_n_trees(X_train, Y_train, X_valid, Y_valid, n_trees):
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
        self.model = pickle.load(open(model_path, 'rb'))
        self.columns = pickle.load(open(columns_path, 'rb'))

    def classify(self, df):
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
