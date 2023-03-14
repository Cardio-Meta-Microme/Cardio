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


print('loading data')
df = pd.read_pickle('../data/cleaned_data.pkl').copy()
df = df.dropna(axis=0)
del df['Gender']

print('making target column')
df['IHD'] = ((df.Status == 'IHD372') ^ (df.Status == 'CIHD158')).astype(int)
print('target is presence of IHD, codes IHD372 and CIHD158')
print('dataset contains {} patients with IHD, {} patients without, {} patients total'.format(
        df.IHD.sum(), len(df.IHD) - df.IHD.sum(), len(df.IHD)))

X_cols = df.columns[2:-1]
Y_cols = ['IHD']

X = df.loc[:, X_cols].values
Y = df.loc[:, Y_cols].values.flatten()

print('training data initially contains {} patients and {} features'.format(X.shape[0], X.shape[1]))


def compute_metrics(true, pred):
    accuracy, precision, recall = round(accuracy_score(true, pred), 2),\
                                  round(precision_score(true, pred), 2),\
                                  round(recall_score(true, pred), 2)
            
    print('accuracy: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))
    return accuracy, precision, recall

# we begin with a lot of features so starting by subsetting columns using Univariate f-test

print('selecting subset of features using univariate f-test')
selection = GenericUnivariateSelect(f_classif).fit(X, Y)
keep_columns = selection.pvalues_ < 0.025
print('keeping {}/{} features with f-statistic p-values < 0.025'.format(sum(keep_columns), len(keep_columns)))
X = X[:, keep_columns]
X_columns = np.array(X_cols)[keep_columns]

# Splitting data
print('splitting data into train, valid and test')
X_train_valid, X_test, Y_train_valid, Y_test = train_test_split(X, Y, test_size=0.05)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_valid, Y_train_valid, test_size=0.1)
print('train size: {}, valid size: {}, test size: {}'.format(len(X_train), len(X_valid), len(X_test)))


# basic decision tree
basic = tree.DecisionTreeClassifier()
basic.fit(X_train, Y_train)
Yhat_train = basic.predict(X_train)
print('Performance on training set of basic decision tree classifier')
compute_metrics(Y_train, Yhat_train)

Yhat_valid = basic.predict(X_valid)
print('Performance on validation set of basic decision tree classifier')
compute_metrics(Y_valid, Yhat_valid)

# Decision tree is able to get decent performance. Let's now use reverse feature 
# selection - cv, with logistic regression as our model.

scaler = StandardScaler()
X_train_valid_scaled = scaler.fit_transform(X_train_valid)
X_test_scaled = scaler.transform(X_test)
clf = linear_model.LogisticRegression(max_iter=500, solver='liblinear')
cv = StratifiedKFold(5)
print('running RFECV with 5-fold CV, step size of 25, using a logistic regression model as base')
rfecv = RFECV(estimator=clf, step=25, cv=cv, scoring="accuracy", min_features_to_select=1)
rfecv.fit(X_train_valid_scaled, Y_train_valid)
print('RFECV terminated finding {}/{} features useful'.format(sum(rfecv.support_), len(rfecv.support_)))
support = rfecv.support_

X_train_subset = X_train[:, support]
X_valid_subset = X_valid[:, support]
X_train_valid_subset = X_train_valid[:, support]
X_test_subset = X_test[:, support]
X_subset = X[:, support]
X_columns = X_columns[support]


print('Now using random forest with a few numbers of estimators\
      and using the features found by the RFECV')
result_estimator = []
for n_estimators in [100, 250, 500, 1000, 5000, 10000]:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train_subset, Y_train)
    Yhat_valid = clf.predict(X_valid_subset)
    print('Performance on validation set of Random Forest with {} trees'.format(n_estimators))
    accuracy, _, _ = compute_metrics(Y_valid, Yhat_valid)
    result_estimator.append((accuracy, n_estimators))

print('N_Trees - Accuracy')
for result, estimator in result_estimator:
    print(estimator, result)

best = max(result_estimator)
print('Best n_trees: {}'.format(best[1]))

print('Retraining random forest on entire train-valid set and testing on test set')
clf = RandomForestClassifier(n_estimators=best[1])
clf.fit(X_train_valid_subset, Y_train_valid)
Yhat_test = clf.predict(X_test_subset)
print('Performance on test set:')
compute_metrics(Y_test, Yhat_test)

print('Retraining classifier on entire dataset and saving to pickle')
clf = RandomForestClassifier(n_estimators=best[1])
clf.fit(X_subset, Y)
pickle.dump(clf, open('Trained_Production_RF_Classifier_230314.pkl', 'wb'))

pickle.dump(X_columns, open('Trained_Production_RF_Classifier_features_230314.pkl', 'wb'))



