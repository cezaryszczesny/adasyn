import numpy
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# Authors: Cezary Szczesny, Dominik Badora
def test_oversampling_method(X, y, oversampling_method_instance):
    accuracies = []
    precisions = []
    f1_scores = []
    recall_scores = []
    g_means_table = []
    mcc_table = []
    knc = KNeighborsClassifier(n_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
    for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train_resample, y_train_resample = oversampling_method_instance.fit_resample(X_train, y_train)
        knc.fit(X_train_resample, y_train_resample)
        y_predict = knc.predict(X_test)

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        rec_score = recall_score(y_test, y_predict)
        g_mean = geometric_mean_score(y_test, y_predict)
        mcc = matthews_corrcoef(y_test, y_predict)

        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1)
        recall_scores.append(rec_score)
        g_means_table.append(g_mean)
        mcc_table.append(mcc)

        return numpy.average(accuracies), numpy.average(precisions), numpy.average(f1_scores), numpy.average(
            recall_scores), numpy.average(g_means_table), numpy.average(mcc_table)


# Authors: Cezary Szczesny, Dominik Badora
def test_implemented_adasyn(X, y, oversampling_method_instance):
    accuracies = []
    precisions = []
    f1_scores = []
    recall_scores = []
    g_means_table = []
    mcc_table = []
    knc = KNeighborsClassifier(n_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
    for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        X_train_resample, y_train_resample = oversampling_method_instance._fit_resample(X_train, y_train, ratio=0.9)
        knc.fit(X_train_resample, y_train_resample)
        y_predict = knc.predict(X_test)

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        rec_score = recall_score(y_test, y_predict)
        g_mean = geometric_mean_score(y_test, y_predict)
        mcc = matthews_corrcoef(y_test, y_predict)

        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1)
        recall_scores.append(rec_score)
        g_means_table.append(g_mean)
        mcc_table.append(mcc)

        return numpy.average(accuracies), numpy.average(precisions), numpy.average(f1_scores), numpy.average(
            recall_scores), numpy.average(g_means_table), numpy.average(mcc_table)


# Authors: Cezary Szczesny, Dominik Badora
def test_before_oversampling(X, y):
    accuracies = []
    precisions = []
    f1_scores = []
    recall_scores = []
    g_means_table = []
    mcc_table = []
    knc = KNeighborsClassifier(n_neighbors=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)
    for idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        knc.fit(X_train, y_train)
        y_predict = knc.predict(X_test)

        accuracy = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)
        rec_score = recall_score(y_test, y_predict)
        g_mean = geometric_mean_score(y_test, y_predict)
        mcc = matthews_corrcoef(y_test, y_predict)

        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1)
        recall_scores.append(rec_score)
        g_means_table.append(g_mean)
        mcc_table.append(mcc)

        return numpy.average(accuracies), numpy.average(precisions), numpy.average(f1_scores), numpy.average(
            recall_scores), numpy.average(g_means_table), numpy.average(mcc_table)
