import numpy as np
from imblearn.metrics import geometric_mean_score
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from printUtil import print_single_result


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

    return accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table


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
    return accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table


def pair_test(all_scores_table):
    n_experiment_objects = len(all_scores_table)
    n_metrics = 6
    metrics_names = ["Accuracy statistics", "Precision statistics", "F1-score statistics", "Recall statistics",
                     "G-mean statistics", "MCC statistics"]
    method_names = ["Before Oversampling", "Implemented ADASYN", "Imported ADASYN", "SMOTE", "SMOTE-ENN",
                    "BORDERLINE SMOTE"]
    t_student_matrix = np.zeros((n_experiment_objects, n_experiment_objects))
    p_matrix = np.zeros((n_experiment_objects, n_experiment_objects))
    better_metrics_matrix = np.zeros((n_experiment_objects, n_experiment_objects), dtype=bool)
    statistics_matters_matrix = np.zeros((n_experiment_objects, n_experiment_objects), dtype=bool)
    alpha = 0.05

    for z in range(n_metrics):
        print_single_result(metrics_names[z], method_names, all_scores_table[z])
        for i in range(n_experiment_objects):
            for j in range(n_experiment_objects):
                first_scores_table = all_scores_table[z, i, :]
                second_scores_table = all_scores_table[z, j, :]
                stat, p_value = ttest_rel(first_scores_table, second_scores_table)

                t_student_matrix[i, j] = stat
                p_matrix[i, j] = p_value

                better_metrics_matrix[i, j] = np.mean(first_scores_table) > np.mean(second_scores_table)
                better_metrics_matrix[j, i] = np.mean(first_scores_table) <= np.mean(second_scores_table)
                statistics_matters_matrix[i, j] = p_value < alpha

        advantage_matter_stat_matrix = better_metrics_matrix * statistics_matters_matrix
        print("\n Order of methods in columns")
        print(method_names)
        print("\n T-student matrix")
        print(t_student_matrix)
        print("\n P matrix")
        print(p_matrix)
        print("\n Better matrix")
        print(better_metrics_matrix)
        print("\n Stat matter matrix")
        print(statistics_matters_matrix)
        print("\n Adv matter matrix")
        print(advantage_matter_stat_matrix)
        print("\n")
