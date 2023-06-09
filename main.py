import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from scipy.stats import shapiro
from sklearn.datasets import make_classification

from adasyn import Adasyn
from testMethods import test_oversampling_method, test_before_oversampling


# Authors: Cezary Szczesny, Dominik Badora

def print_results(method_name, accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table):
    print("----------------------")
    print("Test method: " + method_name)
    print("Accuracy: " + "{:.3f}".format(np.mean(accuracies)) + " " + "{:.3f}".format(np.std(accuracies)))
    print("Precision: " + "{:.3f}".format(np.mean(precisions)) + " " + "{:.3f}".format(np.std(precisions)))
    print("F1: " + "{:.3f}".format(np.mean(f1_scores)) + " " + "{:.3f}".format(np.std(f1_scores)))
    print("Recall: " + "{:.3f}".format(np.mean(recall_scores)) + " " + "{:.3f}".format(np.std(recall_scores)))
    print("G_mean: " + "{:.3f}".format(np.mean(g_means_table)) + " " + "{:.3f}".format(np.std(g_means_table)))
    print("MCC: " + "{:.3f}".format(np.mean(mcc_table)) + " " + "{:.3f}".format(np.std(mcc_table)))


def make_statistics(accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table):
    n_classifiers = 6
    statistics_matters_matrix = np.zeros((n_classifiers, n_classifiers), dtype=bool)

    metrics = [accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table]
    p_value = [0.05]
    # TODO: do poprawy na wektor zamiast macierz
    for i in range(n_classifiers):
        for j in range(n_classifiers):
            if i != j:
                metric = metrics[i]
                stat, p_val = shapiro(metric)
                if p_val < p_value:
                    statistics_matters_matrix[i, j] = True
    print(statistics_matters_matrix)

    return statistics_matters_matrix


def save_results(method_name, accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table):
    np.save('accuracy_' + method_name + '.npy', np.mean(accuracies))
    np.save('precision_' + method_name + '.npy', np.mean(precisions))
    np.save('f1_' + method_name + 'npy', np.mean(f1_scores))
    np.save('recall_' + method_name + '.npy', np.mean(recall_scores))
    np.save('g_mean_' + method_name + '.npy', np.mean(g_means_table))
    np.save('mcc_' + method_name + '.npy', np.mean(mcc_table))


# Generate samples
X, y = make_classification(n_samples=500, weights=[0.9])

# Before oversampling
scores = test_before_oversampling(X, y)
# save_results("Before oversampling", *scores)
print_results("Before oversampling", *scores)
make_statistics(*scores)

# Oversampling with implemented Adasyn
adasyn = Adasyn()
scores = test_oversampling_method(X, y, adasyn)
# save_results("Implemented Adasyn", *scores)
print_results("Implemented Adasyn", *scores)
make_statistics(*scores)

# Oversampling with imported Adasyn
ADASYN = ADASYN()
scores = test_oversampling_method(X, y, ADASYN)
# save_results("Imported Adasyn", *scores)
print_results("Imported Adasyn", *scores)
make_statistics(*scores)

# Oversampling with SMOTE
SMOTE = SMOTE()
scores = test_oversampling_method(X, y, SMOTE)
# save_results("SMOTE", *scores)
print_results("SMOTE", *scores)
make_statistics(*scores)

# Oversampling with SMOTE-ENN
SMOTE_ENN = SMOTEENN()
scores = test_oversampling_method(X, y, SMOTE_ENN)
# save_results("SMOTE-ENN", *scores)
print_results("SMOTE-ENN", *scores)
make_statistics(*scores)

# Oversampling with SMOTE-BORDERLINE
BORDERLINE_SMOTE = BorderlineSMOTE()
scores = test_oversampling_method(X, y, BORDERLINE_SMOTE)
# save_results("SMOTE-BORDERLINE", *scores)
print_results("SMOTE-BORDERLINE", *scores)
make_statistics(*scores)

print("----------------------")
print("THE END")
