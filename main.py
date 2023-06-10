import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from implemented_adasyn import Adasyn
from scipy.stats import ttest_rel
from sklearn.datasets import make_classification

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


def print_single_result(metric_name, method_names, method_metrics_scores_table):
    print("----------------------")
    print(metric_name)
    print("\n")
    methods_amount = len(method_names)

    for i in range(methods_amount):
        print(method_names[i] + " " + "{:.3f}".format(np.mean(method_metrics_scores_table[i, :])) + " ± " +
              "{:.3f}".format(np.std(method_metrics_scores_table[i, :])))


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
before_over_scores = np.array(test_before_oversampling(X, y))
# save_results("Before oversampling", *before_over_scores)
# print_results(*before_over_scores)

# Oversampling with implemented Adasyn
implemented_adasyn = Adasyn()
impl_adasyn_scores = np.array(test_oversampling_method(X, y, implemented_adasyn))
# save_results("Implemented Adasyn", *impl_adasyn_scores)
# print_results(*impl_adasyn_scores)

# Oversampling with imported Adasyn
adasyn = ADASYN()
imported_adasyn_scores = np.array(test_oversampling_method(X, y, adasyn))
# save_results("Imported Adasyn", *imported_adasyn_scores)
# print_results(*imported_adasyn_scores)

# Oversampling with SMOTE
smote = SMOTE()
smote_scores = np.array(test_oversampling_method(X, y, smote))
# save_results("SMOTE", *smote_scores)
# print_results(*smote_scores)

# Oversampling with SMOTE-ENN
smote_enn = SMOTEENN()
smote_enn_scores = np.array(test_oversampling_method(X, y, smote_enn))
# save_results("SMOTE-ENN", *smote_enn_scores)
# print_results(*smote_enn_scores)

# Oversampling with SMOTE-BORDERLINE
borderline_smote = BorderlineSMOTE()
borderline_smote_scores = np.array(test_oversampling_method(X, y, borderline_smote))
# save_results("SMOTE-BORDERLINE", *borderline_smote_scores)
# print_results(*borderline_smote_scores)

all_scores = np.array([before_over_scores, impl_adasyn_scores, imported_adasyn_scores, smote_scores, smote_enn_scores,
                       borderline_smote_scores])
pair_test(all_scores)

print("----------------------")
print("THE END")
