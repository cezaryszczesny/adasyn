import numpy as np


def save_results(method_name, accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table):
    np.save('accuracy_' + method_name + '.npy', np.mean(accuracies))
    np.save('precision_' + method_name + '.npy', np.mean(precisions))
    np.save('f1_' + method_name + 'npy', np.mean(f1_scores))
    np.save('recall_' + method_name + '.npy', np.mean(recall_scores))
    np.save('g_mean_' + method_name + '.npy', np.mean(g_means_table))
    np.save('mcc_' + method_name + '.npy', np.mean(mcc_table))


def print_single_result(metric_name, method_names, method_metrics_scores_table):
    print("----------------------")
    print(metric_name)
    print("\n")
    methods_amount = len(method_names)

    for i in range(methods_amount):
        print(method_names[i] + " " + "{:.3f}".format(np.mean(method_metrics_scores_table[i, :])) + " ± " +
              "{:.3f}".format(np.std(method_metrics_scores_table[i, :])))


def print_results(method_name, accuracies, precisions, f1_scores, recall_scores, g_means_table, mcc_table):
    print("----------------------")
    print("Test method: " + method_name)
    print("Accuracy: " + "{:.3f}".format(np.mean(accuracies)) + " " + "{:.3f}".format(np.std(accuracies)))
    print("Precision: " + "{:.3f}".format(np.mean(precisions)) + " " + "{:.3f}".format(np.std(precisions)))
    print("F1: " + "{:.3f}".format(np.mean(f1_scores)) + " " + "{:.3f}".format(np.std(f1_scores)))
    print("Recall: " + "{:.3f}".format(np.mean(recall_scores)) + " " + "{:.3f}".format(np.std(recall_scores)))
    print("G_mean: " + "{:.3f}".format(np.mean(g_means_table)) + " " + "{:.3f}".format(np.std(g_means_table)))
    print("MCC: " + "{:.3f}".format(np.mean(mcc_table)) + " " + "{:.3f}".format(np.std(mcc_table)))
