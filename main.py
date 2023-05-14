from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.datasets import make_classification

from adasyn import Adasyn
from testMethods import test_oversampling_method, test_implemented_adasyn, test_before_oversampling


# Authors: Cezary Szczesny

def print_results(method_name, accuracy_score, precision_score, f1_score, recall_score):
    print("----------------------")
    print("Test method: " + method_name)
    print("Accuracy: " + "{:.3f}".format(accuracy_score))
    print("Precision:" + "{:.3f}".format(precision_score))
    print("F1: " + "{:.3f}".format(f1_score))
    print("Recall: " + "{:.3f}".format(recall_score))


# Generate samples
X, y = make_classification(n_samples=500, weights=[0.9])

# Before oversampling
accuracy_score, precision_score, f1_score, recall_score = test_before_oversampling(X, y)
print_results("Before oversampling", accuracy_score, precision_score, f1_score, recall_score)

# Oversampling with implemented Adasyn
adasyn = Adasyn()
accuracy_score, precision_score, f1_score, recall_score = test_implemented_adasyn(X, y, adasyn)
print_results("Implemented Adasyn", accuracy_score, precision_score, f1_score, recall_score)

# Oversampling with imported Adasyn
ADASYN = ADASYN()
accuracy_score, precision_score, f1_score, recall_score = test_oversampling_method(X, y, ADASYN)
print_results("Imported Adasyn", accuracy_score, precision_score, f1_score, recall_score)

# Oversampling with SMOTE
SMOTE = SMOTE()
accuracy_score, precision_score, f1_score, recall_score = test_oversampling_method(X, y, SMOTE)
print_results("SMOTE", accuracy_score, precision_score, f1_score, recall_score)

# Oversampling with SMOTE-ENN
SMOTE_ENN = SMOTEENN()
accuracy_score, precision_score, f1_score, recall_score = test_oversampling_method(X, y, SMOTE_ENN)
print_results("SMOTE-ENN", accuracy_score, precision_score, f1_score, recall_score)

# Oversampling with SMOTE-BORDERLINE
BORDERLINE_SMOTE = BorderlineSMOTE()
accuracy_score, precision_score, f1_score, recall_score = test_oversampling_method(X, y, BORDERLINE_SMOTE)
print_results("SMOTE-BORDERLINE", accuracy_score, precision_score, f1_score, recall_score)

print("----------------------")
print("THE END")
