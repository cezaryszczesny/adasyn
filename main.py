import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn.datasets import make_classification

from adasyn import Adasyn
from testMethods import test_oversampling_method, test_before_oversampling, pair_test

# Authors: Cezary Szczesny, Dominik Badora


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
