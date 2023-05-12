from collections import Counter

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from adasyn import Adasyn

# Generowanie próbek
X, y = make_classification(n_samples=200, weights=[0.9])

print("Liczność próbek przed oversamplingiem:", Counter(y))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='prism')
plt.show()

# Testowanie
print("-------------------------------")
print("Klasyfikator: KNeighborsClassifier ")
kncBefore = KNeighborsClassifier(n_neighbors=5)
kncAfter = KNeighborsClassifier(n_neighbors=5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=56)

# Oversampling
adasyn = Adasyn()
X_oversampled, y_oversampled = adasyn._fit_resample(X, y)
print("Liczność próbek po oversamplingu:", Counter(y_oversampled))
plt.scatter(X_oversampled[:, 0], X_oversampled[:, 1], c=y_oversampled, cmap='prism')
plt.show()

# Przed oversamplingiem
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
kncBefore.fit(X_train, y_train)
y_predicted = kncBefore.predict(X_test)
score = accuracy_score(y_true=y_test, y_pred=y_predicted)
print(f"Wynik przed oversamplingiem: " + "{:.3f}".format(score))
# Po oversamplingu
X_train_after, X_test_after, y_train_after, y_test_after = train_test_split(X_oversampled, y_oversampled, test_size=0.2,
                                                                            train_size=0.8)
kncAfter.fit(X_train_after, y_train_after)
y_predicted_after = kncAfter.predict(X_test_after)
score = accuracy_score(y_true=y_test_after, y_pred=y_predicted_after)
print("Wynik po oversamplingu: " + "{:.3f}".format(score))
print("Koniec: KNeighborsClassifier")
print("-------------------------------")
