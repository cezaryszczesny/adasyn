# Authors: Cezary Szczesny, Dominik Badora

Projekt skupia się na implementacji algorytmu Adasyn
oraz porównanie jego skuteczności do innych algorytmów takich jak:

- ADASYN - zaimplementowany w bibliotece imblearn
- SMOTE
- SMOTE-ENN
- BORDERLINE-SMOTE

Plik adasyn.py zawiera implementacje algorytmu Adasyn.
W celu oversamplingu próbek należy zdefiniowac jej instancje
oraz wykonać na niej metode _fit_resample().

Plik main.py tworzy instancje różnych algorytmów oraz odpowiada
za przedstawienie wyników.

Plik testMethods.py zawiera funkcje, których można użyć do
przetestowania konkretnej instancji algorytmu oversamplingu.

Wykorzystane metryki:

- accuracy_score
- precision_score
- f1_score
- recall_score

Metryki pochodza z biblioteki sklearn.metrics.



