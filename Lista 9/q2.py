import numpy as np
from sklearn.datasets import make_classification

X = np.random.rand(1000, 3) # 1000 amostras e 5 caracteristicas (array bidimensional)
y = np.random.randint(4, size=1000) # classes (0, 1, 2, 3) para 100 amostras

print("Questao 2\n")

print(X)
print("\n")
print(y)

X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, n_clusters_per_class=1)

print(X, "\n")
print(y, "\n")