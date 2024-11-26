import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

# a) Criar um classificar a partir de dados aleatórios ,  utilize o método make_classification para gerar uma base de dados
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=250, n_features=4, n_classes=3, n_clusters_per_class=1)

# b) Separar os grupos de treinamento e teste (20% dos dados  para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# c) Utilize o classificador GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred, "\n")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy, "\n")
confusion = confusion_matrix(y_test, y_pred)
print(confusion, "\n")

# d) Altere os parâmetros da MLP, exemplo: max_iter, alpha, tol)
clf = MLPClassifier(max_iter=500, alpha=0.01, tol=1e-4, random_state=42)

# e) Treine a sua rede MLP com o método fit
clf.fit(X_train, y_train)

# f) Mostrar o resultado da métrica Acurácia (Score)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy, "\n")
# g) Mostre a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)
print(confusion, "\n")