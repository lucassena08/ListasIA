import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.svm import SVC


# a) Criar um classificar a partir de dados aleatórios, utilize o método make_classification para gerar uma base de dados
X, y = make_classification(n_samples=250, n_features=4, n_classes = 3, n_clusters_per_class=1)

# b) Separar os grupos de treinamento e teste (20% dos dados  para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# c) Utilize o classificador SVM
clf = SVC()

# d) Altere os KERNEL da SVC, exemplo {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
clf = SVC(kernel='linear')

# e) Treine a SVM com o método fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# f) Mostrar o resultado da métrica Acurácia (Score)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy, "\n")
# g) Mostre a matriz de confusão
confusion = confusion_matrix(y_test, y_pred)
print(confusion, "\n")