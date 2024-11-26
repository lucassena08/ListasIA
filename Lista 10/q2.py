import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("PlayTennis.csv")

label_encoder = LabelEncoder()

# a) Converter os dados abaixo para valores numéricos.

df = df.apply(lambda col: label_encoder.fit_transform(col))

# b) Utilize o classificador MLP (MLPClassifier)

X = df.drop('Play Tennis', axis=1) # tudo menos 'Play Tennis'
y = df['Play Tennis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MLPClassifier(alpha=1e-05, tol=1e-4, random_state=0, max_iter=10000)
# c) Altere os parâmetros da MLP, exemplo: max_iter,  alpha, tol)
clf = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)

# d) Treine a sua rede MLP com o método fit
clf.fit(X_train, y_train)

# e) Mostrar o resultado da métrica Acurácia (Score)
y_pred = clf.predict(X_test)
print(y_pred, "\n")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy, "\n")

# f) Mostre a matriz de confusão
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix, "\n")