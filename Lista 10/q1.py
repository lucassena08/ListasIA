import pandas as pd
import numpy as np
# d) Importe o método Perceptron de sklearn.linear_mode
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

# a) fazer leitura de dados

df = pd.read_csv("pesquisa_satisfacao.csv")

# b) converter SAT para 1 e INS para 0
df.loc[:, 'CLASSE'] = df.loc[:, 'CLASSE'].map({'SAT': 1, 'INS': 0})

#print(df, "\n")

# c) remova a coluna do atributo RID

df = df.drop(columns=['RID'], axis='columns')

print(df, "\n")

# e) Treinar a sua rede Perceptron com o método fit
X = df[['TEMPO', 'PRECO']].to_numpy()
print(X, "\n")
y = df['CLASSE']
y = pd.to_numeric(y, errors='coerce')
print(y, "\n")

model = Perceptron()
model.fit(X, y)

# f) Classificar todos os dados de entrada no classificador Perceptron

y_pred = model.predict(X)

print(y_pred, "\n")

# g) Mostrar o resultado da métrica Acurácia (Score)

accuracy = accuracy_score(y, y_pred)
print(accuracy, "\n")

# h) Mostre a matriz de confusão

conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix, "\n")