import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

classe_real = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
classe_prevista = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]

df = pd.DataFrame({'Classe Real': classe_real, 'Classe Prevista': classe_prevista})

report = classification_report(df['Classe Real'], df['Classe Prevista'])

print("\nQuestao 1:")
print("Relatorio de Classificacao:")
print(report)

matriz_confusao = confusion_matrix(df['Classe Real'], df['Classe Prevista'])

print("\nMatriz de Confusao:")
print(matriz_confusao)

print("\nQuestao 2:")

print("\na)")
classe_real  = [1, 2, 0, 1, 1, 0, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 2, 2, 1, 2, 0, 0, 2, 2, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 2, 0]
classe_prevista = [2, 0, 2, 1, 1, 2, 2, 1, 0, 0, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 2, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 2, 2, 1, 2, 1]

df = pd.DataFrame({'Classe Real': classe_real, 'Classe Prevista': classe_prevista})
report = classification_report(df['Classe Real'], df['Classe Prevista'])

print("Relatorio de Classificacao:")
print(report)

matriz_confusao = confusion_matrix(df['Classe Real'], df['Classe Prevista'])

print("\nMatriz de Confusao:")
print(matriz_confusao)

print("\nb)")
classe_real = ["cat", "ant", "cat", "cat", "ant", "bird", "cat", "ant", "cat", "cat"]
classe_prevista = ["ant", "ant", "cat", "cat", "ant", "bird", "cat", "ant", "ant", "bird"]

df = pd.DataFrame({'Classe Real': classe_real, 'Classe Prevista': classe_prevista})
report = classification_report(df['Classe Real'], df['Classe Prevista'])

print("Relatorio de Classificacao:")
print(report)

matriz_confusao = confusion_matrix(df['Classe Real'], df['Classe Prevista'])

print("\nMatriz de Confusao:")
print(matriz_confusao)
