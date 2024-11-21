import pandas as pd

pd.options.display.max_rows = 9999

# Carrega o arquivo CSV
df = pd.read_csv('iris_alterado.csv')

# Remove as colunas vazias
df = df.dropna(axis='columns', how='any')

# Salva o resultado em um novo arquivo CSV
df.to_csv('output.csv', index=False)

print(df)