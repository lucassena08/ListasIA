# https://docs.google.com/document/d/1aMPtvI3eIdmpVX1uAQG3fKHW4BgXGw_keMU8AKKKZuo/edit?pli=1&tab=t.0#heading=h.e46mn672x8i8

import pandas as pd

pai1 = list("10100011")
pai2 = list("00111001")

df = pd.DataFrame({
  "Pai 1": pai1,
  "Pai 2": pai2
})

def crossover_um_ponto(df):
  meio = len(df) // 2
  filho1 = list(df["Pai 1"][:meio]) + list(df["Pai 2"][meio:])
  filho2 = list(df["Pai 2"][:meio]) + list(df["Pai 1"][meio:])
  return "".join(filho1), "".join(filho2)

def crossover_dois_pontos(df, corte1, corte2):
  filho1 = (
    list(df["Pai 1"][:corte1]) +
    list(df["Pai 2"][corte1:corte2]) +
    list(df["Pai 1"][corte2:])
  )
  filho2 = (
    list(df["Pai 2"][:corte1]) +
    list(df["Pai 1"][corte1:corte2]) +
    list(df["Pai 2"][corte2:])
  )
  return "".join(filho1), "".join(filho2)

filho1_1corte, filho2_1corte = crossover_um_ponto(df)
corte1, corte2 = 2, 6
filho1_2cortes, filho2_2cortes = crossover_dois_pontos(df, corte1, corte2)

print("DataFrame dos pais:")
print(df)

print("\nCrossover com 1 corte:")
print(f"Filho 1: {filho1_1corte}")
print(f"Filho 2: {filho2_1corte}")

print("\nCrossover com 2 cortes:")
print(f"Filho 1: {filho1_2cortes}")
print(f"Filho 2: {filho2_2cortes}")
