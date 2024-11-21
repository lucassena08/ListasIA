import numpy as np
from sklearn.model_selection import train_test_split

X = np.random.rand(100, 5) # 100 amostras e 5 caracteristicas (array bidimensional)
y = np.random.randint(2, size=100) # classes 0 e 1 para 100 amostras

print("Questao 1\n")
print("a)\n")	
print(X)
print("\n")
print(y)

print("\nb)\n")

print("i)\n")

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.33
)

print(X_train, "\n")
print(y_train, "\n")
print(X_test, "\n")
print(y_test, "\n")

print("ii)\n")

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.20
)

print(X_train, "\n")
print(y_train, "\n")
print(X_test, "\n")
print(y_test, "\n")

print("iii)\n")

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2
)

print(X_train, "\n")
print(y_train, "\n")
print(X_test, "\n")
print(y_test, "\n")