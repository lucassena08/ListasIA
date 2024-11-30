# -*- coding: utf-8 -*-
"""Aula - AGs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JZJzvK0zg1FxZuAjy9Pc9oaQtL7mwwVQ

Genetic Algorithm Maximization of Non-Linear Function
Ankur Chattopadhyay
https://gist.github.com/chttrjeankr/bbc399f6f2653da2993ff9fca3633212
"""

# Algoritmo Genetico para função de otimização
from numpy.random import randint
from numpy.random import rand


# função de fitness / função objetivo /  maximizar o objetivo
def funcao_de_fitness(x):

    # maximizar o função objetivo
    # x**2 + 3*y  + 4

    # considerando x[0] == x e x[1] == y
    return (x[0] ** 3.0) + (2 * x[1] ) + 3


# converte bitstring para numero real
def bit2real(limites, n_bits, bitstring):

    populacao_real = list()
    maximo = 2 ** n_bits

    for i in range(len(limites)):

        inicio, fim = i * n_bits, (i * n_bits) + n_bits

        substring = bitstring[inicio:fim]
        # convert bitstring to a string of chars
        chars = "".join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)

        value = limites[i][0] + (integer / maximo) * (limites[i][1] - limites[i][0])

        populacao_real.append(value)

    return populacao_real


# seleção de k torneio 
def selection(populacao, lista_avaliacao, k=3):

    # selecao aleatoria do primeiro individuo 
    selecao_ix = randint(len(populacao))

    for ix in randint(0, len(populacao), k - 1):

        # seleciona o individuo com melhor Função de fitness
        if lista_avaliacao[ix] > lista_avaliacao[selecao_ix]:
            selecao_ix = ix
    return populacao[selecao_ix]



# Operação de crossover de 2 pais para gerar 2 filhos
def crossover(pai1, pai2, r_cross):
    """
    r_cross : taxa que determina se o crossover será realizado ou não,
    caso contrario, os pais são copiados para a próxima geração
    
    r_cross possui valor próximo a 1.0
    """

    # filhos are copies of parents by default
    filho1, filho2 = pai1.copy(), pai2.copy()

    # sorteio dos pais que vão fazer crossover
    if rand() < r_cross:

        # um ponto de corte
        pt = randint(1, len(pai1) - 2)

        filho1 = pai1[:pt] + pai2[pt:]
        filho2 = pai2[:pt] + pai1[pt:]

    return [filho1, filho2]


# Operação de mutação
def mutacao(bitstring, r_mut):
    """
    mutate bitstring itself, NOT the copy
    """
    for i in range(len(bitstring)):
        # sorteiro do gen para sofre a mutacao
        if rand() < r_mut:
            # invertendo o valor do bit
            """
                1 - (1) => 0
                1 - (0) => 1
            """
            bitstring[i] = 1 - bitstring[i]


# algortimo genetico
def algoritmo_genetico(funcao_de_fitness, limites, n_bits, n_iter, n_populacao, r_cross, r_mut):

    # iniciando a populacao aleatoriamente com string de bits
    populacao = [randint(0, 2, n_bits * len(limites)).tolist() for _ in range(n_populacao)]

    #for i, p in enumerate (populacao):
     # print('individuo: ',i , p , bit2real(limites, n_bits,p) )
   
    """ Exemplo
    populacao =

    [ [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]  
      [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0] 
      [0, 1, 1, 0, 0, 0, 0, 0] 
      ...
      [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]
      [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0] 
    ]
    """

    # Melhor individuo por geracao
    melhor_individuo, melhor_avaliacao = 0, funcao_de_fitness(bit2real(limites, n_bits, populacao[0]))

    # geraçoes da populacao
    for geracao in range(n_iter):

        # populacao com valores reais
        populacao_real = [bit2real(limites, n_bits, p) for p in populacao]


        # avaliacao dos individuos da populacao
        lista_avaliacao = [funcao_de_fitness(d) for d in populacao_real]


        # Identifica o melhor individuo da geração 
        for i in range(n_populacao):
            if lista_avaliacao[i] > melhor_avaliacao:
                melhor_individuo, melhor_avaliacao = populacao[i], lista_avaliacao[i]
                print("Geração>%d, melhor funcao de fitness f(%s) = %f" % (geracao, populacao_real[i], lista_avaliacao[i]))


        # Selecao dos pais
        individuos_selecionados = [selection(populacao, lista_avaliacao) for _ in range(n_populacao)]

        # Criando a proxima geração
        filhos = list()
        for i in range(0, n_populacao, 2):

            # selecionado pais em pares
            pai1, pai2 = individuos_selecionados[i], individuos_selecionados[i + 1]

            # Operacao de crossover e mutacao
            for c in crossover(pai1, pai2, r_cross):
              
                # mutacao
                mutacao(c, r_mut)

                # filhos adicionados para a nova geracao
                filhos.append(c)

        # replace population
        populacao = filhos

    return [melhor_individuo, melhor_avaliacao]



# limite de cada atributo
limites = [[0.0, 51], [10.0, 51]]

# define the total iterations
n_iter = 500

# quantidade de bits por atributo
n_bits = 8 

# tamanho da população
n_populacao = 100

# taxa de crossover rate: alta probabilidade
r_cross = 0.9

# taxa de mutação: baixa probabilidade
r_mut = 1.0 / (float(n_bits) * len(limites))


# resultado do algoritmo geracaoetico
melhor_individuo, avaliacao = algoritmo_genetico (
    funcao_de_fitness, limites, n_bits, n_iter, n_populacao, r_cross, r_mut
)

# print("Melhor individuo: ",melhor_individuo,  decode(limites, n_bits,best), "=", avaliacao)

print()


print("1.2)", "\n")

individuo_real = bit2real(limites, n_bits, melhor_individuo)

print("f( %s ) = %f" % (individuo_real, avaliacao), "\n")

print("1.3)", "\n")

n_populacao = 50
n_bit = 6
limites = [[1.5, 20], [10.0, 30]]

individuo_real = bit2real(limites, n_bits, melhor_individuo)

print("f( %s ) = %f" % (individuo_real, avaliacao), "\n")
