#Importação da biblioteca Pandas
import pandas as pd

#Definição das colunas e da base de dados
nome_colunas=['sepal length', 'sepal width', 'petal length', 'petal width', 'Tipo'] 
banco_de_dados = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names=nome_colunas)


#Definindo a saída como a coluna Tipo
saida = banco_de_dados[['Tipo']] 

#Separando a saída em três colunas a serem preenchidas de forma booleana
saida = pd.get_dummies(saida, columns=['Tipo']) 
values = list(saida.columns.values)

saida.head()



#Retirada da coluna de saída do grupo de dados da entrada
rede = banco_de_dados.drop(columns=['Tipo'])



#Importação de função e divisão entre treino e teste
from sklearn.model_selection import train_test_split

rede_treino, rede_teste, saida_treino, saida_teste = train_test_split(rede, saida,test_size=0.2)

print("\nRede de Treino (sem a saída):\n")
print(rede_treino.head())
print(rede_treino.shape)

print("\nRede de Teste (sem a saída):\n")
print(rede_teste.head())
print(rede_teste.shape)



#Importação do modelo Sequencial e da forma densa para camadas
from keras.models import Sequential
from keras.layers import Dense

#Criação do modelo da rede
model = Sequential()

#Definição de variável armazenando o número de colunas
numero_colunas = rede.shape[1]


#Adicionando as camadas
model.add(Dense(10, activation='relu', input_shape=(numero_colunas,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(3))


#Compilação do modelo e definição da optimização e função de perda
model.compile(optimizer='adam', loss='mean_squared_error')


#Interrompe as interações quando a rede deixa de se aprimorar
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=3)

#Treino do modelo
teste = model.fit(rede_treino, saida_treino, validation_data=(rede_teste, saida_teste),  epochs=150, callbacks=[early_stopping_monitor])


import matplotlib.pyplot as plt
fig, graf_perda = plt.subplots()

graf_perda.plot(teste.history['loss'], 'r', marker ='.', label='Perda no Treino')
graf_perda.plot(teste.history['val_loss'], 'b', marker = '.', label='Perda na Validação') 
graf_perda.legend()

model.predict(rede_treino.iloc[0:5])


#Importação da matriz de confusão
import numpy as np
from sklearn.metrics import confusion_matrix

valor_pred = np.zeros(rede_treino.shape[0])

#Fazendo a predição da rede
predicao = model.predict(rede_treino)
valor_real = np.array(saida_treino).argmax(axis=1)

for i in range(rede_treino.shape[0]):
  valor_pred[i] = predicao[i].argmax()

#Imprimindo a matroz
confusion_matrix(valor_real, valor_pred)



#Calculando a acurária do algoritmo
from sklearn.metrics import accuracy_score
acuracia = (accuracy_score(valor_real, valor_pred) * 100)

print ("A acurária dos casos de treino foi de %d" % (acuracia),"%.")


#Importação da matriz de confusão
import numpy as np
from sklearn.metrics import confusion_matrix

valor_pred = np.zeros(rede_teste.shape[0])

#Fazendo a predição da rede
predicao = model.predict(rede_teste)
valor_real = np.array(saida_teste).argmax(axis=1)

for i in range(rede_teste.shape[0]):
  valor_pred[i] = predicao[i].argmax()

#Imprimindo a matriz
confusion_matrix(valor_real, valor_pred)


#Calculando a acurária do algoritmo
from sklearn.metrics import accuracy_score
acuracia = (accuracy_score(valor_real, valor_pred) * 100)

print ("A acurária dos casos de teste foi de %d" % (acuracia),"%.")


