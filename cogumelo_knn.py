# Importação das bibliotecas

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import plotly.io as pio

# Exploração dos dados
base_cogumelos = pd.read_csv('bases/secondary_data.csv')
print(base_cogumelos)
print('PRIMEIROS 5 REGITROS')
print(base_cogumelos.head())
print('ÚLTIMOS 5 REGITROS')
print(base_cogumelos.tail())
print('ALGUMAS ESTATÍSTICAS SOBRE A BASE')
print(base_cogumelos.describe())
print('QUANTIDADE DE VALORES FALTANTES')
print(base_cogumelos.isnull().sum())

# Visualização dos dados

# Quantidade de redistros que pertencem a cada classe
print(np.unique(base_cogumelos['class'], return_counts=True))

# Visualização de gráficos

# Gráfico que mostra a contagem de registros que pertencem a uma classe ou a outra
sns.countplot(x = base_cogumelos['class'])

# Gráfico que mostra a relação entre a classificação e a estação do ano
pio.renderers.default = "svg"
grafico = px.parallel_categories(base_cogumelos, dimensions=['class', 'season'])
grafico.show()

# Gráfico que mostra a relação entre a classificação e a cor da tampa
grafico = px.parallel_categories(base_cogumelos, dimensions=['class', 'cap-color'])
grafico.show()

# Divisão entre previsores e classe
print('##### COLUNAS DA BASE DE DADOS #####')
print(base_cogumelos.columns)

# Variável que vai conter os atribtos previsores 
X_cogumelo = base_cogumelos.iloc[:, 1:].values
print(f'Previsores: {X_cogumelo} \n')
# Variável que vai conter o atributo de classe 
y_cogumelo = base_cogumelos.iloc[:, 0].values
print(f'Classe: {y_cogumelo}')


# TRATAMENTO DOS ATRIBUTOS CATEGÓRICOS
 
from sklearn.preprocessing import LabelEncoder

# para aplicar o LabelEncoder, precisa-se instanciar objetos para todos os atributos que necessitam
lbe_cap_shape = LabelEncoder() # cap-shape
lbe_cap_surface = LabelEncoder() # cap-surface
lbe_cap_color = LabelEncoder() # cap-color
lbe_bruise_bleed = LabelEncoder() # does-bruise-bleed
lbe_g_attatchment = LabelEncoder() # gill_attachment
lbe_g_spacing = LabelEncoder() # gill-spacing
lbe_g_color = LabelEncoder() # gill-color
lbe_s_root = LabelEncoder() # stem-root
lbe_s_surface = LabelEncoder() # stem-surface
lbe_s_color = LabelEncoder() # stem-color
lbe_v_type = LabelEncoder() # veil-type
lbe_v_color = LabelEncoder() # veil-color
lbe_has_ring = LabelEncoder() # has-ring
lbe_ring_type = LabelEncoder() # ring-type
lbe_spore_pc = LabelEncoder() # spore-print-color
lbe_habitat = LabelEncoder() # habitat
lbe_season = LabelEncoder() # season

# depois, faz a aplicação para cada atributo
X_cogumelo[:, 1] = lbe_cap_shape.fit_transform(X_cogumelo[:, 1])
X_cogumelo[:, 2] = lbe_cap_surface.fit_transform(X_cogumelo[:, 2])
X_cogumelo[:, 3] = lbe_cap_color.fit_transform(X_cogumelo[:, 3])
X_cogumelo[:, 4] = lbe_bruise_bleed.fit_transform(X_cogumelo[:, 4])
X_cogumelo[:, 5] = lbe_g_attatchment.fit_transform(X_cogumelo[:, 5])
X_cogumelo[:, 6] = lbe_g_spacing.fit_transform(X_cogumelo[:, 6])
X_cogumelo[:, 7] = lbe_g_color.fit_transform(X_cogumelo[:, 7])
X_cogumelo[:, 10] = lbe_s_root.fit_transform(X_cogumelo[:, 10])
X_cogumelo[:, 11] = lbe_s_surface.fit_transform(X_cogumelo[:, 11])
X_cogumelo[:, 12] = lbe_s_color.fit_transform(X_cogumelo[:, 11])
X_cogumelo[:, 13] = lbe_v_type.fit_transform(X_cogumelo[:, 13])
X_cogumelo[:, 14] = lbe_v_color.fit_transform(X_cogumelo[:, 14])
X_cogumelo[:, 15] = lbe_has_ring.fit_transform(X_cogumelo[:, 15])
X_cogumelo[:, 16] = lbe_ring_type.fit_transform(X_cogumelo[:, 16])
X_cogumelo[:, 17] = lbe_spore_pc.fit_transform(X_cogumelo[:, 17])
X_cogumelo[:, 18] = lbe_habitat.fit_transform(X_cogumelo[:, 18])
X_cogumelo[:, 19] = lbe_season.fit_transform(X_cogumelo[:, 19])

# mostrando a primeira linha dos atributos previsores pra ver se deu certo
print(f'LabelEncoder: {X_cogumelo[0]}')

# ESCALONAMENTO/PADRONZAÇÃO DE VALORES
from sklearn.preprocessing import StandardScaler
scaler_cogumelo = StandardScaler() # instanciando um objeto dessa classe
X_cogumelo = scaler_cogumelo.fit_transform(X_cogumelo)
print(f'Padronização: {X_cogumelo[0]}')

# a padronização é importante para manter os valores na mesma escala, para que os números muito distantes
# não causem confusão ao serem colocados para treinamento


# Divisão das bases em treinamento e teste
#   Uma coisa importante a se fazer é dividir as bases em treinamento e teste, para que o algoritmo de 
#   aprendizagem escolhido possa ser avaliado.

from sklearn.model_selection import train_test_split

X_cogumelo_treinamento, X_cogumelo_teste, y_cogumelo_treinamento, y_cogumelo_teste = \
    train_test_split(X_cogumelo, y_cogumelo, test_size=0.80, random_state=0)
    
# normalmente, a maior parte da base vai para o treinamento, mas como eu já havia feito o teste com 80% para o 
# treinamento, e vou ver como fica a acurácia com os 80% para o teste (isso é definido no parâmetro test_size)

print('Treinamento:', X_cogumelo_treinamento.shape, y_cogumelo_treinamento.shape)
print('Teste:', X_cogumelo_teste.shape, y_cogumelo_teste.shape)


# Salvando as variáveis/bases
import pickle
with open('cogumelos.pkl', mode='wb') as f:
    pickle.dump([X_cogumelo_treinamento, y_cogumelo_treinamento, X_cogumelo_teste, y_cogumelo_teste], f)
    
# Aplicação de um algoritmo de aprendizagem
# no meu caso, de acordo com testes anteriores, escolhi o algoritmo knn

# Aplicação do kNN
with open('cogumelos.pkl', 'rb') as f:
    X_cogumelo_treinamento, y_cogumelo_treinamento, X_cogumelo_teste, y_cogumelo_teste = pickle.load(f)

print('Treinamento:', X_cogumelo_treinamento.shape, y_cogumelo_treinamento.shape)
print('Teste:', X_cogumelo_teste.shape, y_cogumelo_teste.shape)

from sklearn.neighbors import KNeighborsClassifier

knn_cogumelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
knn_cogumelo.fit(X_cogumelo_treinamento, y_cogumelo_treinamento)
# a função fit encaixa o algoritmo na base de treinamento, com isso, o algoritmo é treinado
# OBS.: Não existe um treinamento e um modelo para o kNN

# abaixo vamos fazer as previsões
prev = knn_cogumelo.predict(X_cogumelo_teste)
print(prev)

# agora, fazemos a análise da acurácia
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_cogumelo_teste, prev))
# 99,4% - (80 teste, 20 treinamento)
# 99,96% - (20 teste, 80 treinamento)

print(classification_report(y_cogumelo_teste, prev))


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(knn_cogumelo)
cm.fit(X_cogumelo_treinamento, y_cogumelo_treinamento)
cm.score(X_cogumelo_teste, y_cogumelo_teste)








