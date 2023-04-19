import pandas as pd
import numpy as np

# Importando a base de dados
base_parkinson = pd.read_csv('bases/parkinsons_updrs.csv')
print('Base de dados: ')
print(base_parkinson)


# por curiosidade, para ver todos os atributos da base (que é o cabeçalho do arquivo csv)
print(base_parkinson.columns)

# definição dos atributos dependentes em 2 
y_motor = base_parkinson.iloc[:, 4].values
print(y_motor)
y_total = base_parkinson.iloc[:, 5].values
print(y_total)

# definindo a variável que contém os atributos previsores (independentes)
base = base_parkinson.drop(['motor_UPDRS', 'total_UPDRS'], axis=1)
X_parkinson = base.iloc[:, 0:].values
print(X_parkinson)
print(X_parkinson.shape)

y_parkinson = base_parkinson.iloc[:, 4:6].values
print(y_parkinson)
print(y_parkinson.shape)