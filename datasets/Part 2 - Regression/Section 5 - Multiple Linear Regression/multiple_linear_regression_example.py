# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 23:38:46 2020

@author: carloslme
"""

# Regresion Lineal Multiple

# =============================================================================
# Cómo importar las librerías
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Importar el dataset
# =============================================================================
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# =============================================================================
# Codificar datos categóricos, pasar a datos dummy la columna State
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # El numero es la columna a convertir de strings a datos numericos

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
### Evitando la trampa de las variables dummy, es decir, borrar 1, es decir, n-1
X = X[:, 1:] # Tomamos todas las columnas a partir de la 1

# =============================================================================
# Dividir el data set en conjunto de entrenamiento y conjunto de testing
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# Ajustar el modelo de regresion linear multiple con el conjunto de entrenamiento
# =============================================================================
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
#[Out]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

# =============================================================================
# Prediccion de los resultados en el conjuntos de testing
# =============================================================================
y_pred = regression.predict(X_test)

## Construir el modelo optimo de RLM utilizando la Eliminacion hacia atras
import statsmodels.regression.linear_model as sm 
""" Nos permite agregar una columna de 1's que corresponde al coeficiente del termino independiente,
    arr = X -> Indica que el arreglo es X
    values = np.ones((50,1)).astype(int) -> Ingresar 50 filas con 1's de tipo int
    axis = 1 -> Indica que añade en columnas, si se quisiera en filas sería '0'
"""
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 


## Paso 1: Seleccionar nivel de significacion para permanecer en el modelo
SL = 0.05

## Paso 2: Al usar statsmodel se necesita crear un nuevo regresor, el de LinearRegression no funciona aqui
### Crear nueva matriz de variables independientes optimas y estadisticamente significativas
X_opt = X[:, [0,1,2,3,4,5]]
""" sm.OLS(endog = __, exog = __)
    endog -> La variable que se desea predecir, es un vector unidimensional
    exog -> La variable externa que representa la matriz de caracteristicas
"""
#regression_OLS = sm.ols(endog = y, exog = X_opt.tolist()).fit() # OLS = Ordinary List Square
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
## Paso 3: Considera la variable predictora con el p-valor más grande con base en el summary
## Paso 4: Se eliminan variables predictoras 
## Paso 5: Se ajusta el modelo sin dicha variable
X_opt = X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0,3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

## Se concluye que el modelo esta listo, que el mejor modelo de regresion lineal 
## multiple es lineal simple y como unica variable predictora el gasto en I+D