# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 00:06:35 2019

@author: carloslme
"""

# Regresion Linear Simple
 
# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Toma todas las columnas menos la ultima
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Escalado de variables
# EN ESTE ALGORITMO NO HACE FALTA ESCALAR LAS VARIABLES, LA LIBRERIA LO HACE AUTOMATICAMENTE
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Crear modelo de Regresion Lineal Simple con el conjunto de entrenamiento
# Importar la libreria
from sklearn.linear_model import LinearRegression
# Crear el objeto para hacer la regresion
regression = LinearRegression() # No hace falta ingresar argumentos
# Ajustar el modelo linear
regression.fit(X_train, y_train) # Tanto como X_train y y_train deben el mismo tamaño
"""
Out[]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
fit_intercept=True => Calculara la ordenada en el origen 
n_jobs=None => Define cuantos procesadores realizan la tarea
normalize=False => Permite normalizar los datos antes de hacer el modelo 
---
Ahora la "maquina" ha aprendido el conjunto de relaciones entre las variables y deducido
cual es la mejor ordenada en el origen y cual es la mejor pendiente basado en la tecnica
de los minimos cuadrados, en resumen, ha creado el modelo.
"""

# Predecir el conjunto de entrenamiento
y_pred = regression.predict(X_test) # Vector que tiene los sueldos pronosticados por parte del modelo de regresion lineal

# Visualizar los datos de entrenamiento
plt.scatter(X_train, y_train, color="red") # Pintar una nube de puntos
plt.plot(X_train, regression.predict(X_train), color="blue") # Coordenadas X y Y
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualizar los datos de test
plt.scatter(X_test, y_test, color="red") # Pintar una nube de puntos
plt.plot(X_train, regression.predict(X_train), color="blue") # Se usa X_train ya que representa la recta de regresion lineal previamente entrenada
plt.title("Sueldo vs Años de Experiencia (Conjunto de Test)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

