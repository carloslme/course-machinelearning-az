# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:51:00 2020

@author: carloslme
"""

# =============================================================================
# SVR
# =============================================================================

# =============================================================================
# # Cómo importar las librerías
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# # Importar el data set
# =============================================================================
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# =============================================================================
# # Dividir el data set en conjunto de entrenamiento y conjunto de testing
# =============================================================================
# No se divide ya que el data set es muy pequeño
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)  # fit_transform crea el escalado y lo aplica a X
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X, y)


# Predicción de nuestros modelos
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform([[6.5]]))) # transform solo aplica el escalado a X

# Visualización de los resultados de SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


