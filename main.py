#Importamos los módulos y librerías que vamos a necesitar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import seaborn as sns

from patsy import dmatrices
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Cargamos los datos
dta = sm.datasets.fair.load_pandas().data
dta.head(10)

dta.columns = ['rate_marriage', 'age', 'yrs_married', 'children', 'religious', 'educ', 'occupation', 'occupation_husb','affairs']

#Cargamos los datos
dta = sm.datasets.fair.load_pandas().data
dta.head(10)

#Información sobre el dataset: descripción general, origen, 
#definición de variables,tipo de variables

print(sm.datasets.fair.NOTE)
print(sm.datasets.fair.SOURCE)
print(sm.datasets.fair.DESCRLONG)

dta.info()

#Comprobamos que no falten datos (Resultado booleano: true=falta dato, false=dato)
#También se puede visualizar si faltan datos con los mapas de calor de seaborn.
#En este caso, no hace falta.

dta.isnull().head(10)

# Veamos ahora la matriz de correlación. 
# Deberíamos eliminar las variables altamente correlacionadas >0,90
# Edad, años matrimonio-- lógica
# Correlación positiva--religious/rate marriage,age/yrs_marriage
# Correlación negativa: affairs/children, religious

print(dta.corr())

#Edad, años matrimonio-- lógicamente no son independientes, para eliminarlos habría que hacer:
#dta.drop(['age','yrs_married'],axis=1,inplace=True)
#dta.head()


# histograma sobre influencia del nivel educativo
dta.educ.hist()
plt.title('Influencia del Nivel Educativo')
plt.xlabel('Nivel Académico')
plt.ylabel('Frecuencia infidelidad')

# Creamos una nueva variable binaria "infidelity" para tratarlo
#como un problema de clasificaciÃ³n 0=fiel, 1=infiel
# Mostramos los 10 primeros ... infieles
dta['infidelity'] = (dta.affairs > 0).astype(int)
dta.head(10)

#Exploramos el dataset
print(dta.describe())

#Agrupamos por la variable "infidelity": 0=fiel, 1=infiel
dta.groupby('infidelity').mean()

#Agrupamos por la variable "religious": 1=no religiosa, 4=muy religiosa
dta.groupby('religious').mean()



# histograma sobre influencia del nivel educativo
dta.educ.hist()
plt.title('Influencia del Nivel Educativo')
plt.xlabel('Nivel AcadÃ©mico')
plt.ylabel('Frecuencia infidelidad')


# histograma sobre influencia del sentimiento religioso
dta.religious.hist()
plt.title('Histograma Nivel Sentimiento religioso')
plt.xlabel('Sentimiento religioso')
plt.ylabel('Frecuencia infidelidad')