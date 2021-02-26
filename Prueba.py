# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:29:44 2021

@author: LauraSandinoPerdomo
"""
# Datos
# ==============================================================================
import pandas as pd
import numpy as np
import pandas_profiling


# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler


# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')


# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')




df = pd.read_excel('D:\Laura\PRUEBA TECNICA\Base de datos prueba tecnica.xlsx', sheet_name='DB')

print(df.dtypes)
df[['estrato','cliente_id']] = df[['estrato','cliente_id']].astype(str)


#profile = pandas_profiling.ProfileReport(df,title="Reporte Servicio", explorative=True)
#profile.to_widgets()
#$profile.to_file("D:\Laura\PRUEBA TECNICA\Reporte.html")

df.duplicated().value_counts()
df = df.drop_duplicates()


df['antiguedad_meses_binned'] = pd.qcut(df['antiguedad_meses'], q=4, precision=0)


df= pd.get_dummies(df, columns=['REGIONAL','TECNOL','GERENCIA', 'tipo_fuerza_venta',
                                'portafolio','productos' ,'fallo','antiguedad_meses_binned','estrato'])
df_modelo =df.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64","uint8"])
profile = pandas_profiling.ProfileReport(df_modelo,title="Reporte Servicio", explorative=True)
profile.to_file("D:\Laura\PRUEBA TECNICA\ReporteMod.html")

# División de los datos en train y test
# ==============================================================================
X = df_modelo.drop(columns = ['Incumplimiento_pago'])
y = df_modelo['Incumplimiento_pago']

X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y.values.reshape(-1,1),
                                        train_size   = 0.7,
                                        random_state = 1234,
                                        shuffle      = True)

# Creación del modelo utilizando matrices como en scikitlearn
# ==============================================================================
# A la matriz de predictores se le tiene que añadir una columna de 1s para el intercept del modelo
X_train = sm.add_constant(X_train, prepend=True)
#modelo = sm.OLS(y, X.astype(float)).fit()
modelo = sm.Logit(endog=y_train, exog=X_train)
modelo = modelo.fit()
print(modelo.summary())