# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 03:53:16 2021

@author: LauraSandinoPerdomo
"""

# CASO DE CONSULTORIA
"""
Una de las operaciones de cobranzas de la compañía quiere generar estrategias diferenciadas para el proceso de gestión de recuperación de cartera de clientes de acuerdo con el riesgo de no pago de la primera factura.

La estrategia se divide en 3 grupos de intervención:

1.	Alto riesgo: Llamarlos al 5 día de mora.
2.	Medio riesgo: Enviar mensaje de texto al 5 día de mora.
3.	Bajo riesgo: Enviar mensaje de texto al día 15 de mora.
Los costos por cada tipo de contacto son los siguientes:

•	Llamada asesor de cobranza 1700 pesos
•	Mensaje de texto 40 pesos
Instrucciones

1.	Muestre un análisis descriptivo y/o diagnóstico inicial de la información insumo para el modelo.
2.	Construya un modelo estadístico que calcule la probabilidad de que un cliente no pague la primera factura. Explique por qué escogió las variables con las que va a trabajar y si debió hacer modificaciones de estas.
3.	Defina los puntos de corte que determinen a que grupo de estrategia pertenece cada cliente.
4.	Describa el perfil de los clientes con un alto riesgo de no pago.
5.	¿Qué sugerencias haría usted al equipo de cobranzas de acuerdo con el análisis de la información del modelo?
6.	Explique el modelo y sustente su validez estadística, así como los puntos de corte, la cantidad de clientes que pertenecen a cada estrategia, los perfiles de riesgo y sus sugerencias y conclusiones.
7.	Adjunte la base de datos con la probabilidad de riesgo de cada cliente.
"""


#LIBRERÍAS
# Datos
# ==============================================================================
import pandas as pd
import numpy as np
import pandas_profiling


# Preprocesado y modelado
# ==============================================================================
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

#Lectura BD
df = pd.read_excel('D:\Laura\PRUEBA TECNICA\Base de datos prueba tecnica.xlsx', sheet_name='DB')
df = df.set_index('cliente_id')

#Revisión de los tipos de datos en la base
print("Tipos de datos en BD")
df.dtypes

#Cambio de tipos de datos a las variables Estrato y cliente_id
df[['estrato']] = df[['estrato']].astype(str)
print("Tipos de datos en BD")
df.dtypes

#Rta 1/ Muestre un análisis descriptivo y/o diagnóstico inicial de la información insumo para el modelo.
profile = pandas_profiling.ProfileReport(df,title="Reporte Servicio", explorative=True)
profile.to_file("Reporte.html")

## Descriptivo de los datos recibidos

profile.to_widgets()

## Conclusiones de los descriptivos anteriores
"""
- El conjunto de datos tiene 7393 (37.1%) registros duplicados
- DEPARTAMENTO y REGIONAL están altamente correlacionadas -> (Probablemente sea mejor modelar con alguna, no con ambas) 
- CANAL_HOMOLOGADO_MILLICON y GERENCIA están altamente correlacionadas -> (Probablemente sea mejor modelar con alguna, no con ambas) 
- tipo_fuerza_venta y CANAL_HOMOLOGADO_MILLICON están altamente correlacionadas -> (Probablemente sea mejor modelar con alguna, no con ambas) 
- portafolio y productos están altamente correlacionadas -> (Probablemente sea mejor modelar con alguna, no con ambas) 
- antiguedad_meses tiene 1678 (8.4%) valores faltantes -> Como es numérica tal vez valga la pena categorizarla y dejar los faltantes con ND
- no_serv_tecnicos tiene 13033 (65.4%) valores faltantes -> Deben considerarse como Ceros
- fallo tiene 13033 (65.4%) valores faltantes -> Deberían considerarse como NA porque no tuvieron solicitud de servicios técnicos
"""

## Teniendo en cuenta los descriptivos se evidencian registros duplicados que serán retirados para posteriores análisis

#Retira registros duplicados
df = df.drop_duplicates()

print("Verifica la cantidad de registros duplicados (True=0)")
print(df.duplicated().value_counts())

print("Distribución de la variable antiguedad_meses, para discretizar en cuartiles")
df['antiguedad_meses'].describe()

df['antiguedad_meses_binned'] = pd.qcut(df['antiguedad_meses'], q=4, precision=0)
df.head()

df_modelo = df
df_modelo = df_modelo.drop(columns = ['antiguedad_meses'])
df_modelo = pd.get_dummies(df_modelo, columns=['REGIONAL','TECNOL','tipo_fuerza_venta','portafolio','antiguedad_meses_binned','estrato'])                        
df_modelo = df_modelo.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64","uint8"])
df_modelo['no_serv_tecnicos'] = df_modelo['no_serv_tecnicos'].replace(np.nan,0.0)

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
modelo = sm.Logit(endog=y_train, exog=X_train)
modelo = modelo.fit()
print(modelo.summary())

# Predicciones con intervalo de confianza 
# ==============================================================================
predicciones = modelo.predict(exog = X_train)
X_train ['pedicciones'] = predicciones

# Clasificación predicha
# ==============================================================================
clasificacion = np.where(predicciones<0.5, 0, 1)
clasificacion

# Accuracy de test del modelo 
# ==============================================================================
X_test = sm.add_constant(X_test, prepend=True)
predicciones = modelo.predict(exog = X_test)
X_test ['pedicciones'] = predicciones
with pd.ExcelWriter('D:\Laura\PRUEBA TECNICA\Predicciones.xlsx') as writer:  
    X_train.to_excel(writer, sheet_name='DatosEntrenamiento')
    X_test.to_excel(writer, sheet_name='DatosPrueba')
    


clasificacion = np.where(predicciones<0.5, 0, 1)
accuracy = accuracy_score(
            y_true    = y_test,
            y_pred    = clasificacion,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

# Matriz de confusión de las predicciones de test
# ==============================================================================
confusion_matrix = pd.crosstab(
    y_test.ravel(),
    clasificacion,
    rownames=['Real'],
    colnames=['Predicción']
)
confusion_matrix

### 2.Construya un modelo estadístico que calcule la probabilidad de que un cliente no pague la primera factura. Explique por qué escogió las variables con las que va a trabajar y si debió hacer modificaciones de estas.
"""
Utilicé una regresión logística porque me permite revisar la influencia de las variables sobre la variable dependiente. Dentro de las transformaciones que tuve que hacer estuvieron:
- Dummies para las variables categóricas
- Llevar la antiguedad a una varable categórica para poder utilizarla en el modelo

Dentro del análisis descriptivo preliminar se pudo evidenciar la correlación de algunas variables (como se mencionó anteriormente después del descriptivo), y no incluir variables correlacionadas.

A medida que corría el modelo podía ver a través del p-value si las variables eran estadísticamente relevantes para el modelo o no. A pesar de que algunas de ellas no lo son, llevé el modelo hasta el punto donde obtuve el máximo accuracy.

Además, a través de la herramienta KNIME AP pude hacer una exploración de las variables para revisar cuáles eran más relevantes a partir de un forward/backward feature selection que me permitió ver que variables como no_servicios y quejas eran relevantes.
"""