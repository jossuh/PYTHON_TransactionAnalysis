"""Importación de librerías"""


import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

"""Lectura del archivo"""

df = pd.read_csv('/content/DF08 - BaseRetiros - H7-TablaHechos.csv')

"""Análisis del archivo"""

print(df.head(), "\n")
print(df.info(), "\n")
print(df.describe())

"""Filtro del año 2012"""

df_2012 = df[df['AÑO']==2012]

"""Transacciones por mes a lo largo de los años"""

retiro_mes = df[df['TRX'] == 1].groupby('MES')['TRX'].count()
deposito_mes = df[df['TRX'] == 2].groupby('MES')['TRX'].count()

trx_mes = pd.DataFrame({'Retiro': retiro_mes, 'Deposito': deposito_mes})

plt.figure(figsize=(8, 6))
trx_mes.plot(kind='bar', color=['lightblue', 'lightcoral'])
plt.title('Total de Transacciones por Mes')
plt.xlabel('Meses')
plt.ylabel('Transacciones Totales')
plt.xticks(rotation=0)
plt.legend(title='Tipo de Transacción')
plt.show()

"""Transacciones exitosas vs rechazadas en 2012"""

df_2012['RESPUESTA_LABEL'] = df_2012['RESPUESTA'].map({1: 'Aprobado', 0: 'Rechazado'})

transacciones_respuesta = df_2012['RESPUESTA_LABEL'].value_counts()

plt.figure(figsize=(8, 6))
transacciones_respuesta.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Transacciones exitosas vs rechazadas en 2012')
plt.ylabel('')
plt.show()

"""Errores mas comunes en 2012"""

errores_2012 = df_2012[df_2012['RESPUESTA'] == 0]

errores_mensaje = errores_2012['MENSAJE'].value_counts()

plt.figure(figsize=(10, 6))
errores_mensaje.plot(kind='bar', color='lightcoral')
plt.title('Errores más comunes en 2012')
plt.xlabel('Tipo de error')
plt.ylabel('Frecuencia')
plt.xticks(rotation=0)
plt.show()

"""Proyección"""

#filtracion de retiros
retiros = df[df['TRX'] == 1]

#transformacion de fechas
retiros['fecha'] = pd.to_datetime(retiros[['AÑO', 'MES', 'DIA']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')


#creacion del dataframe para prophet
#agrupacion de los retiros por día
retiros_por_dia = retiros.groupby('fecha').size().reset_index(name='y')
retiros_por_dia = retiros_por_dia.rename(columns={'fecha': 'ds'})

#envio de datos
m = Prophet()
m.fit(retiros_por_dia)

#preparacion de dataframe con los resultados
proyeccion = m.make_future_dataframe(periods=12, freq='M', include_history=False)
proyeccion

#ejecucion de la proyeccion
forecast = m.predict(proyeccion)
forecast[['ds','yhat','yhat_lower','yhat_upper']]

"""Graficacion del resultado"""

plt.figure(figsize=(8, 6))
plt.plot(forecast['ds'], forecast['yhat'], marker='o', linestyle='-', color='green', label='Retiros Esperados')

plt.xlabel('Fecha')
plt.ylabel('Cantidad de retiros')
plt.title('Cantidad de retiros esperados para los próximos 12 meses')
plt.legend()
plt.grid(True)

plt.show()

"""CREACION DEL TABLERO FINAL"""

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

plt.subplots_adjust(hspace=0.3, wspace=0.3)

trx_mes.plot(kind='bar', color=['lightblue', 'lightcoral'], ax=axs[0, 0])
axs[0, 0].set_title('Total de Transacciones por Mes')
axs[0, 0].set_xlabel('Meses')
axs[0, 0].set_ylabel('Transacciones Totales')
axs[0, 0].legend(title='Tipo de Transacción')

transacciones_respuesta.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], ax=axs[0, 1])
axs[0, 1].set_title('Transacciones Exitosas vs Rechazadas en 2012')
axs[0, 1].set_ylabel('')

errores_mensaje.plot(kind='bar', color='lightcoral', ax=axs[1, 0])
axs[1, 0].set_title('Errores Más Comunes en 2012')
axs[1, 0].set_xlabel('Tipo de Error')
axs[1, 0].set_ylabel('Frecuencia')
axs[1, 0].set_xticklabels(errores_mensaje.index, rotation=0)

axs[1, 1].plot(forecast['ds'], forecast['yhat'], marker='o', linestyle='-', color='green', label='Retiros Esperados')
axs[1, 1].set_title('Cantidad de Retiros Esperados para los Próximos 12 Meses')
axs[1, 1].set_xlabel('Fecha')
axs[1, 1].set_ylabel('Cantidad de Retiros')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.show()