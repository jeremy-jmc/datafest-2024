# LA PLANCHA

## Objetivo:

Estimar la demanda de efectivo para los cajeros automáticos (ATMs) de
las agencias del BCP. Los estudiantes deben calcular la cantidad de efectivo necesaria para satisfacer la demanda diaria en las diferentes denominaciones (20, 50, 100 soles).


## Dataset:

### EDA/Ideas de negocio:
- Los sectores ABC1 usan menos efectivo que los sectores D y E.
- La transición hacia las billeteras digitales como Yape tienen un impacto en la demanda de efectivo.
- Intentar clusterizar por georef XY o por distritos o por agencias, de acuerdo a su comportamiento.
- Segmentar los ATMs por el dinero promedio que se retira por transacción.
- Verificar estacionalidad, tendencia y estacionariedad.
- Usar binning por quantiles o por bandas para generar variables categóricas.
- Predecir el logaritmo de la demanda de efectivo. Y luego, deshacer la transformación.
- Imputar NaNs con la media del dia de la semana anterior o siguiente acorde a una ventana de tiempo con sentido.

## Forecast:
- Serie temporal que no incluye domingos ni feriados (posiblemente).
- Modelos a probar
    - ARIMA
    - SARIMA
    - SARIMAX
- Posibles covariables o variables exógenas
    - Día de la semana
    - Mes
    - Feriados
    - Tipo de cambio
    - Inflación
    - Tasa de interés
    - PBI
    - Desempleo
    - Vacaciones
- Consideraciones:
    - Para hacer el forecasting a 3 meses a veces solo es necesario 5 veces más (apróx. 2 años) dado que la data de hace 2 años captura el comportamiento de eventos como la pandemia.

## Optimización:

- Plantear el caso como un problema lineal de programación entera.

### Preguntas:
- Cual es el costo de oportunidad de mantener efectivo en los cajeros automáticos?
- Cual es el costo de oportunidad de no tener efectivo en los cajeros automáticos?

### Supuestos:

- Tasa de interés de préstamo al consumo
- Tasa flat en el año
- Ajustar la tasa de interés por la inflación
- Tasa de interés de la Reserva Federal de USA.

## Metricas
- Cantidad de dinero que se logra ahorrar.
- Validacion de supuestos.
- En cuanto se reducen los costos.