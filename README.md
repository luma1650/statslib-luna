# statslib-luna

Biblioteca de estadistica en Python 3.12 o superior. Incluye series, tablas de
frecuencias inmutables y acumuladores para procesar datos sin almacenarlos.

## Instalacion y pruebas

```sh
python -m pip install -e ".[test]"
python -m pytest
```

## Uso

```python
from statslib import StatsSeries, FrequencyTable, WelfordVariance

series = StatsSeries([1, 2, 2, 3, 7])
print(series.mean, series.median, series.mode)
print(series.variance(), series.stdev(ddof=1))
print(series.quartiles, series.interquartile_range)
print(series.percentile(90, method="linear"))
print(series.skewness, series.excess_kurtosis)

# Trabaja con conteos grandes sin expandir las observaciones.
table = FrequencyTable.from_frequencies({10: 1_000_000, 20: 2_000_000})
print(table.mean, table.median)
print(table.cumulative(reverse=True))

accumulator = WelfordVariance()
accumulator.extend(iter([1, 2, 3]))
print(accumulator.mean, accumulator.variance())
```

## Convenciones

- La varianza usa `ddof=0` (poblacional); `ddof=1` aplica la correccion muestral.
- `quantile(k)` usa rango mas cercano, con `0 <= k <= 1`. `method="linear"`
  interpola en `(n - 1) * k`; cuartiles e IQR usan esa interpolacion.
- `percentile(p)` acepta valores de 0 a 100.
- Asimetria y curtosis son momentos sin correccion de sesgo. La curtosis es
  excedente (se resta 3); ambas requieren varianza positiva.
- `origin_moment`, `central_moment` y `standard_moment` aceptan orden entero
  no negativo. Los momentos estandarizados requieren varianza positiva.
- El coeficiente de variacion divide la desviacion entre el valor absoluto
  de la media; una media cero produce `ValueError`.
- Las tablas conservan el orden de primera aparicion. Las acumuladas ordenan
  los valores y mantienen caches independientes para orden ascendente y
  descendente. Las claves personalizadas se recalculan en cada llamada.
- La moda elige el primer valor encontrado entre empates hashables. Para
  valores no hashables ordenables se usa orden clasificado; para los que
  solo permiten igualdad se usa orden de aparicion. La multimoda no repite valores.
- Los calculos que requieren observaciones lanzan `ValueError` con entradas
  vacias; multimoda y mapas de frecuencias vacios devuelven colecciones vacias.

## Rendimiento

Las propiedades de `StatsSeries` se calculan bajo demanda. Varianza y desviacion
comparten un acumulador; asimetria y curtosis comparten otro. Los tres cuartiles
se obtienen con una sola ordenacion. La seleccion usa particion en tres grupos
para manejar valores repetidos y limita las particiones antes de recurrir a
ordenacion. Las consultas de rango en tablas reutilizan indices acumulados y
busqueda binaria, sin expandir los conteos.


## Graficos con Matplotlib

Instala la dependencia opcional:

```sh
python -m pip install -e ".[plot]"
# Para desarrollo con pruebas de graficos:
python -m pip install -e ".[test,plot]"
```

```python
import matplotlib.pyplot as plt
from statslib import StatsSeries, FrequencyTable

series = StatsSeries([1, 2, 2, 3, 5, 8])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
series.plot(ax=axes[0, 0], marker="o", color="teal")
series.hist(bins=5, ax=axes[0, 1], edgecolor="white")
series.boxplot(ax=axes[1, 0])
series.ecdf(ax=axes[1, 1], color="purple")
fig.savefig("series.png", dpi=150)
plt.show()

# Barras para categorias, en orden de primera aparicion.
table = FrequencyTable(["rojo", "azul", "rojo", "verde"])
ax = table.bar(relative=True, color="coral")  # table.plot() tambien usa barras
ax.set_title("Frecuencias relativas")
plt.show()

# Histograma y ECDF numericos sin expandir millones de observaciones.
numeric = FrequencyTable.from_frequencies({1: 1_000_000, 4: 2_000_000})
ax = numeric.hist(bins=[0, 2, 5])
ax.figure.savefig("frecuencias.png")
numeric.ecdf()
plt.show()
```

Todos los metodos devuelven `matplotlib.axes.Axes`, aceptan `ax=` para usar
un eje existente y reenvian opciones de estilo al metodo de Matplotlib.
No llaman a `show()` ni cambian el estilo o backend global. Matplotlib solo
se importa al solicitar un grafico; las operaciones estadisticas no lo requieren.
Las entradas vacias producen `ValueError` antes de crear una figura.

- `StatsSeries`: `plot()`, `hist(bins=10)`, `boxplot()`, `ecdf()`.
- `FrequencyTable`: `bar(relative=False)`, `plot()` como alias de barras,
  `hist(bins=10)` y `ecdf()` con pesos tomados de los conteos.
- Histogramas: `density=True` permite representar densidad. En tablas, usa un
  numero de intervalos o limites explicitos; las reglas como `bins="auto"`
  no son compatibles con pesos. No se permite reemplazar los pesos de la tabla.
- La ECDF admite opciones como `complementary=True`, `orientation="horizontal"`
  y `compress=True`, segun Matplotlib (requiere version 3.8 o superior).
- Cajas, histogramas y ECDF requieren observaciones numericas; las barras
  admiten categorias. Los errores de tipos y opciones se delegan a Matplotlib.


## Metodos de FrequencyTable

Los calculos trabajan directamente con las frecuencias, sin llamar a
`to_series()` ni repetir observaciones. `len(table)` sigue contando valores
distintos y `table.total` cuenta todas las observaciones.

```python
from statslib.frequency import FrequencyTable

table = FrequencyTable.from_frequencies({10: 3, 20: 5, 30: 2})
print(table.variance(), table.stdev(ddof=1))
print(table.quartiles, table.interquartile_range)
print(table.percentile(90, method="linear"))
print(table.mean_absolute_deviation, table.median_absolute_deviation)
print(table.pmf(20))   # 0.5: probabilidad de X == 20
print(table.cdf(20))   # 0.8: probabilidad de X <= 20
print(table.most_common(2))  # [(20, 5), (10, 3)]
combined = table.merge({20: 2, 40: 1})  # Crea una tabla nueva.
```

| Metodo o propiedad | Resultado |
| --- | --- |
| `frequency(value)` | Conteo; cero si el valor no aparece |
| `relative_frequency(value)`, `pmf(value)` | Probabilidad empirica del valor |
| `cumulative_frequency(value)` | Conteo de observaciones menores o iguales |
| `cdf(value)`, `cumulative_relative_frequency(value)` | Probabilidad acumulada |
| `variance(ddof=0)`, `stdev(ddof=0)` | Varianza y desviacion, poblacionales por defecto |
| `coefficient_of_variation(ddof=0)` | Desviacion dividida entre media absoluta |
| `data_range` | Maximo menos minimo |
| `quantile(k, method="nearest")` | Cuantil para k de 0 a 1 |
| `percentile(p, method="nearest")` | Percentil para p de 0 a 100 |
| `quartiles`, `interquartile_range` | Q1, Q2, Q3 e IQR, con interpolacion lineal |
| `mean_absolute_deviation` | Desviacion absoluta media respecto de la media |
| `median_absolute_deviation` | Mediana de desviaciones absolutas respecto de la mediana |
| `most_common(n=None)` | Pares valor-conteo por frecuencia descendente |
| `merge(other)` | Tabla nueva que suma conteos con otra tabla o mapping |

`quantile` y `percentile` admiten `method="linear"` para interpolacion.
Rango mas cercano, CDF y acumuladas admiten categorias ordenables; los calculos
aritmeticos requieren valores numericos. Las consultas de posicion y CDF
reutilizan el indice ordenado: primera consulta O(m log m), siguientes O(log m),
donde m es el numero de valores distintos. La varianza usa sumas centradas y
comparte el resultado entre diferentes valores de `ddof` y desviacion estandar.

`most_common` conserva el orden de aparicion en empates. `merge` no modifica
las tablas originales, conserva el orden de las claves existentes y agrega
las nuevas al final. Valida conteos enteros no negativos y omite ceros.
`percentage()` conserva su comportamiento anterior: devuelve todo el mapa.
`table[value]` sigue lanzando `KeyError` para valores ausentes.

Las probabilidades, cuantiles y medidas numericas requieren una tabla no vacia;
con una vacia lanzan `ValueError`. `table.py` y `cumulative_frequency` devuelven
cero en tablas vacias, y `most_common` devuelve una lista vacia. `ddof` debe ser
un entero no booleano, con `0 <= ddof < total`. El coeficiente de variacion
requiere media distinta de cero. `find_pos` requiere una posicion entera.
