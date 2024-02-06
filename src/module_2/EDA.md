<h1 align="center">Exploratory Data Analysis</h1>

Libraries


```python
import pandas as pd
import pyarrow
import matplotlib.pyplot as plt
import seaborn as sns
```

Descargamos el DataSet y visualizamos las primeras filas que contiene:


```python
file_path = '/home/oscar/data/feature_frame.csv'
df = pd.read_csv(file_path, low_memory=False)
#print(df.head())
```

**Quick checks**


```python
print("Número de filas y columnas:", df.shape)
```


```python
print(df.info())
```

Observamos que no hay valores nulos ni nan a priori en el Dataset para ninguna de las variables:


```python
print(df.isnull().sum())
```


```python
print(df.describe())
```

Tras revisar los datos tenemos un conjunto con 27 características y 2880549 observaciones. Ninguna de las características presenta null-values. Evaluamos los diferentes datos y revisamos los datatypes:

1. **Información general del pedido y usuario:**
- 'order_id' : identificador único del pedido (int64)
- 'user_id' : identificado único del usuario (int64)
- 'created_at' : fecha y hora de creación del registro (object)
- 'order_date' : fecha del pedido (object)

El identificador único del pedido y del usuario debería ser tratado como una variable categórica y las diferentes fechas del pedido y de creación del registro deben de ser de tipo datetime:


```python
df['order_id'] = df['order_id'].astype('object')
df['user_id'] = df['user_id'].astype('object')
df['created_at'] = pd.to_datetime(df['created_at'])
df['order_date'] = pd.to_datetime(df['order_date'])
```

Evalúo el número de órdenes únicas y de usuarios únicos, tengo 3446 números de orden únicos y 1937 usuarios únicos.


```python
# Explorar 'order_id'
print("Número de órdenes únicas:", len(df['order_id'].unique()))
print(df['order_id'].value_counts())

# Explorar 'user_id'
print("Número de usuarios únicos:", len(df['user_id'].unique()))
print(df['user_id'].value_counts())
```

Se observa que hay usuarios repetidos que realizan más de un pedido. Estas variables se toman como categóricas porque son un identificador -> ¿Luego las codifico meterlas al modelo con un label encoding por ejemplo?


```python
# Histograma de 'created_at'
df['created_at'].hist(bins=20, edgecolor='black')
plt.xlabel('Fecha y Hora de Creación')
plt.ylabel('Frecuencia')
plt.title('Distribución de Registros por Fecha y Hora de Creación')
plt.show()

# Histograma de 'order_date'
df['order_date'].hist(bins=20, edgecolor='black')
plt.xlabel('Fecha de Pedido')
plt.ylabel('Frecuencia')
plt.title('Distribución de Pedidos por Fecha')
plt.show()
```

Hay una tendencia ascendente de la fecha del pedido y la fecha y hora de creación desde el 10-2020 hasta el 03-2021. Esto indica que se han comprado mayor cantidad de productos últimamente o que, la mayor parte de los datos se concentran en fechas de 2021.

**2. Información del producto y variante:**

- 'variant_id': Identificador único de la variante del producto. (int64)
- 'product_type': Tipo de producto. (object)
- 'vendor': Proveedor del producto. (object)
- 'normalised_price': Precio normalizado del producto. (float64)
- 'discount_pct': Porcentaje de descuento aplicado. (float64)

Vemos que hay 976 variantes únicas de los productos. En este sentido, las más vistas aparecen 3446 veces mientras que la menos vista aparece únicamente 25 veces en el DataSet.


```python
df['variant_id'] = df['variant_id'].astype('object')
print("Número de variantes únicas:", len(df['variant_id'].unique()))
print(df['variant_id'].value_counts())
```

Vemos el número de tipos de productos distintos y de vendedores que existen. Existen 62 tipos de productos distintos y 264 vendedores, destacando entre los productos los alimentos enlatados como los más vistos, seguido de condimentos y aderezos y, a su vez, de arroz, pasta y legumbres.
En cuanto a las marcas más vistas destacan biona, ecover y method.


```python
print("Número de tipos de producto únicos:", len(df['product_type'].unique()))
print(df['product_type'].value_counts())

print("Número de tipos de vendedores:", len(df['vendor'].unique()))
print(df['vendor'].value_counts())
```

Al analizar el precio normalizado, su valor máximo es 1, siendo este el producto más caro. Los precios están normalizados entre 0 y 1 para facilitar las comparaciones probablemente. En cuanto al descuento, hay cifras de descuento menores a cero y mayores a 1 lo cual, a priori, no tendría sentido, estaríamos hablando de un descuento superior al 100% y no tiene sentido hablar de un descuento negativo.


```python
print("Estadísticas de 'normalised_price':")
print(df['normalised_price'].describe())

print("\nEstadísticas de 'discount_pct':")
print(df['discount_pct'].describe())
```

Evaluamos los casos en los que tenemos un descuento  superior a 1 e inferior a 0:


```python
unusual_dicounts_df_negative = df[(df['discount_pct'] < 0)]
print(len(unusual_dicounts_df_negative.index))

unusual_dicounts_df_up_to_1 = df[(df['discount_pct'] > 1)]
print(len(unusual_dicounts_df_up_to_1.index))

```

Vemos que hay 1587 casos con un descuento negativo y 58582 casos con un descuento superior a 1. Habría que preguntar al negocio si esto realmente tiene sentido.

**3. Características del usuario y comportamiento de compra:** 

- user_order_seq: Secuencia de orden del usuario. (int64)
- outcome: Indica si el producto fue comprado o no en ese pedido. (float64)
- ordered_before: Indica si el producto fue ordenado previamente. (float64)
- abandoned_before: Indica si el producto fue abandonado anteriormente. (float64)
- active_snoozed: Indica si el producto está activo o en modo de espera. (float 64)
- set_as_regular: Indica si el producto se estableció como regular. (float 64)


```python
print("Número secuencia de orden del usuario:", len(df['user_order_seq'].unique()))
print(df['user_order_seq'].value_counts())
```

La variable 'user_order_seq' está directamente relacionada con el usuario e indica el número de pedidos que ha realizado cada uno de los usuarios. Se observa una clara tendencia decreciente a medida que aumenta el número de orden, concentrándose la mayor parte de usuarios en 2 pedidos.

Visualizo esa tendencia decreciente de la secuencia de orden de los usuarios:


```python
unique_seq_lengths = len(df['user_order_seq'].unique())
seq_count = df['user_order_seq'].value_counts()

plt.bar(seq_count.index, seq_count.values, color='blue', alpha=0.7)

plt.xlabel('Número de Secuencias de Orden')
plt.ylabel('Número de Usuarios')
plt.title(f'Histograma de Número de Secuencias de Orden por Usuario (Total: {unique_seq_lengths} longitudes únicas)')

plt.show()
```

Ahora voy a ver cómo se comportan el resto de variables binarias. Están codificadas todas ellas como float pero, al tratase de variables binarias, por simplicidad, se van a tratar como int.


```python
binary_variables = ['outcome', 'ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
df[binary_variables] = df[binary_variables].astype(int)

for var in binary_variables:
    print(f'Valores únicos y frecuencia para {var}:\n{df[var].value_counts()}\n')
```

Se observa que, del total de datos, únicamente 33000 registros aproximadamente acaban comprando el producto y que, la mayoría de los productos no se establecen como regulares, ni fueron abandonados previamente ni están en espera ni fueron ordenados previamente.

**4. Información sobre la popularidad y conteo de productos:**

- global_popularity: Popularidad global del producto. (float64)
- count_adults, count_children, count_babies, count_pets: Cantidad de adultos, niños, bebés y mascotas en el pedido. (float64)
- people_ex_baby: Cantidad de personas excluyendo bebés en el pedido. (float64)

Evaluamos primero los diferentes grupos demográficos asociados a cada pedido. Como conclusión se ve que la mayor parte de los pedidos están asociados a dos adultos sin niños, bebés ni mascotas aunque hay excepciones, llegando a 5 niños, 1 bebé o incluso 6 mascotas.


```python
for var in ['count_adults', 'count_children', 'count_babies', 'count_pets', 'people_ex_baby']:
    print(f"Valores únicos y frecuencia para {var}:\n{df[var].value_counts()}\n")

```

Evaluamos ahora la variable de la popularidad global del producto y cómo se distribuye:


```python
sns.set(style="whitegrid")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x=df['global_popularity'])
plt.title('Boxplot de global_popularity')
plt.xlabel('Puntuación de Popularidad Global')

plt.subplot(1, 2, 2)
sns.kdeplot(df['global_popularity'], fill=True)
plt.title('Distribución de global_popularity (KDE)')
plt.xlabel('Puntuación de Popularidad Global')
plt.ylabel('Densidad')

# Ajustes de diseño
plt.tight_layout()
plt.show()
```

Tras visualizar la distribución de los datos, vemos que la mayor parte de ellos se concentran entre 0 y 0.1. No obstante, hay varios ejemplos que salen del rango intercuartílico*1.5 y podrían ser considerados outliers pero, en este caso, dado que son los productos de mayor popularidad, no tendría sentido quitarlos ya que resultan información muy valiosa.

**5. Información temporal, todos float64:**

- days_since_purchase_variant_id: Días desde la compra para la variante específica.
- avg_days_to_buy_variant_id: Promedio de días para comprar la variante específica.
- std_days_to_buy_variant_id: Desviación estándar de días para comprar la variante específica.
- days_since_purchase_product_type: Días desde la compra para el tipo de producto.
- avg_days_to_buy_product_type: Promedio de días para comprar el tipo de producto.
- std_days_to_buy_product_type: Desviación estándar de días para comprar el tipo de producto.


```python
temporal_variables = [
    'days_since_purchase_variant_id', 'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
    'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 'std_days_to_buy_product_type'
]

plt.figure(figsize=(14, 8))

for i, var in enumerate(temporal_variables, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(df[var], fill=True)
    plt.title(f'Distribución de {var}')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')

plt.tight_layout()
plt.show()
```


```python
print(df[[
    'days_since_purchase_variant_id', 'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
    'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 'std_days_to_buy_product_type'
]].describe())
```

Tras comprobar los valores, parece que todos los valores son coherentes y no hay valores negativos. Además, se observa en las distintas distribuciones que la media y la distribución típica de los días para comprar una variante específica se corresponde a una distribución muy similar a una normal, mientras que para el tipo de producto esta distribución es más variable. Con respecto a la distribución de los días desde la última compra para cada producto y cada variante, la distribución es prácticamente una normal y la mayor parte de sus valores se concentran en los 33 días. 

Terminando con las últimas comprobaciones, vamos a ver si existen filas u observaciones duplicadas:


```python
print(df.duplicated().sum())
```

Tras la corrección de los diferentes Datatypes y el análisis univariante observamos los diferentes tipos de nuestras variables:


```python
print(df.info())
```

**ANÁLISIS MULTIVARIABLE**


```python
df_numeric = df.select_dtypes(include='number')

correlation_matrix = df_numeric.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title('Matriz de Correlaciones')
plt.show()
```

Las correlaciones a destacar son las siguientes:
- Correlación de +0.84 entre 'avg_days_to_buy_product_type' y 'std_days_to_buy_product_type' : esto indica que hay una correlación lineal entre la media de días que se compra un determinado producto y la regularidad con la que se compra dicho producto. Es decir, de forma general, cuanto mayor es la media de días que se tarda en comprar un producto, menor es la regularidad con la que se compra dicho producto.
- Hay una correlación people_ex_babies y las variables count_adults y count_children. Esto se debe a que, probablmente, esta variable esté formada por una combinación lineal de las anteriores, lo cual tiene sentido porque la variable toma las personas sin contar los bebés.

El resto de las correlaciones son muy bajas, por lo que no se pueden extraer resultados significativos.

**ENCODING: HAY QUE PENSAR CÓMO CODIFICO CADA VARIABLE**
- La marca ¿label encoding? ¿por frecuencia?, no tiene sentido one hot, demasiadas marcas ¿target encoding con la probabilidad de que no se compre la marca? (no sé si tiene sentido hacer esto último)

- Los id ¿cómo codificarlos? Si los codifico directamente con su valor numérico, valores demasiado altos, ¿solución?

**TEST DE HIPÓTESIS: HAY QUE PENSAR CÓMO AFECTA LA VARIABLE RESPUESTA EN EL RESTO DE VARIABLES**
- ¿Hay diferencias estadísticamente significativas en la popularidad de un producto y que se compre o no? -> entiendo que sí, hay que comprobarlo
- ¿Hay diferencias estadísticamente significativas en la distribución de la media de días en comprar un producto o de su variabilidad a que se compre o no? -> entiendo que sí, que los productos más comprados se compran de forma más regular y en menos días
- ¿Hay diferencias estadísticamente signficativas entre los productos con más descuento y los productos con menos descuento? Es decir, ¿los productos más comprados son los que más descuento tienen o los que menor precio tienen?
- ¿ Hay diferencias estadísticamente significativas en las horas a las que se realizan los pedidos? ¿Hay alguna hora a la que se realicen más pedidos o se distribuyen igual y son independientes de la hora?

