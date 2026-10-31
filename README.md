# Recomendador de libros (competencia de Kaggle)

Trabajo de la materia **Sistemas de Recomendación** (Especialización en Minería de Datos y Descubrimiento del Conocimiento, UBA). La consigna es recomendar **20 libros** a cada uno de **832 lectores**, y Kaggle lo evalúa con **NDCG@20**.

**Autora:** María Florencia Firpo

El análisis original está en el notebook de Colab (`../sistems_de_recomendacion.py`). Este proyecto lo reorganiza en módulos, le agrega una evaluación offline por lector y suma dos modelos de filtrado colaborativo.

## Resultados

Evaluación offline, con 1.000 lectores de test elegidos al azar (siempre los mismos) y N = 20:

| Estrategia | NDCG@20 | precision@20 | recall@20 | hit rate@20 |
|---|---|---|---|---|
| azar | 0,0003 | 0,0003 | 0,0003 | 0,006 |
| genero (notebook) | 0,0004 | 0,0004 | 0,0005 | 0,008 |
| knn_regresion (notebook) | 0,0619 | 0,0380 | 0,0891 | 0,394 |
| knn_regresion escalado | 0,0610 | 0,0387 | 0,0794 | 0,376 |
| popularidad | 0,1128 | 0,0629 | 0,1132 | 0,458 |
| user-kNN (k=50) | 0,2394 | 0,1064 | 0,2323 | 0,621 |
| **EASE (λ=500)** | **0,2949** | **0,1328** | **0,2902** | **0,688** |

Diferencia de NDCG@20 contra popularidad, con intervalo de confianza del 95 % (*bootstrap* pareado):

| Estrategia | Diferencia | IC 95 % |
|---|---|---|
| EASE | +0,182 | [+0,167; +0,198] |
| user-kNN | +0,127 | [+0,112; +0,141] |
| knn_regresion (notebook) | −0,051 | [−0,062; −0,040] |

### Resultados en Kaggle

| Submission | NDCG@20 (Kaggle) | vs mejor anterior |
|---|---|---|
| KNN de regresión, versión base (notebook) | 0,02247 | referencia |
| KNN de regresión con afinidad (notebook) | 0,02105 | −6 % |
| user-kNN | 0,03180 | +42 % |
| **EASE** | **0,04175** | **+86 %** |

El orden coincide con la evaluación offline (EASE > user-kNN > KNN de regresión), así que la evaluación offline sirve para elegir modelos. Los valores absolutos son mucho más bajos en Kaggle, porque su test es más difícil (ver la advertencia más abajo).

### Qué dicen estos números

1. **EASE es la mejor estrategia por lejos**: casi triplica el NDCG@20 de la popularidad. En el proyecto de recetas la mejora era de apenas +17 %. La diferencia es que acá hay más datos: 10.000 lectores con lecturas muy variadas, contra 501 usuarios con gustos parecidos.
2. **El KNN de regresión del notebook rinde peor que recomendar lo más popular.** El modelo predice qué rating le pondría el lector a un libro, pero la competencia pregunta qué libros va a leer. Un libro poco leído con promedio 9 sube en el ranking, aunque casi nadie lo elija. Además, los candidatos salen de un recorte a los 200 libros más leídos de sus géneros, así que el modelo nunca ve el resto.
3. **Escalar las variables del KNN no cambia nada.** En el notebook, `cantidad_interacciones_libro` (hasta 2.230) domina la distancia frente a los promedios (1 a 10). Probamos estandarizar las variables y el resultado es el mismo, así que el problema es el enfoque de regresión de ratings, no la escala.
4. **La estrategia por género sola no sirve** (0,0004, igual que el azar). Elegir al azar dentro de los géneros del lector sigue dejando miles de candidatos.

### Ojo con comparar estos números con Kaggle

El mejor puntaje en Kaggle fue 0,0225 con el KNN de regresión. Acá ese mismo modelo da 0,062. No son comparables: el test de Kaggle está oculto, tiene otros lectores y probablemente otro criterio, quizás temporal o con todas las lecturas y no solo las positivas. Esta evaluación sirve para **ordenar** las estrategias. La prueba definitiva es subir los CSV a Kaggle (`python main.py kaggle --todas`).

## Cómo está organizado

```
sistema_recomendacion_libros/
├── main.py          punto de entrada: split, evaluar, kaggle
├── config.py        rutas y parámetros
├── base_datos.py    carga de data.db (renombra genero -> genero_libro / genero_lector)
├── dataset.py       separación train / test 80/20 por lector
├── modelos.py       las 7 estrategias
├── evaluacion.py    NDCG, precision, recall, hit rate e intervalos de confianza
├── tests/           tests automáticos (pytest)
└── submissions/     CSV para subir a Kaggle (los genera main.py kaggle)
```

Los datos (`data.db` y `ejemplo.csv`) van en la carpeta de arriba (`libros/`).

## Uso

```bash
pip install pandas numpy scikit-learn pytest

python main.py split                 # separa data.db en data_train.db y data_test.db
python main.py evaluar               # tabla de resultados (tarda unos 8 minutos, casi todo es el KNN de regresión)
python main.py kaggle ease           # submissions/submission_ease.csv
python main.py kaggle --todas        # un CSV por estrategia
python -m pytest                     # tests
```

Para `kaggle` hace falta `ejemplo.csv` (la lista de los 832 lectores) en la carpeta `libros/`. Ese comando entrena con **toda** la base, no solo con train.

## Metodología

- **Split 80/20 por lector.** A cada lector se le esconde el 20 % de sus lecturas, elegidas entre las positivas (rating ≥ 8; el 50 % de las lecturas lo cumple). Cada lector conserva al menos 3 lecturas en train. Quedan 89.559 lecturas en test, repartidas entre 7.185 lectores.
  - Esto reemplaza al `train_test_split` por filas del notebook. Ese split mezcla las filas de todos los lectores y sirve para medir RMSE, pero no responde la pregunta de la competencia: qué 20 libros recomendarle a cada lector.
- **Sin fuga de datos.** Todas las variables (promedios, cantidad de lecturas, correlación entre géneros, matrices) se calculan solo con train.
- **Filtrado colaborativo con matriz binaria** (leyó / no leyó), usando solo los libros con al menos 10 lecturas en train: unos 6.600 libros que cubren el 79 % de las lecturas. Con los 48.000 libros, la matriz de EASE no entraría en memoria.

## Estrategias

| Estrategia | Idea |
|---|---|
| `azar` | Libros al azar que el lector no leyó |
| `popularidad` | Los libros más leídos |
| `genero` | Del notebook: al azar entre los géneros del lector y sus géneros vecinos (correlación ≥ 0,15, top 3) |
| `knn_regresion` | Del notebook: candidatos por género, recortados a los 200 más leídos, ordenados por el rating que predice un `KNeighborsRegressor` (k = 20) |
| `user-kNN` | Recomienda lo que leyeron los 50 lectores más parecidos (similitud coseno) |
| `EASE` | Modelo lineal libro × libro con fórmula cerrada (Steck, 2019) |

## Tests

Cubren:
- **Split anti-leakage**: ninguna lectura queda en train y test a la vez, test solo tiene lecturas positivas, se respeta el 20 %, la separación es reproducible y no modifica la base original.
- **Métricas**: NDCG calculado a mano, y que premie acertar arriba.
- **Estrategias**: ninguna recomienda libros ya leídos ni repite libros, un lector desconocido recibe popularidad y user-kNN recomienda lo que leyó el lector más parecido.

## Limitaciones y próximos pasos

- `K_VECINOS` y `LAMBDA_EASE` no se ajustaron para este dataset: son los valores del proyecto de recetas. Con una validación tomada de train, EASE probablemente mejore.
- Para lectores sin historial, todas las estrategias caen en popularidad.
- El catálogo de EASE y user-kNN excluye los libros con menos de 10 lecturas, así que esas estrategias nunca los recomiendan.
