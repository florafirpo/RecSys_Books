"""
Configuracion del proyecto de libros: rutas y parametros, en un solo lugar.
"""

import os

# ---------------------------------------------------------------------------
# Carpetas y archivos
# ---------------------------------------------------------------------------

CARPETA_PROYECTO = os.path.dirname(os.path.abspath(__file__))

# Donde estan data.db y ejemplo.csv. Por defecto, la carpeta de arriba
# ("libros"). Se puede cambiar con la variable de entorno LIBROS_DATOS.
CARPETA_DATOS = os.environ.get("LIBROS_DATOS", os.path.dirname(CARPETA_PROYECTO))

DB_COMPLETA = os.path.join(CARPETA_DATOS, "data.db")        # base original de la competencia
DB_TRAIN = os.path.join(CARPETA_DATOS, "data_train.db")     # copia sin las reseñas de test
DB_TEST = os.path.join(CARPETA_DATOS, "data_test.db")       # solo las reseñas reservadas
CSV_LECTORES_KAGGLE = os.path.join(CARPETA_DATOS, "ejemplo.csv")  # los 832 lectores a predecir

CARPETA_SUBMISSIONS = os.path.join(CARPETA_PROYECTO, "submissions")

# ---------------------------------------------------------------------------
# Parametros generales
# ---------------------------------------------------------------------------

# Ratings de 1 a 10. Desde cuanto consideramos que el libro "le gusto".
# (El 50% de las reseñas tiene 8 o mas.)
RATING_POSITIVO = 8

# Cuantos libros se recomiendan (la competencia pide 20)
N_RECOMENDACIONES = 20

# ---------------------------------------------------------------------------
# Separacion train / test y evaluacion
# ---------------------------------------------------------------------------

FRACCION_TEST = 0.2     # que parte de las reseñas de cada lector va a test (80/20 por lector)
MIN_QUE_QUEDAN = 3      # minimo de reseñas que le dejamos en train a cada lector
SEMILLA = 42            # fija el azar, asi la separacion es reproducible

# Hay ~10.000 lectores y el KNN de regresion tarda; evaluamos sobre una
# muestra fija de lectores de test (siempre la misma, por la semilla).
USUARIOS_A_EVALUAR = 1000

# ---------------------------------------------------------------------------
# Parametros de los modelos (modelos.py)
# ---------------------------------------------------------------------------

# Estrategia por genero / KNN de regresion (los del notebook original)
UMBRAL_GENERO_FRECUENTE = 100   # generos con menos lecturas se agrupan en "Otros"
UMBRAL_CORRELACION_GENERO = 0.15
TOP_GENEROS_VECINOS = 3
MAX_CANDIDATOS = 200            # candidatos por lector antes de predecir con el KNN
K_REGRESION = 20                # vecinos del KNeighborsRegressor

# Filtrado colaborativo: solo libros con al menos esta cantidad de lecturas
# en train (con 48.000 libros la matriz completa no entra en memoria para EASE)
MIN_LECTURAS_LIBRO = 10
K_VECINOS = 50          # user-kNN
LAMBDA_EASE = 500       # EASE

# Hibrido: que estrategia usar segun cuantos libros leyo el lector
MIN_LECTURAS_USER_KNN = 3   # menos de 3 libros -> popularidad
MIN_LECTURAS_EASE = 7       # de 3 a 6 -> user-kNN; 7 o mas -> EASE

# ---------------------------------------------------------------------------
# Recomendador en dos etapas (dos_etapas.py)
# ---------------------------------------------------------------------------

# Retrieval: cuantos candidatos aporta cada fuente
K_CAND_EASE = 200
K_CAND_USER_KNN = 200
K_CAND_RECIENTES = 100   # mas leidos en el ultimo año
K_CAND_AUTOR = 50        # de autores que el lector ya leyo

# Ranking: corte temporal interno para entrenar el modelo de ranking.
# Evaluacion (train = hasta 2023): aprende con lo leido en 2023.
# Kaggle (todos los datos): aprende con lo leido en 2024.
CORTE_RANKER_EVALUACION = "2023-01-01"
CORTE_RANKER_KAGGLE = "2024-01-01"
MAX_LECTORES_RANKER = 1500   # lectores para entrenar el ranking (para que no tarde demasiado)
