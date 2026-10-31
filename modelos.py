"""
Estrategias de recomendacion de libros.

Todas siguen la misma receta:
    estrategia = crear_xxx(datos)        # "entrena" con los datos de train
    estrategia(id_lector)                # -> lista de id_libro, de mejor a peor
Asi se pueden evaluar todas juntas en evaluacion.py y generar el CSV de
Kaggle con cualquiera de ellas.

Estrategias:
  - azar          : baseline, libros al azar que el lector no leyo.
  - popularidad   : los libros mas leidos.
  - genero        : la del notebook: libros al azar de los generos que lee el
                    lector y de los generos "vecinos" (correlacionados).
  - knn_regresion : la del notebook (mejor resultado en Kaggle): candidatos por
                    genero, recortados a los 200 mas populares, ordenados por el
                    rating que predice un KNeighborsRegressor.
  - user_knn      : filtrado colaborativo, lectores parecidos (coseno).
  - ease          : modelo lineal libro x libro (Steck, 2019).
  - hibrido       : elige la estrategia segun cuanto leyo el lector:
                    menos de 3 libros -> popularidad, de 3 a 6 -> user-kNN,
                    mas de 6 -> EASE.
"""

import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import config
from base_datos import cargar_interacciones, cargar_lectores, cargar_libros


# ===========================================================================
# Datos de entrenamiento
# ===========================================================================

def cargar_datos(ruta_db):
    """Carga todo lo que necesitan las estrategias, en un diccionario."""
    return construir_datos(cargar_interacciones(ruta_db), cargar_libros(ruta_db), cargar_lectores(ruta_db))


def construir_datos(df, libros, lectores):
    """Arma el diccionario de datos a partir de las tablas ya cargadas.
    (Separado de cargar_datos para poder armarlo con un pedazo de df,
    por ejemplo solo las lecturas anteriores a una fecha.)"""
    en_catalogo = set(libros["id_libro"])
    return {
        "df": df,
        "libros": libros,
        "lectores": lectores,
        "catalogo": libros["id_libro"].tolist(),
        "genero_de_libro": libros.set_index("id_libro")["genero_libro"].to_dict(),
        "genero_de_lector": lectores.set_index("id_lector")["genero_lector"].to_dict(),
        "autor_de_libro": libros.set_index("id_libro")["autor"].to_dict(),
        "leidos": df.groupby("id_lector")["id_libro"].apply(set).to_dict(),
        # solo libros del catalogo (hay unas pocas lecturas de libros que no estan en 'libros')
        "populares": [b for b in df["id_libro"].value_counts().index if b in en_catalogo],
    }


def completar_con_populares(recomendados, datos, id_lector, n):
    """Si una estrategia no llega a n libros, completa con los mas populares
    que el lector no leyo (y que no estan ya en la lista)."""
    if len(recomendados) >= n:
        return recomendados
    leidos = datos["leidos"].get(id_lector, set())
    ya = set(recomendados)
    extra = [b for b in datos["populares"] if b not in leidos and b not in ya]
    return recomendados + extra[:n - len(recomendados)]


# ===========================================================================
# Baselines
# ===========================================================================

def crear_azar(datos, n=config.N_RECOMENDACIONES, semilla=config.SEMILLA):
    azar = random.Random(semilla)

    def estrategia(id_lector):
        leidos = datos["leidos"].get(id_lector, set())
        recomendados = []
        # Sorteamos libros hasta juntar n no leidos. El tope de intentos evita
        # que se cuelgue si el lector ya leyo casi todo el catalogo.
        for _ in range(100 * n):
            if len(recomendados) == n:
                break
            libro = azar.choice(datos["catalogo"])
            if libro not in leidos and libro not in recomendados:
                recomendados.append(libro)
        return recomendados

    return estrategia


def crear_popularidad(datos, n=config.N_RECOMENDACIONES):
    def estrategia(id_lector):
        return completar_con_populares([], datos, id_lector, n)
    return estrategia


# ===========================================================================
# Estrategias del notebook: candidatos por genero
# ===========================================================================

def calcular_generos_vecinos(df, umbral=config.UMBRAL_CORRELACION_GENERO, top_n=config.TOP_GENEROS_VECINOS):
    """Para cada genero, los generos cuya lectura esta mas correlacionada
    entre lectores (los que leen A tambien suelen leer B)."""
    tabla_lector_genero = pd.crosstab(df["id_lector"], df["genero_libro"])
    correlacion = tabla_lector_genero.corr()
    vecinos = {}
    for genero in correlacion.columns:
        similares = correlacion[genero].drop(genero).sort_values(ascending=False)
        vecinos[genero] = set(similares[similares >= umbral].head(top_n).index)
    return vecinos


def candidatos_por_genero(datos, id_lector, generos_vecinos, libros_por_genero):
    """Libros no leidos de los generos del lector y de sus generos vecinos."""
    generos_lector = datos["generos_por_lector"].get(id_lector, set())
    generos_a_buscar = set(generos_lector)
    for genero in generos_lector:
        generos_a_buscar |= generos_vecinos.get(genero, set())
    candidatos = set()
    for genero in generos_a_buscar:
        candidatos |= libros_por_genero.get(genero, set())
    return candidatos - datos["leidos"].get(id_lector, set())


def preparar_generos(datos):
    """Calcula (una sola vez) lo que necesitan las estrategias por genero."""
    df = datos["df"]
    if "generos_por_lector" not in datos:
        datos["generos_por_lector"] = df.groupby("id_lector")["genero_libro"].apply(set).to_dict()
        datos["libros_por_genero"] = df.groupby("genero_libro")["id_libro"].apply(set).to_dict()
        datos["generos_vecinos"] = calcular_generos_vecinos(df)


def crear_genero(datos, n=config.N_RECOMENDACIONES, semilla=config.SEMILLA):
    """Libros al azar entre los candidatos por genero."""
    preparar_generos(datos)

    def estrategia(id_lector):
        candidatos = candidatos_por_genero(datos, id_lector, datos["generos_vecinos"], datos["libros_por_genero"])
        candidatos = sorted(candidatos)  # orden fijo antes de sortear, para que sea reproducible
        recomendados = random.Random(semilla).sample(candidatos, min(n, len(candidatos)))
        return completar_con_populares(recomendados, datos, id_lector, n)

    return estrategia


def crear_knn_regresion(datos, n=config.N_RECOMENDACIONES, escalar=False):
    """El modelo del notebook: predice el rating con un KNeighborsRegressor y
    recomienda los candidatos con mayor rating predicho.

    Features: genero del libro (agrupado), genero del lector, rating promedio
    del libro, rating promedio del lector y cantidad de lecturas del libro.
    Todas se calculan SOLO con train (sin fuga de datos).

    escalar=True estandariza las features numericas antes del KNN (en el
    notebook no se escalan: ver README)."""
    preparar_generos(datos)
    df = datos["df"].copy()

    # Generos de libro poco frecuentes -> "Otros"
    conteo = df["genero_libro"].value_counts()
    frecuentes = set(conteo[conteo >= config.UMBRAL_GENERO_FRECUENTE].index)

    def agrupar(genero):
        return genero if genero in frecuentes else "Otros"

    promedio_libro = df.groupby("id_libro")["rating"].mean().to_dict()
    promedio_lector = df.groupby("id_lector")["rating"].mean().to_dict()
    lecturas_libro = df.groupby("id_libro")["rating"].count().to_dict()
    promedio_global = df["rating"].mean()

    def armar_features(filas):
        """filas: DataFrame con id_libro, id_lector, genero_libro, genero_lector."""
        X = pd.concat([
            pd.get_dummies(filas["genero_libro"].map(agrupar), prefix="genero_libro"),
            pd.get_dummies(filas["genero_lector"], prefix="genero_lector"),
            filas["id_libro"].map(promedio_libro).fillna(promedio_global).rename("rating_promedio_libro"),
            filas["id_lector"].map(promedio_lector).fillna(promedio_global).rename("rating_promedio_lector"),
            filas["id_libro"].map(lecturas_libro).fillna(0).rename("cantidad_interacciones_libro"),
        ], axis=1)
        return X.astype(float)

    X = armar_features(df)
    columnas = X.columns
    numericas = ["rating_promedio_libro", "rating_promedio_lector", "cantidad_interacciones_libro"]
    escalador = None
    if escalar:
        escalador = StandardScaler().fit(X[numericas])
        X[numericas] = escalador.transform(X[numericas])

    knn = KNeighborsRegressor(n_neighbors=config.K_REGRESION)
    knn.fit(X.values, df["rating"].values)

    def estrategia(id_lector):
        candidatos = candidatos_por_genero(datos, id_lector, datos["generos_vecinos"], datos["libros_por_genero"])
        # Recortamos a los MAX_CANDIDATOS mas leidos antes de predecir (si no, tarda horas)
        candidatos = sorted(candidatos, key=lambda b: (-lecturas_libro.get(b, 0), b))[:config.MAX_CANDIDATOS]
        if not candidatos:
            return completar_con_populares([], datos, id_lector, n)

        filas = pd.DataFrame({
            "id_libro": candidatos,
            "id_lector": id_lector,
            "genero_libro": [datos["genero_de_libro"].get(b) for b in candidatos],
            "genero_lector": datos["genero_de_lector"].get(id_lector, "-"),
        })
        Xc = armar_features(filas).reindex(columns=columnas, fill_value=0)
        if escalador is not None:
            Xc[numericas] = escalador.transform(Xc[numericas])
        prediccion = knn.predict(Xc.values)
        orden = np.argsort(-prediccion, kind="stable")
        recomendados = [candidatos[k] for k in orden[:n]]
        return completar_con_populares(recomendados, datos, id_lector, n)

    return estrategia


# ===========================================================================
# Filtrado colaborativo (matriz binaria lector x libro)
# ===========================================================================

def armar_matriz(datos, min_lecturas=None):
    """Matriz binaria lector x libro (1 = lo leyo), solo con los libros que
    tienen al menos min_lecturas lecturas (por defecto, config.MIN_LECTURAS_LIBRO).
    Devuelve (X, lectores, libros)."""
    if min_lecturas is None:
        min_lecturas = config.MIN_LECTURAS_LIBRO
    df = datos["df"]
    lecturas = df["id_libro"].value_counts()
    en_catalogo = set(datos["catalogo"])
    libros = sorted(b for b in lecturas[lecturas >= min_lecturas].index if b in en_catalogo)
    lectores = sorted(df["id_lector"].unique())
    fila_de = {u: f for f, u in enumerate(lectores)}
    columna_de = {b: c for c, b in enumerate(libros)}

    X = np.zeros((len(lectores), len(libros)), dtype=np.float32)
    for u, b in zip(df["id_lector"], df["id_libro"]):
        if b in columna_de:
            X[fila_de[u], columna_de[b]] = 1
    return X, lectores, libros


def quedarse_con_los_k_mayores(matriz, k):
    """En cada fila deja los k valores mas altos y pone 0 en el resto."""
    resultado = matriz.copy()
    if k < matriz.shape[1]:
        posiciones_a_borrar = np.argpartition(-matriz, k, axis=1)[:, k:]
        np.put_along_axis(resultado, posiciones_a_borrar, 0, axis=1)
    return resultado


def estrategia_desde_puntajes(puntajes, X, lectores, libros, datos, n):
    """Convierte una matriz de puntajes (lectores x libros) en una estrategia."""
    fila_de = {u: f for f, u in enumerate(lectores)}

    def estrategia(id_lector):
        if id_lector not in fila_de:  # lector sin historial: popularidad
            return completar_con_populares([], datos, id_lector, n)
        f = fila_de[id_lector]
        p = puntajes[f].copy()
        p[X[f] > 0] = -np.inf  # no recomendar lo que ya leyo
        cuantos = min(n, len(p) - 1)
        mejores = np.argpartition(-p, cuantos)[:n] if cuantos > 0 else np.arange(len(p))
        mejores = mejores[np.argsort(-p[mejores], kind="stable")]
        recomendados = [libros[c] for c in mejores if p[c] != -np.inf]
        return completar_con_populares(recomendados, datos, id_lector, n)

    return estrategia


def entrenar_user_knn(datos, k=config.K_VECINOS):
    """user-kNN con similitud coseno: puntaje(lector, libro) = suma de las
    similitudes de los k lectores mas parecidos que leyeron ese libro.
    Devuelve (puntajes, X, lectores, libros)."""
    X, lectores, libros = armar_matriz(datos)
    largos = np.sqrt(X.sum(axis=1))
    largos[largos == 0] = 1
    Xn = X / largos[:, None]
    puntajes = np.zeros_like(X)
    # Por bloques de lectores, para no armar de una la matriz 10.000 x 10.000
    for inicio in range(0, X.shape[0], 1000):
        similitud = Xn[inicio:inicio + 1000] @ Xn.T
        for f in range(similitud.shape[0]):
            similitud[f, inicio + f] = 0  # un lector no es vecino de si mismo
        vecinos = quedarse_con_los_k_mayores(similitud, k)
        puntajes[inicio:inicio + 1000] = vecinos @ X
    return puntajes, X, lectores, libros


def crear_user_knn(datos, n=config.N_RECOMENDACIONES, k=config.K_VECINOS):
    puntajes, X, lectores, libros = entrenar_user_knn(datos, k)
    return estrategia_desde_puntajes(puntajes, X, lectores, libros, datos, n)


def entrenar_ease(datos, lambda_=config.LAMBDA_EASE):
    """EASE: pesos libro x libro con formula cerrada (regresion ridge con la
    diagonal forzada a 0, para que un libro no se recomiende a si mismo).
    Devuelve (puntajes, X, lectores, libros)."""
    X, lectores, libros = armar_matriz(datos)
    G = (X.T @ X).astype(np.float64)
    G[np.diag_indices_from(G)] += lambda_
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0)
    puntajes = X @ B.astype(np.float32)
    return puntajes, X, lectores, libros


def crear_ease(datos, n=config.N_RECOMENDACIONES, lambda_=config.LAMBDA_EASE):
    puntajes, X, lectores, libros = entrenar_ease(datos, lambda_)
    return estrategia_desde_puntajes(puntajes, X, lectores, libros, datos, n)


# ===========================================================================
# Ensamble por actividad del lector
# ===========================================================================

def crear_hibrido(datos, n=config.N_RECOMENDACIONES,
                  min_para_user_knn=config.MIN_LECTURAS_USER_KNN, min_para_ease=config.MIN_LECTURAS_EASE):
    """Elige la estrategia segun cuantos libros leyo el lector:
      - menos de min_para_user_knn       -> popularidad (cold start)
      - de min_para_user_knn a min_para_ease - 1 -> user-kNN
      - min_para_ease o mas              -> EASE
    Los modelos se entrenan una sola vez y se reutilizan para todos los lectores."""
    popularidad = crear_popularidad(datos, n)
    user_knn = crear_user_knn(datos, n)
    ease = crear_ease(datos, n)

    def estrategia(id_lector):
        leidos = len(datos["leidos"].get(id_lector, ()))
        if leidos < min_para_user_knn:
            return popularidad(id_lector)
        if leidos < min_para_ease:
            return user_knn(id_lector)
        return ease(id_lector)

    return estrategia
