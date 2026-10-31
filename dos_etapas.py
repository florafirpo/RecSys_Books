"""
Recomendador en dos etapas: retrieval (generacion de candidatos) + ranking.

1) RETRIEVAL: para cada lector se juntan unos ~350 candidatos de varias fuentes:
     - los mejores K_CAND_EASE segun EASE
     - los mejores K_CAND_USER_KNN segun user-kNN
     - los K_CAND_RECIENTES libros mas leidos en el ultimo año
     - hasta K_CAND_AUTOR libros de autores que el lector ya leyo
   Objetivo: que el libro correcto este entre los candidatos (recall).

2) RANKING: un modelo (HistGradientBoostingClassifier de scikit-learn) mira
   cada par lector-candidato con varias variables y estima la probabilidad
   de que el lector lo lea. Se recomiendan los N con mayor probabilidad.
   Objetivo: el orden (NDCG).

¿Como se entrena el ranking sin hacer trampa? Con un corte temporal DENTRO
de los datos disponibles: las lecturas anteriores a 'corte_interno' sirven
para armar candidatos y variables, y las lecturas positivas posteriores son
la respuesta ("lo leyo / no lo leyo"). Despues, para recomendar, se vuelven
a armar candidatos y variables con TODOS los datos disponibles.

Diferencia con el KNN de regresion del notebook: aca el modelo aprende a
predecir SI el lector va a leer el libro (lo que mide la competencia), no
que rating le pondria.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import config
import modelos

VARIABLES = [
    "ease_puntaje", "ease_puesto", "uknn_puntaje", "uknn_puesto",
    "de_ease", "de_uknn", "de_recientes", "de_autor",
    "log_lecturas_libro", "log_lecturas_recientes", "rating_promedio_libro",
    "libros_leidos_del_autor", "afinidad_genero", "log_lecturas_lector",
]


# ===========================================================================
# Fechas
# ===========================================================================

def fechas_de(df):
    """Convierte la columna 'fecha' ("dd-mm-aaaa", o "aaaa-mm-dd" en las pocas
    lecturas cargadas por la pagina) a fechas. Las ilegibles quedan NaT."""
    f = pd.to_datetime(df["fecha"], format="%d-%m-%Y", errors="coerce")
    otras = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce")
    return f.fillna(otras)


def fecha_final(fechas):
    """Fecha "actual" de los datos. Se usa el percentil 99,99 en lugar del
    maximo porque unas pocas lecturas cargadas por la pagina web tienen fecha
    de 2026 y correrian la ventana de "ultimo año"."""
    return fechas.dropna().quantile(0.9999)


# ===========================================================================
# Retrieval: fuentes de candidatos y variables
# ===========================================================================

def preparar_fuentes(datos):
    """Entrena todo lo necesario para generar candidatos y variables con 'datos'."""
    df = datos["df"]
    fechas = fechas_de(df)
    fin = fecha_final(fechas)
    recientes = df[fechas > fin - pd.Timedelta(days=365)]
    en_catalogo = set(datos["catalogo"])

    ease = modelos.entrenar_ease(datos)
    uknn = modelos.entrenar_user_knn(datos)

    libros_por_autor = {}
    for libro, autor in datos["autor_de_libro"].items():
        if isinstance(autor, str):
            libros_por_autor.setdefault(autor, []).append(libro)

    # proporcion de lecturas de cada lector en cada genero
    conteo_genero = df.groupby(["id_lector", "genero_libro"]).size()
    total_lector = df.groupby("id_lector").size()
    afinidad = (conteo_genero / total_lector.reindex(conteo_genero.index.get_level_values(0)).values).to_dict()

    return {
        "ease": ease,
        "uknn": uknn,
        "fila_ease": {u: f for f, u in enumerate(ease[2])},
        "columna_ease": {b: c for c, b in enumerate(ease[3])},
        "fila_uknn": {u: f for f, u in enumerate(uknn[2])},
        "columna_uknn": {b: c for c, b in enumerate(uknn[3])},
        "recientes": [b for b in recientes["id_libro"].value_counts().index if b in en_catalogo],
        "lecturas_libro": df["id_libro"].value_counts().to_dict(),
        "lecturas_recientes": recientes["id_libro"].value_counts().to_dict(),
        "rating_libro": df.groupby("id_libro")["rating"].mean().to_dict(),
        "lecturas_lector": total_lector.to_dict(),
        "libros_por_autor": libros_por_autor,
        "afinidad": afinidad,
    }


def mejores(modelo, fila_de, id_lector, leidos, k):
    """Los k libros con mayor puntaje para el lector (sin los ya leidos)."""
    puntajes, X, _, libros = modelo
    if id_lector not in fila_de:
        return []
    p = puntajes[fila_de[id_lector]].copy()
    p[X[fila_de[id_lector]] > 0] = -np.inf
    k = min(k, len(p) - 1)
    idx = np.argpartition(-p, k)[:k]
    return [libros[i] for i in idx if p[i] != -np.inf and libros[i] not in leidos]


def candidatos_con_variables(fuentes, datos, id_lector):
    """DataFrame con una fila por candidato y las columnas de VARIABLES."""
    leidos = datos["leidos"].get(id_lector, set())

    de_ease = mejores(fuentes["ease"], fuentes["fila_ease"], id_lector, leidos, config.K_CAND_EASE)
    de_uknn = mejores(fuentes["uknn"], fuentes["fila_uknn"], id_lector, leidos, config.K_CAND_USER_KNN)
    de_recientes = [b for b in fuentes["recientes"] if b not in leidos][:config.K_CAND_RECIENTES]

    autores_leidos = {}
    for b in leidos:
        autor = datos["autor_de_libro"].get(b)
        if isinstance(autor, str):
            autores_leidos[autor] = autores_leidos.get(autor, 0) + 1
    del_autor = {b for a in autores_leidos for b in fuentes["libros_por_autor"].get(a, []) if b not in leidos}
    de_autor = sorted(del_autor, key=lambda b: (-fuentes["lecturas_libro"].get(b, 0), b))[:config.K_CAND_AUTOR]

    candidatos = list(dict.fromkeys(de_ease + de_uknn + de_recientes + de_autor))
    if not candidatos:
        return pd.DataFrame(columns=["id_libro"] + VARIABLES)

    def puntajes_y_puestos(modelo, fila_de, columna_de):
        puntajes = np.full(len(candidatos), np.nan)
        if id_lector in fila_de:
            fila = modelo[0][fila_de[id_lector]]
            for i, b in enumerate(candidatos):
                if b in columna_de:
                    puntajes[i] = fila[columna_de[b]]
        puestos = pd.Series(puntajes).rank(ascending=False).values  # 1 = el mejor entre los candidatos
        return puntajes, puestos

    ease_p, ease_r = puntajes_y_puestos(fuentes["ease"], fuentes["fila_ease"], fuentes["columna_ease"])
    uknn_p, uknn_r = puntajes_y_puestos(fuentes["uknn"], fuentes["fila_uknn"], fuentes["columna_uknn"])
    s_ease, s_uknn, s_rec, s_aut = set(de_ease), set(de_uknn), set(de_recientes), set(de_autor)

    return pd.DataFrame({
        "id_libro": candidatos,
        "ease_puntaje": ease_p,
        "ease_puesto": ease_r,
        "uknn_puntaje": uknn_p,
        "uknn_puesto": uknn_r,
        "de_ease": [b in s_ease for b in candidatos],
        "de_uknn": [b in s_uknn for b in candidatos],
        "de_recientes": [b in s_rec for b in candidatos],
        "de_autor": [b in s_aut for b in candidatos],
        "log_lecturas_libro": [np.log1p(fuentes["lecturas_libro"].get(b, 0)) for b in candidatos],
        "log_lecturas_recientes": [np.log1p(fuentes["lecturas_recientes"].get(b, 0)) for b in candidatos],
        "rating_promedio_libro": [fuentes["rating_libro"].get(b, np.nan) for b in candidatos],
        "libros_leidos_del_autor": [autores_leidos.get(datos["autor_de_libro"].get(b), 0) for b in candidatos],
        "afinidad_genero": [fuentes["afinidad"].get((id_lector, datos["genero_de_libro"].get(b)), 0.0)
                            for b in candidatos],
        "log_lecturas_lector": np.log1p(fuentes["lecturas_lector"].get(id_lector, 0)),
    })


# ===========================================================================
# Ranking
# ===========================================================================

def entrenar_ranker(datos, corte_interno, rating_positivo=config.RATING_POSITIVO,
                    min_lecturas=config.MIN_QUE_QUEDAN, max_lectores=config.MAX_LECTORES_RANKER,
                    semilla=config.SEMILLA):
    """Entrena el modelo de ranking con un corte temporal dentro de 'datos'.
    Devuelve (modelo, cuantas filas de entrenamiento, cuantos positivos)."""
    df = datos["df"]
    fechas = fechas_de(df)
    corte = pd.Timestamp(corte_interno)
    pasado = df[~(fechas >= corte)]  # las fechas ilegibles cuentan como pasado
    futuro = df[(fechas >= corte) & (df["rating"] >= rating_positivo)]

    datos_pasado = modelos.construir_datos(pasado, datos["libros"], datos["lectores"])
    fuentes_pasado = preparar_fuentes(datos_pasado)

    con_pasado = {u for u, n in pasado.groupby("id_lector").size().items() if n >= min_lecturas}
    respuestas = futuro[futuro["id_lector"].isin(con_pasado)].groupby("id_lector")["id_libro"].apply(set).to_dict()
    lectores = sorted(respuestas)
    if max_lectores and len(lectores) > max_lectores:
        lectores = sorted(np.random.default_rng(semilla).choice(lectores, max_lectores, replace=False))

    tablas = []
    for id_lector in lectores:
        tabla = candidatos_con_variables(fuentes_pasado, datos_pasado, id_lector)
        tabla["leido"] = tabla["id_libro"].isin(respuestas[id_lector])
        tablas.append(tabla)
    entrenamiento = pd.concat(tablas, ignore_index=True)

    modelo = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            random_state=semilla)
    modelo.fit(entrenamiento[VARIABLES].astype(float), entrenamiento["leido"].astype(int))
    return modelo, len(entrenamiento), int(entrenamiento["leido"].sum())


def crear_dos_etapas(datos, n=config.N_RECOMENDACIONES, corte_interno=config.CORTE_RANKER_EVALUACION):
    """Estrategia en dos etapas, con la misma forma que las de modelos.py."""
    ranker, filas, positivos = entrenar_ranker(datos, corte_interno)
    print(f"    ranker entrenado con {filas} pares lector-libro ({positivos} leidos), corte interno {corte_interno}")
    fuentes = preparar_fuentes(datos)

    def estrategia(id_lector):
        tabla = candidatos_con_variables(fuentes, datos, id_lector)
        if tabla.empty:
            return modelos.completar_con_populares([], datos, id_lector, n)
        tabla["probabilidad"] = ranker.predict_proba(tabla[VARIABLES].astype(float))[:, 1]
        recomendados = tabla.sort_values(["probabilidad", "id_libro"], ascending=[False, True])["id_libro"]
        return modelos.completar_con_populares(recomendados.head(n).tolist(), datos, id_lector, n)

    return estrategia


def crear_populares_recientes(datos, n=config.N_RECOMENDACIONES):
    """Baseline: los libros mas leidos en el ultimo año."""
    fechas = fechas_de(datos["df"])
    recientes = datos["df"][fechas > fecha_final(fechas) - pd.Timedelta(days=365)]
    en_catalogo = set(datos["catalogo"])
    lista = [b for b in recientes["id_libro"].value_counts().index if b in en_catalogo]

    def estrategia(id_lector):
        leidos = datos["leidos"].get(id_lector, set())
        recomendados = [b for b in lista if b not in leidos][:n]
        return modelos.completar_con_populares(recomendados, datos, id_lector, n)

    return estrategia
