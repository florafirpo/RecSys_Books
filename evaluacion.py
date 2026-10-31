"""
Evaluacion offline contra las reseñas reservadas en data_test.db.

Metricas por lector (despues se promedian), con N = 20 como en Kaggle:
  - NDCG@N      : la metrica de la competencia. Premia acertar y acertar arriba
                  (un acierto en el puesto 1 vale mas que en el puesto 20).
  - precision@N : de los N recomendados, que fraccion estaba en test.
  - recall@N    : de los libros de test, que fraccion aparecio entre los N.
  - hit rate@N  : % de lectores con al menos un acierto.

Ademas compara cada estrategia contra la popularidad con un intervalo de
confianza del 95% (bootstrap pareado), para saber si la diferencia es real.
"""

import math
import os
import random
import sqlite3
import time
from collections import defaultdict

import config
import modelos


def cargar_test(db_test=config.DB_TEST):
    """{id_lector: set(id_libro reservados)}."""
    con = sqlite3.connect(db_test)
    reservados = defaultdict(set)
    for id_lector, id_libro in con.execute("SELECT id_lector, id_libro FROM interacciones"):
        reservados[id_lector].add(id_libro)
    con.close()
    return reservados


def elegir_lectores(reservados, cantidad=config.USUARIOS_A_EVALUAR, semilla=config.SEMILLA):
    """Muestra fija de lectores de test (siempre la misma para todas las estrategias)."""
    lectores = sorted(reservados)
    if cantidad and cantidad < len(lectores):
        lectores = random.Random(semilla).sample(lectores, cantidad)
    return {u: reservados[u] for u in lectores}


def metricas(recomendados, reservados, n):
    """Metricas de UN lector."""
    aciertos = [i for i, libro in enumerate(recomendados[:n]) if libro in reservados]
    dcg = sum(1 / math.log2(i + 2) for i in aciertos)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n, len(reservados))))
    return {
        "NDCG@N": dcg / idcg,
        "precision@N": len(aciertos) / n,
        "recall@N": len(aciertos) / len(reservados),
        "hit rate@N": 1.0 if aciertos else 0.0,
    }


def evaluar(estrategia, reservados_por_lector, n):
    """Devuelve (promedios, valores por lector)."""
    por_lector = {}
    for id_lector, reservados in reservados_por_lector.items():
        por_lector[id_lector] = metricas(estrategia(id_lector), reservados, n)
    nombres = next(iter(por_lector.values())).keys()
    promedios = {m: sum(v[m] for v in por_lector.values()) / len(por_lector) for m in nombres}
    return promedios, por_lector


def intervalo_diferencia(por_lector_a, por_lector_b, metrica, repeticiones=2000, semilla=0):
    """Diferencia promedio (a - b) e intervalo de confianza del 95% por
    bootstrap pareado: se re-sortean los lectores con reposicion."""
    diferencias = [por_lector_a[u][metrica] - por_lector_b[u][metrica] for u in por_lector_a]
    azar = random.Random(semilla)
    promedios = sorted(
        sum(azar.choice(diferencias) for _ in diferencias) / len(diferencias)
        for _ in range(repeticiones)
    )
    return (sum(diferencias) / len(diferencias),
            promedios[int(0.025 * repeticiones)],
            promedios[int(0.975 * repeticiones) - 1])


def estrategias_a_evaluar(datos, n):
    """Nombre -> funcion que crea la estrategia. Aca se suman modelos nuevos."""
    return {
        "azar": lambda: modelos.crear_azar(datos, n),
        "popularidad": lambda: modelos.crear_popularidad(datos, n),
        "genero (notebook)": lambda: modelos.crear_genero(datos, n),
        "knn_regresion (notebook)": lambda: modelos.crear_knn_regresion(datos, n),
        "knn_regresion escalado": lambda: modelos.crear_knn_regresion(datos, n, escalar=True),
        f"user-kNN (k={config.K_VECINOS})": lambda: modelos.crear_user_knn(datos, n),
        f"EASE (lambda={config.LAMBDA_EASE})": lambda: modelos.crear_ease(datos, n),
    }


def evaluar_todo(db_train=config.DB_TRAIN, db_test=config.DB_TEST, n=config.N_RECOMENDACIONES):
    if not (os.path.exists(db_train) and os.path.exists(db_test)):
        print("Faltan las bases de train/test: corré primero  python main.py split")
        return None

    datos = modelos.cargar_datos(db_train)
    reservados = elegir_lectores(cargar_test(db_test))
    print(f"Evaluando {len(reservados)} lectores, N = {n}\n")

    resultados, por_lector = {}, {}
    for nombre, crear in estrategias_a_evaluar(datos, n).items():
        inicio = time.time()
        estrategia = crear()
        resultados[nombre], por_lector[nombre] = evaluar(estrategia, reservados, n)
        print(f"  {nombre:28s} listo ({time.time() - inicio:.0f} s)")

    columnas = list(next(iter(resultados.values())).keys())
    ancho = max(len(nombre) for nombre in resultados) + 2
    print("\n" + "estrategia".ljust(ancho) + "".join(c.rjust(13) for c in columnas))
    for nombre, valores in resultados.items():
        print(nombre.ljust(ancho) + "".join(f"{valores[c]:13.4f}" for c in columnas))

    print("\nDiferencia de NDCG@N contra popularidad (IC 95%, bootstrap pareado):")
    for nombre in resultados:
        if nombre == "popularidad":
            continue
        dif, inf, sup = intervalo_diferencia(por_lector[nombre], por_lector["popularidad"], "NDCG@N")
        veredicto = "significativa" if inf > 0 or sup < 0 else "no significativa"
        print(f"  {nombre:28s} {dif:+.4f}  [{inf:+.4f}, {sup:+.4f}]  {veredicto}")
    return resultados
