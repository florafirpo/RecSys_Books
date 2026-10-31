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


# ===========================================================================
# Rendimiento segun cuanto lee cada lector
# ===========================================================================

# Grupos de lectores segun cuantas lecturas tienen en train
GRUPOS_ACTIVIDAD = [(3, 9), (10, 24), (25, 49), (50, 99), (100, 249), (250, None)]

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # orden fijo de la paleta categorica
MARCADORES = ["o", "s", "^", "D"]


def nombre_grupo(desde, hasta):
    return f"{desde}+" if hasta is None else f"{desde}-{hasta}"


def intervalo_promedio(valores, repeticiones=2000, semilla=0):
    """Promedio e intervalo de confianza del 95% (bootstrap) de una lista."""
    azar = random.Random(semilla)
    promedios = sorted(sum(azar.choice(valores) for _ in valores) / len(valores) for _ in range(repeticiones))
    return sum(valores) / len(valores), promedios[int(0.025 * repeticiones)], promedios[int(0.975 * repeticiones) - 1]


def evaluar_por_actividad(db_train=config.DB_TRAIN, db_test=config.DB_TEST, n=config.N_RECOMENDACIONES,
                          carpeta_salida=os.path.join(config.CARPETA_PROYECTO, "resultados")):
    """NDCG@N de cada estrategia segun la cantidad de libros que cada lector
    leyo en train. Evalua a TODOS los lectores de test (no a una muestra),
    imprime la tabla y guarda un grafico y un CSV en 'resultados/'."""
    if not (os.path.exists(db_train) and os.path.exists(db_test)):
        print("Faltan las bases de train/test: corré primero  python main.py split ...")
        return None

    datos = modelos.cargar_datos(db_train)
    reservados = cargar_test(db_test)
    lecturas = {u: len(libros) for u, libros in datos["leidos"].items()}
    print(f"Evaluando {len(reservados)} lectores, N = {n}\n")

    creadores = {
        "popularidad": lambda: modelos.crear_popularidad(datos, n),
        "knn_regresion (notebook)": lambda: modelos.crear_knn_regresion(datos, n),
        "user-kNN": lambda: modelos.crear_user_knn(datos, n),
        "EASE": lambda: modelos.crear_ease(datos, n),
    }
    por_lector = {}
    for nombre, crear in creadores.items():
        inicio = time.time()
        _, por_lector[nombre] = evaluar(crear(), reservados, n)
        print(f"  {nombre:26s} listo ({time.time() - inicio:.0f} s)")

    # Armamos la tabla: una fila por grupo y estrategia
    filas = []
    for desde, hasta in GRUPOS_ACTIVIDAD:
        lectores = [u for u in reservados
                    if lecturas.get(u, 0) >= desde and (hasta is None or lecturas.get(u, 0) <= hasta)]
        if not lectores:
            continue
        for nombre in creadores:
            prom, inf, sup = intervalo_promedio([por_lector[nombre][u]["NDCG@N"] for u in lectores])
            filas.append({"grupo": nombre_grupo(desde, hasta), "lectores": len(lectores), "estrategia": nombre,
                          "NDCG": prom, "IC_inf": inf, "IC_sup": sup})

    import pandas as pd
    tabla = pd.DataFrame(filas)
    ancha = tabla.pivot(index="grupo", columns="estrategia", values="NDCG")
    ancha = ancha.reindex([g for g in dict.fromkeys(tabla["grupo"])])[list(creadores)]
    ancha.insert(0, "lectores", tabla.groupby("grupo")["lectores"].first())
    print("\nNDCG@N segun lecturas en train:")
    print(ancha.round(4).to_string())

    os.makedirs(carpeta_salida, exist_ok=True)
    tabla.to_csv(os.path.join(carpeta_salida, "ndcg_por_actividad.csv"), index=False)

    ruta = os.path.join(carpeta_salida, "ndcg_por_actividad.png")
    graficar_por_actividad(tabla, n, ruta)
    print(f"\nGrafico guardado en {ruta}")
    return tabla


def graficar_por_actividad(tabla, n, ruta):
    """Grafico de lineas: NDCG por grupo de actividad, una linea por estrategia,
    con la banda del intervalo de confianza. 'tabla' es la que arma
    evaluar_por_actividad (tambien se puede leer de ndcg_por_actividad.csv)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    creadores = list(dict.fromkeys(tabla["estrategia"]))
    lectores_por_grupo = tabla.groupby("grupo", sort=False)["lectores"].first()
    grupos = list(lectores_por_grupo.index)
    etiquetas = [f"{g}\n({lectores_por_grupo[g]} lectores)" for g in grupos]
    x = list(range(len(grupos)))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    finales = {}
    for k, nombre in enumerate(creadores):
        sub = tabla[tabla["estrategia"] == nombre].set_index("grupo").loc[grupos]
        ax.fill_between(x, sub["IC_inf"], sub["IC_sup"], color=COLORES[k], alpha=0.12, linewidth=0)
        ax.plot(x, sub["NDCG"], color=COLORES[k], linewidth=2, marker=MARCADORES[k], markersize=8,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=nombre)
        finales[nombre] = sub["NDCG"].iloc[-1]

    # Etiqueta directa al final de cada linea (para no depender solo del color),
    # separadas verticalmente si quedan muy juntas
    separacion = 0.045 * max(tabla["IC_sup"].max(), 1e-9)
    ultima_y = None
    for nombre, y in sorted(finales.items(), key=lambda par: par[1]):
        y_texto = y if ultima_y is None else max(y, ultima_y + separacion)
        ax.annotate(nombre, (x[-1], y), xytext=(x[-1] + 0.12, y_texto), textcoords="data",
                    va="center", fontsize=9, color="#52514e")
        ultima_y = y_texto
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_xlabel("Libros leidos por el lector (en train)", color="#52514e")
    ax.set_ylabel(f"NDCG@{n}", color="#52514e")
    ax.set_xlim(-0.3, len(x) - 1 + 1.6)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#e6e5e1")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.07, 0.95, f"NDCG@{n} de cada estrategia segun cuanto lee el lector", fontsize=13,
             fontweight="bold", color="#0b0b0b", ha="left")
    fig.text(0.07, 0.905, "Banda: intervalo de confianza del 95% (bootstrap). Grupos chicos = bandas anchas.",
             fontsize=9.5, color="#52514e", ha="left")
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.09, right=0.97)
    fig.savefig(ruta, dpi=160)
    plt.close(fig)
