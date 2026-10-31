"""
Punto de entrada del recomendador de libros (competencia de Kaggle).

    python main.py split                  separa data.db en train y test (80/20 por lector, al azar)
    python main.py split --temporal 2024-01-01
                                          corte por fecha: train = antes, test = desde esa fecha
    python main.py split --ultimas        a cada lector le esconde su ultimo 20% (por fecha)
    python main.py evaluar                compara todas las estrategias sobre test
    python main.py kaggle ease            arma el CSV para Kaggle con la estrategia elegida
    python main.py kaggle --todas         arma un CSV por cada estrategia

Estrategias para 'kaggle': azar, popularidad, genero, knn_regresion,
knn_regresion_escalado, user_knn, ease.
Las rutas y los parametros se cambian en config.py.
"""

import argparse
import os

import pandas as pd

import config

ESTRATEGIAS_KAGGLE = ["azar", "popularidad", "genero", "knn_regresion",
                      "knn_regresion_escalado", "user_knn", "ease"]


def crear_estrategia(nombre, datos, n):
    import modelos
    if nombre == "knn_regresion_escalado":
        return modelos.crear_knn_regresion(datos, n, escalar=True)
    return getattr(modelos, f"crear_{nombre}")(datos, n)


def comando_split(args):
    import dataset
    if args.temporal:
        dataset.separar_train_test_temporal(args.temporal)
    elif args.ultimas:
        dataset.separar_train_test_ultimas()
    else:
        dataset.separar_train_test()


def comando_evaluar(args):
    import evaluacion
    evaluacion.evaluar_todo()


def comando_kaggle(args):
    """Entrena con TODA la base (data.db) y recomienda N libros a cada lector de ejemplo.csv."""
    import modelos

    if not os.path.exists(config.CSV_LECTORES_KAGGLE):
        print(f"No encuentro {config.CSV_LECTORES_KAGGLE}: copiá ahí el CSV de ejemplo de la competencia.")
        return
    lectores = pd.read_csv(config.CSV_LECTORES_KAGGLE, dtype=str)["id_lector"].unique()
    nombres = ESTRATEGIAS_KAGGLE if args.todas else [args.estrategia]
    n = config.N_RECOMENDACIONES

    datos = modelos.cargar_datos(config.DB_COMPLETA)
    os.makedirs(config.CARPETA_SUBMISSIONS, exist_ok=True)
    for nombre in nombres:
        estrategia = crear_estrategia(nombre, datos, n)
        filas = [(id_lector, libro) for id_lector in lectores for libro in estrategia(id_lector)[:n]]
        ruta = os.path.join(config.CARPETA_SUBMISSIONS, f"submission_{nombre}.csv")
        pd.DataFrame(filas, columns=["id_lector", "id_libro"]).to_csv(ruta, index=False)
        print(f"{nombre}: {len(filas)} filas -> {ruta}")


def crear_parser():
    parser = argparse.ArgumentParser(description="Recomendador de libros (Kaggle)")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("split", help="separa data.db en train y test")
    p.add_argument("--temporal", metavar="AAAA-MM-DD", default=None,
                   help="corte por fecha en lugar de al azar")
    p.add_argument("--ultimas", action="store_true",
                   help="a cada lector le esconde su ultimo 20%% de lecturas (por fecha)")
    p.set_defaults(funcion=comando_split)

    p = sub.add_parser("evaluar", help="compara las estrategias sobre test")
    p.set_defaults(funcion=comando_evaluar)

    p = sub.add_parser("kaggle", help="arma el CSV de submission")
    p.add_argument("estrategia", nargs="?", default="ease", choices=ESTRATEGIAS_KAGGLE)
    p.add_argument("--todas", action="store_true", help="un CSV por cada estrategia")
    p.set_defaults(funcion=comando_kaggle)
    return parser


def main():
    args = crear_parser().parse_args()
    args.funcion(args)


if __name__ == "__main__":
    main()
