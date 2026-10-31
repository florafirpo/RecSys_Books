"""Tests del recomendador en dos etapas, con datos de juguete fechados."""

import pandas as pd
import pytest

import dos_etapas
import modelos


@pytest.fixture
def datos_fechados(base, monkeypatch):
    """Lecturas en 2021-2022 (pasado) y algunas en 2023 (lo que el ranking aprende a predecir)."""
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    datos = modelos.cargar_datos(base)
    df = datos["df"].copy()
    df["fecha"] = "01-06-2021"
    df.loc[df.index[::4], "fecha"] = "15-03-2023"
    return modelos.construir_datos(df, datos["libros"], datos["lectores"])


def test_fechas_entiende_los_dos_formatos_y_las_ilegibles():
    df = pd.DataFrame({"fecha": ["03-02-2024", "2026-09-19", "muy poco creibles"]})
    f = dos_etapas.fechas_de(df)
    assert f.iloc[0].year == 2024 and f.iloc[1].year == 2026 and pd.isna(f.iloc[2])


def test_fecha_final_ignora_fechas_sueltas_muy_nuevas():
    fechas = pd.Series(pd.to_datetime(["2024-12-31"] * 100000 + ["2026-09-19"]))
    assert dos_etapas.fecha_final(fechas).year == 2024


def test_candidatos_no_incluyen_libros_leidos(datos_fechados):
    fuentes = dos_etapas.preparar_fuentes(datos_fechados)
    for lector, leidos in datos_fechados["leidos"].items():
        tabla = dos_etapas.candidatos_con_variables(fuentes, datos_fechados, lector)
        assert not set(tabla["id_libro"]) & leidos
        assert list(tabla.columns) == ["id_libro"] + dos_etapas.VARIABLES


def test_ranker_solo_aprende_con_lo_anterior_al_corte(datos_fechados, monkeypatch):
    # si el ranking viera el futuro, los libros de 2023 aparecerian como "leidos" en sus candidatos
    monkeypatch.setattr(dos_etapas.config, "MIN_QUE_QUEDAN", 1)
    capturado = {}
    original = dos_etapas.preparar_fuentes

    def espia(datos):
        capturado["fechas"] = dos_etapas.fechas_de(datos["df"])
        return original(datos)

    monkeypatch.setattr(dos_etapas, "preparar_fuentes", espia)
    try:
        dos_etapas.entrenar_ranker(datos_fechados, "2023-01-01", min_lecturas=1)
    except ValueError:
        pass  # con tan pocos datos el modelo puede no tener las dos clases; lo que importa es el corte
    assert (capturado["fechas"] < pd.Timestamp("2023-01-01")).all()
