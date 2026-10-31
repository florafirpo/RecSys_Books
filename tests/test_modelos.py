"""Tests de las estrategias sobre la base de juguete."""

import pytest

import modelos


@pytest.fixture
def datos(base):
    return modelos.cargar_datos(base)


ESTRATEGIAS = ["crear_azar", "crear_popularidad", "crear_genero", "crear_user_knn", "crear_ease"]


@pytest.mark.parametrize("nombre", ESTRATEGIAS)
def test_ninguna_estrategia_recomienda_libros_ya_leidos(datos, nombre, monkeypatch):
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    estrategia = getattr(modelos, nombre)(datos, n=2)
    for lector, leidos in datos["leidos"].items():
        assert not set(estrategia(lector)) & leidos


@pytest.mark.parametrize("nombre", ESTRATEGIAS)
def test_las_estrategias_no_repiten_libros(datos, nombre, monkeypatch):
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    recomendados = getattr(modelos, nombre)(datos, n=3)("caro")
    assert len(recomendados) == len(set(recomendados))


def test_knn_regresion_no_recomienda_leidos(datos, monkeypatch):
    monkeypatch.setattr(modelos.config, "K_REGRESION", 3)
    monkeypatch.setattr(modelos.config, "UMBRAL_GENERO_FRECUENTE", 1)
    estrategia = modelos.crear_knn_regresion(datos, n=2)
    assert not set(estrategia("caro")) & datos["leidos"]["caro"]


def test_popularidad_ordena_por_cantidad_de_lecturas(datos):
    # caro no leyo L2 (3 lecturas), L5 ni L6 (1 lectura cada uno): primero L2
    assert modelos.crear_popularidad(datos, n=1)("caro") == ["L2"]


def test_lector_desconocido_recibe_popularidad(datos, monkeypatch):
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    ease = modelos.crear_ease(datos, n=3)
    assert ease("nuevo") == modelos.crear_popularidad(datos, n=3)("nuevo")


def test_user_knn_recomienda_lo_que_leyo_el_lector_mas_parecido(datos, monkeypatch):
    # beto y caro comparten L1, L3 y L4; beto ademas leyo L2 -> a caro le toca L2
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    assert modelos.crear_user_knn(datos, n=1, k=1)("caro") == ["L2"]


def test_la_matriz_respeta_el_minimo_de_lecturas(datos):
    X, lectores, libros = modelos.armar_matriz(datos, min_lecturas=3)
    # con 3+ lecturas solo quedan L1 (3), L2 (3), L3 (2 -> no), L4 (4)
    assert libros == ["L1", "L2", "L4"]
    assert X.shape == (4, 3) and X.sum() == 10


def test_hibrido_elige_la_estrategia_segun_lecturas(datos, monkeypatch):
    # en la base de juguete: dani leyo 2 libros, caro 3, ana 5
    monkeypatch.setattr(modelos.config, "MIN_LECTURAS_LIBRO", 1)
    hibrido = modelos.crear_hibrido(datos, n=2, min_para_user_knn=3, min_para_ease=5)
    assert hibrido("dani") == modelos.crear_popularidad(datos, n=2)("dani")
    assert hibrido("caro") == modelos.crear_user_knn(datos, n=2)("caro")
    assert hibrido("ana") == modelos.crear_ease(datos, n=2)("ana")
