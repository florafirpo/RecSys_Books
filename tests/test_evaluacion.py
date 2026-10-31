"""Tests de las metricas (cuentas hechas a mano)."""

import math

import pytest

from evaluacion import intervalo_diferencia, metricas


def test_ndcg_perfecto_es_1():
    assert metricas(["a", "b"], {"a", "b"}, n=2)["NDCG@N"] == pytest.approx(1.0)


def test_ndcg_acierto_en_el_puesto_2():
    # DCG = 1/log2(3); IDCG (1 libro reservado) = 1/log2(2) = 1
    m = metricas(["x", "a", "y"], {"a"}, n=3)
    assert m["NDCG@N"] == pytest.approx(1 / math.log2(3))
    assert m["precision@N"] == pytest.approx(1 / 3)
    assert m["recall@N"] == 1.0
    assert m["hit rate@N"] == 1.0


def test_ndcg_premia_acertar_arriba():
    arriba = metricas(["a", "x", "y"], {"a"}, n=3)["NDCG@N"]
    abajo = metricas(["x", "y", "a"], {"a"}, n=3)["NDCG@N"]
    assert arriba > abajo


def test_sin_aciertos_todo_da_cero():
    m = metricas(["x", "y"], {"a"}, n=2)
    assert m == {"NDCG@N": 0.0, "precision@N": 0.0, "recall@N": 0.0, "hit rate@N": 0.0}


def test_intervalo_detecta_una_mejora_clara():
    buena = {u: {"NDCG@N": 1.0} for u in range(50)}
    mala = {u: {"NDCG@N": 0.0} for u in range(50)}
    dif, inf, sup = intervalo_diferencia(buena, mala, "NDCG@N")
    assert dif == 1.0 and inf > 0
