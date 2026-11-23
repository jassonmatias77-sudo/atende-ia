"""Métricas Prometheus.

POR QUE MÉTRICAS E NÃO LOGS
    Um log diz o que aconteceu numa requisição. Uma métrica diz o que está
    a acontecer em mil. "A latência subiu de 0,3s para 4s nas últimas 2 horas"
    é uma pergunta de métrica, não de log.

    Aqui as métricas são de NEGÓCIO, não só técnicas:
      - quantas perguntas foram escaladas para humano
      - qual é a confiança média das respostas
      - quanto tempo o agente leva

    É isso que permite responder "o assistente está a piorar?" com números.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

# --- tráfego ---------------------------------------------------------------
REQUISICOES = Counter(
    "atende_requisicoes_total", "Requisições HTTP", ["metodo", "rota", "estado"])

# --- negócio ---------------------------------------------------------------
PERGUNTAS = Counter(
    "atende_perguntas_total", "Perguntas respondidas", ["categoria", "escalada"])

CONFIANCA = Histogram(
    "atende_confianca", "Score de similaridade da resposta",
    buckets=(0.0, 0.3, 0.45, 0.55, 0.63, 0.7, 0.8, 0.9, 1.0))

LATENCIA = Histogram(
    "atende_latencia_s", "Tempo por etapa do agente",
    ["etapa"], buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0))

ESCALADAS = Counter(
    "atende_escaladas_total", "Conversas encaminhadas a humano", ["motivo"])

# --- estado ----------------------------------------------------------------
DOCUMENTOS = Gauge("atende_documentos", "Chunks indexados na base")
FALHAS_DEPENDENCIA = Counter(
    "atende_falhas_dependencia_total", "Falhas a falar com dependências",
    ["dependencia"])


@contextmanager
def medir_etapa(etapa: str):
    """Mede a duração de um bloco. Uso:

        with medir_etapa("rag"):
            docs = rag.buscar(...)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        LATENCIA.labels(etapa=etapa).observe(time.perf_counter() - t0)


def registar_atendimento(categoria: str, escalada: bool, confianca: float,
                         motivo: str = "") -> None:
    """Regista o resultado de um atendimento."""
    PERGUNTAS.labels(categoria=categoria or "desconhecida",
                     escalada=str(bool(escalada)).lower()).inc()
    if confianca > 0:
        CONFIANCA.observe(min(1.0, confianca))
    if escalada:
        ESCALADAS.labels(motivo=(motivo or "sem contexto")[:40]).inc()


def actualizar_documentos(n: int) -> None:
    DOCUMENTOS.set(n)


def registar_falha(dependencia: str) -> None:
    FALHAS_DEPENDENCIA.labels(dependencia=dependencia).inc()


def exportar() -> tuple[bytes, str]:
    """Payload para o endpoint /metrics."""
    return generate_latest(), CONTENT_TYPE_LATEST
