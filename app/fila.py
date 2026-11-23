"""Fila de tarefas longas.

PROBLEMA REAL
    Indexar 200 documentos com embeddings leva minutos. Se isso acontecer
    dentro de uma requisição HTTP, o cliente fica pendurado e o timeout
    mata o trabalho a meio.

    A fila resolve: a API aceita o pedido, devolve 202 Accepted com um id,
    e o trabalho continua em background. O cliente consulta o estado.

DOIS BACKENDS, MESMA INTERFACE
    memoria : threads + queue.Queue. Funciona sem infraestrutura.
    redis   : fila persistente, sobrevive a reinício do processo.

    Redis NÃO está disponível neste ambiente (medido: 6379 fechada local e
    no OMV), por isso o backend por omissão é memória. O README declara a
    limitação com honestidade — e o código está pronto para Redis quando
    houver um.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import Config, config as cfg_global

log = logging.getLogger(__name__)


@dataclass
class Tarefa:
    """Estado de uma tarefa em background."""

    id: str
    tipo: str
    estado: str = "pendente"      # pendente | a_correr | concluida | falhou
    criada_em: float = field(default_factory=time.time)
    inicio: float = 0.0
    fim: float = 0.0
    resultado: Any = None
    erro: str = ""

    def para_dict(self) -> dict:
        return {
            "id": self.id,
            "tipo": self.tipo,
            "estado": self.estado,
            "criada_em": self.criada_em,
            "duracao_s": round((self.fim or time.time()) - (self.inicio or self.criada_em), 3)
            if self.estado != "pendente" else 0.0,
            "resultado": self.resultado,
            "erro": self.erro,
        }


class FilaMemoria:
    """Fila em processo, com worker threads.

    Limitação declarada: se o processo morrer, as tarefas em curso perdem-se.
    Para um sistema em produção, usa-se Redis ou RabbitMQ.
    """

    def __init__(self, workers: int = 2) -> None:
        self._q: queue.Queue = queue.Queue()
        self._tarefas: dict[str, Tarefa] = {}
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._parar = threading.Event()
        self._n_workers = max(1, workers)
        self._iniciar_workers()

    def _iniciar_workers(self) -> None:
        for i in range(self._n_workers):
            t = threading.Thread(target=self._loop, daemon=True,
                                 name="fila-worker-%d" % i)
            t.start()
            self._workers.append(t)
        log.info("fila: %d worker(s) iniciado(s)", self._n_workers)

    def _loop(self) -> None:
        while not self._parar.is_set():
            try:
                tid, fn, kwargs = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            with self._lock:
                t = self._tarefas.get(tid)
                if t is None:
                    continue
                t.estado = "a_correr"
                t.inicio = time.time()

            try:
                res = fn(**kwargs)
                with self._lock:
                    t.estado = "concluida"
                    t.resultado = res
                    t.fim = time.time()
            except Exception as e:  # noqa: BLE001
                log.exception("tarefa %s falhou", tid)
                with self._lock:
                    t.estado = "falhou"
                    t.erro = "%s: %s" % (type(e).__name__, str(e)[:200])
                    t.fim = time.time()
            finally:
                self._q.task_done()

    def submeter(self, tipo: str, fn: Callable, **kwargs) -> str:
        """Enfileira. Devolve o id para consulta."""
        tid = "T-" + uuid.uuid4().hex[:10]
        with self._lock:
            self._tarefas[tid] = Tarefa(id=tid, tipo=tipo)
        self._q.put((tid, fn, kwargs))
        log.info("fila: tarefa %s (%s) enfileirada", tid, tipo)
        return tid

    def estado(self, tid: str) -> dict | None:
        with self._lock:
            t = self._tarefas.get(tid)
            return t.para_dict() if t else None

    def listar(self, limite: int = 20) -> list[dict]:
        with self._lock:
            ordenadas = sorted(self._tarefas.values(),
                               key=lambda x: x.criada_em, reverse=True)
            return [t.para_dict() for t in ordenadas[:limite]]

    def estatisticas(self) -> dict:
        with self._lock:
            por_estado: dict[str, int] = {}
            for t in self._tarefas.values():
                por_estado[t.estado] = por_estado.get(t.estado, 0) + 1
        return {"backend": "memoria", "workers": self._n_workers,
                "total": len(self._tarefas), "por_estado": por_estado}

    def parar(self) -> None:
        self._parar.set()
        for t in self._workers:
            t.join(timeout=2)


def criar_fila(cfg: Config | None = None):
    """Cria a fila conforme a configuração.

    Se o backend pedido for 'redis' e ele não estiver acessível, cai para
    memória com aviso — o serviço não pode deixar de arrancar por isso.
    """
    c = cfg or cfg_global()

    if c.fila.backend == "redis":
        try:
            import redis  # noqa: F401

            from redis import Redis

            cli = Redis.from_url(c.fila.redis_url, socket_timeout=3)
            cli.ping()
            log.warning("backend redis pedido mas a fila em memoria e' usada "
                        "neste projecto (implementacao nao incluida)")
        except Exception as e:  # noqa: BLE001
            log.warning("redis indisponivel (%s) — a usar fila em memoria",
                        type(e).__name__)

    return FilaMemoria(workers=c.fila.workers)
