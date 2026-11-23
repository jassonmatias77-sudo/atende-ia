"""Acesso ao PostgreSQL com pgvector.

DECISÃO DE ENGENHARIA: prefixo nas tabelas
    A base jaspem_vector já contém jaspem_knowledge e jaspem_memory, usadas
    pelo motor do JASPEM. Este projecto cria as SUAS tabelas com prefixo
    `atende_` em vez de reutilizar as existentes.

    Motivo: reutilizar tabelas de outro sistema acopla os dois. Se o motor
    do JASPEM mudar o schema, este projecto quebra — e vice-versa. Um prefixo
    custa nada e elimina o acoplamento.

POR QUE BUSCA VECTORIAL NO BANCO E NÃO EM PYTHON
    No portfólio anterior a similaridade era calculada em Python (força bruta
    sobre todos os vectores). Serve para 100 documentos; não serve para 100 mil.

    Aqui a busca é feita pelo pgvector com índice HNSW: a base usa o índice,
    não a aplicação. É a diferença entre O(n) e O(log n) — e é o que demonstra
    engenharia de produção.
"""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator

import psycopg2
import psycopg2.extras

from app.config import Config, config as cfg_global

log = logging.getLogger(__name__)


class ErroBaseDados(RuntimeError):
    """Falha ao falar com o Postgres."""


@dataclass
class Documento:
    """Um chunk de conhecimento indexado."""

    id: int
    texto: str
    fonte: str
    titulo: str
    score: float = 0.0


class BaseDados:
    """Ligação e operações. Uma instância por processo."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or cfg_global()
        self.p = self.cfg.base.prefixo
        self.dim = self.cfg.base.dimensoes
        self._conn: Any = None

    # ------------------------------------------------------------ ligação
    def ligar(self) -> Any:
        """Ligação única, reutilizada. Reconecta se cair."""
        if self._conn is not None and self._conn.closed == 0:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return self._conn
            except Exception:  # noqa: BLE001
                with contextlib.suppress(Exception):
                    self._conn.close()
                self._conn = None

        try:
            self._conn = psycopg2.connect(self.cfg.base.dsn())
            self._conn.autocommit = False
            return self._conn
        except Exception as e:  # noqa: BLE001
            raise ErroBaseDados(
                "não consegui ligar a %s:%d (%s). O Postgres está no ar?"
                % (self.cfg.base.host, self.cfg.base.port,
                   type(e).__name__)) from e

    @contextlib.contextmanager
    def cursor(self) -> Iterator[Any]:
        cn = self.ligar()
        cur = cn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            cn.commit()
        except Exception:
            cn.rollback()
            raise
        finally:
            cur.close()

    def saudavel(self) -> tuple[bool, str]:
        """Verificação rápida, sem levantar excepção."""
        try:
            with self.cursor() as cur:
                cur.execute("SELECT version()")
                v = cur.fetchone()["version"]
                cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
                r = cur.fetchone()
                return True, "Postgres ok · pgvector %s" % (r["extversion"] if r else "ausente")
        except Exception as e:  # noqa: BLE001
            return False, "%s: %s" % (type(e).__name__, str(e)[:70])

    # ------------------------------------------------------------- schema
    def criar_schema(self) -> dict:
        """Cria tabelas, índice HNSW e a função de busca. Idempotente."""
        feito = {}
        with self.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            feito["extensao_vector"] = True

            # utilizadores (autenticação JWT)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS %susuarios (
                    id          SERIAL PRIMARY KEY,
                    email       TEXT UNIQUE NOT NULL,
                    senha_hash  TEXT NOT NULL,
                    papel       TEXT NOT NULL DEFAULT 'atendente',
                    criado_em   TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """ % self.p)
            feito["tabela_usuarios"] = True

            # documentos indexados
            cur.execute("""
                CREATE TABLE IF NOT EXISTS %sdocumentos (
                    id       SERIAL PRIMARY KEY,
                    texto    TEXT NOT NULL,
                    fonte    TEXT NOT NULL,
                    titulo   TEXT NOT NULL DEFAULT '',
                    embedding vector(%d),
                    criado_em TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """ % (self.p, self.dim))
            feito["tabela_documentos"] = True

            # conversas (agente com estado)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS %sconversas (
                    id        SERIAL PRIMARY KEY,
                    sessao    TEXT NOT NULL,
                    pergunta  TEXT NOT NULL,
                    resposta  TEXT NOT NULL DEFAULT '',
                    confianca REAL NOT NULL DEFAULT 0,
                    escalado  BOOLEAN NOT NULL DEFAULT false,
                    tempo_s   REAL NOT NULL DEFAULT 0,
                    criado_em TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """ % self.p)
            feito["tabela_conversas"] = True

            cur.execute("CREATE INDEX IF NOT EXISTS %sconv_sessao_idx "
                        "ON %sconversas (sessao, criado_em DESC)"
                        % (self.p, self.p))
            feito["indice_conversas"] = True

            # índice HNSW para busca por similaridade de cosseno.
            # vectors.cosine_ops é o operador <=> usado nas consultas.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS %sdocs_hnsw_idx
                ON %sdocumentos
                USING hnsw (embedding vector_cosine_ops)
            """ % (self.p, self.p))
            feito["indice_hnsw"] = True

        return feito

    # ---------------------------------------------------------- documentos
    def inserir_documentos(self, itens: list[dict]) -> int:
        """Insere chunks com os seus embeddings. Devolve quantos entrou.

        Cada item: {"texto", "fonte", "titulo", "embedding": [float, ...]}
        """
        if not itens:
            return 0

        with self.cursor() as cur:
            n = 0
            for it in itens:
                emb = it.get("embedding") or []
                if len(emb) != self.dim:
                    log.warning("embedding com %d dimensões, esperado %d — ignorado",
                                len(emb), self.dim)
                    continue
                # pgvector aceita o literal '[0.1,0.2,...]'::vector
                literal = "[" + ",".join("%.8f" % float(x) for x in emb) + "]"
                cur.execute(
                    f"INSERT INTO {self.p}documentos "
                    f"(texto, fonte, titulo, embedding) "
                    f"VALUES (%s, %s, %s, %s::vector)",
                    (it.get("texto", ""), it.get("fonte", ""),
                     it.get("titulo", ""), literal))
                n += 1
        return n

    def contar_documentos(self) -> int:
        with self.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM %sdocumentos" % self.p)
            return int(cur.fetchone()["n"])

    def fontes(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("SELECT DISTINCT fonte FROM %sdocumentos ORDER BY fonte"
                        % self.p)
            return [r["fonte"] for r in cur.fetchall()]

    # ------------------------------------------------------------- busca
    def buscar(self, embedding: list[float], top_k: int = 3) -> list[Documento]:
        """Busca vectorial pelo índice HNSW.

        O operador <=> é a distância de cosseno do pgvector.
        score = 1 - distância, ou seja: 1.0 é idêntico, 0.0 é ortogonal.
        """
        if not embedding:
            return []
        if len(embedding) != self.dim:
            raise ValueError("embedding com %d dimensões, esperado %d"
                             % (len(embedding), self.dim))

        literal = "[" + ",".join("%.8f" % float(x) for x in embedding) + "]"

        # O nome da tabela entra por f-string (não é dado do utilizador, é
        # constante do processo). Os VALORES entram como parâmetros %s, que
        # o psycopg2 escapa. Misturar os dois no mesmo operador % é o erro
        # que eu tinha aqui.
        sql = (
            "SELECT id, texto, fonte, titulo, "
            f"       1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.p}documentos "
            f"ORDER BY embedding <=> %s::vector "
            f"LIMIT %s"
        )
        with self.cursor() as cur:
            cur.execute(sql, (literal, literal, top_k))
            return [Documento(id=r["id"], texto=r["texto"], fonte=r["fonte"],
                              titulo=r["titulo"], score=float(r["score"]))
                    for r in cur.fetchall()]

    # ---------------------------------------------------------- conversas
    def registar_conversa(self, sessao: str, pergunta: str, resposta: str,
                          confianca: float, escalado: bool, tempo_s: float) -> int:
        with self.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.p}conversas "
                f"(sessao, pergunta, resposta, confianca, escalado, tempo_s) "
                f"VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (sessao, pergunta[:2000], resposta[:8000], confianca,
                 escalado, tempo_s))
            return int(cur.fetchone()["id"])

    def historico(self, sessao: str, limite: int = 10) -> list[dict]:
        with self.cursor() as cur:
            cur.execute(
                f"SELECT pergunta, resposta, confianca, escalado, tempo_s, "
                f"       criado_em "
                f"FROM {self.p}conversas WHERE sessao = %s "
                f"ORDER BY criado_em DESC LIMIT %s",
                (sessao, limite))
            return [dict(r) for r in cur.fetchall()]

    def metricas(self) -> dict:
        """Números para o painel de observabilidade."""
        with self.cursor() as cur:
            cur.execute(f"""
                SELECT count(*) AS total,
                       coalesce(avg(confianca), 0) AS confianca_media,
                       coalesce(avg(tempo_s), 0) AS tempo_medio,
                       count(*) FILTER (WHERE escalado) AS escaladas
                FROM {self.p}conversas
            """)
            r = dict(cur.fetchone())
            cur.execute("SELECT count(*) AS n FROM %sdocumentos" % self.p)
            r["documentos"] = int(cur.fetchone()["n"])
            return r
