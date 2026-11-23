"""Fixtures dos testes.

PRINCIPIO: os testes NAO dependem de servicos externos
    Um teste que exige Postgres, Ollama e Redis a correr falha na maquina de
    quem avalia o projecto. Isso e' um teste fragil, nao um teste rigoroso.

    Aqui a base e o LLM sao substituidos por duplos deterministas. O que
    fica sem teste automatizado — a integracao real com pgvector e Ollama —
    foi validado por scripts de integracao contra os servicos reais, e o
    README declara essa fronteira.

NOTA SOBRE O EMBEDDING FALSO
    Usa saco-de-palavras: dois textos com vocabulario comum ficam proximos.
    Nao e' semantico como o nomic-embed-text, mas reproduz o COMPORTAMENTO
    necessário para testar o fluxo (relevante perto, alheio longe).

    Consequencia: o valor exacto do score nao tem significado nestes testes.
    O limiar usado nos testes de integracao e' 0.0 (documento presente ->
    responde). O limiar de producao (0.63) foi calibrado por medicao contra
    o pgvector real, noutro script.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


@pytest.fixture(autouse=True)
def _ambiente(monkeypatch):
    monkeypatch.setenv("ATENDE_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("PG_PASSWORD", "senha_de_teste")
    monkeypatch.delenv("ATENDE_LIMIAR_ESCALONAMENTO", raising=False)

    import app.config as ac

    ac.config.cache_clear()
    yield
    ac.config.cache_clear()


@pytest.fixture
def cfg_teste():
    """Config com limiar 0.0 — para os testes de integracao com o duplo.

    Com o embedding de saco-de-palavras, o score exacto nao e' significativo.
    Usar 0.0 isola o que se quer testar: o FLUXO do agente.
    O limiar real (0.63) e' testado em test_config e validado contra o
    pgvector em script de integracao.
    """
    from app.config import Config

    return Config(limiar_escalonamento=0.0)


class BDduplo:
    """Duplo da base de dados, em memoria."""

    def __init__(self, prefixo: str = "atende_", dim: int = 768) -> None:
        self.prefixo = prefixo
        self.dim = dim
        self.documentos: list[dict] = []
        self.conversas: list[dict] = []
        self._id = 0

    def saudavel(self):
        return True, "duplo em memoria"

    def criar_schema(self):
        return {"duplo": True}

    def cursor(self):  # alguns testes constroem ServicoAuth
        raise NotImplementedError("o duplo nao implementa cursor")

    def inserir_documentos(self, itens):
        n = 0
        for it in itens:
            if len(it.get("embedding") or []) != self.dim:
                continue
            self._id += 1
            self.documentos.append({**it, "id": self._id})
            n += 1
        return n

    def contar_documentos(self) -> int:
        return len(self.documentos)

    def fontes(self) -> list[str]:
        return sorted({d["fonte"] for d in self.documentos})

    def buscar(self, embedding, top_k: int = 3):
        """Similaridade de cosseno real sobre os documentos em memoria."""
        from app.database import Documento

        def cos(a, b):
            if not a or not b or len(a) != len(b):
                return 0.0
            p = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return p / (na * nb) if na and nb else 0.0

        pontuados = [
            Documento(id=d["id"], texto=d["texto"], fonte=d["fonte"],
                      titulo=d.get("titulo", ""),
                      score=cos(embedding, d["embedding"]))
            for d in self.documentos
        ]
        pontuados.sort(key=lambda x: x.score, reverse=True)
        return pontuados[:top_k]

    def registar_conversa(self, sessao, pergunta, resposta, confianca,
                          escalado, tempo_s):
        self._id += 1
        self.conversas.append({
            "id": self._id, "sessao": sessao, "pergunta": pergunta,
            "resposta": resposta, "confianca": confianca,
            "escalado": escalado, "tempo_s": tempo_s})
        return self._id

    def historico(self, sessao, limite=10):
        return [c for c in self.conversas if c["sessao"] == sessao][-limite:]

    def metricas(self):
        confs = [c["confianca"] for c in self.conversas]
        tempos = [c["tempo_s"] for c in self.conversas]
        return {
            "total": len(self.conversas),
            "confianca_media": sum(confs) / len(confs) if confs else 0.0,
            "tempo_medio": sum(tempos) / len(tempos) if tempos else 0.0,
            "escaladas": sum(1 for c in self.conversas if c["escalado"]),
            "documentos": len(self.documentos),
        }


class LLMduplo:
    """Duplo do Ollama.

    Embedding por saco-de-palavras: palavras comuns aproximam os vectores.
    """

    def __init__(self, dim: int = 768, falhar: bool = False,
                 texto: str = "resposta do modelo") -> None:
        self.dim = dim
        self.falhar = falhar
        self._texto = texto
        self.chamadas = 0
        self.prompts: list[str] = []

    def disponivel(self, timeout: float = 8.0) -> bool:
        return not self.falhar

    def modelos(self):
        if self.falhar:
            from app.llm import OllamaIndisponivel

            raise OllamaIndisponivel("duplo em falha")
        return ["qwen2.5-coder:7b", "nomic-embed-text:latest"]

    def diagnostico(self):
        if self.falhar:
            return {"ollama": False, "faltando": ["Ollama (duplo)"]}
        return {"ollama": True, "faltando": []}

    def embeddings(self, texto: str):
        """Saco-de-palavras: cada palavra contribui para uma posicao fixa."""
        if self.falhar:
            from app.llm import OllamaIndisponivel

            raise OllamaIndisponivel("duplo em falha")

        v = [0.0] * self.dim
        palavras = re.findall(r"[0-9a-zA-Z\u00c0-\u017f]+", (texto or "").lower())
        for p in palavras:
            # hash estavel da palavra
            h = 0
            for ch in p:
                h = (h * 31 + ord(ch)) % 1000003
            v[h % self.dim] += 1.0

        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def gerar(self, mensagens, sistema=""):
        from app.llm import Resposta

        self.chamadas += 1
        self.prompts.append((mensagens[0].get("content", "")
                             if mensagens else "")[:600])
        if self.falhar:
            from app.llm import OllamaIndisponivel

            raise OllamaIndisponivel("duplo em falha")
        return Resposta(texto=self._texto, modelo="duplo", latencia_s=0.01,
                        tokens=10)


@pytest.fixture
def bd_duplo():
    return BDduplo()


@pytest.fixture
def llm_duplo():
    return LLMduplo()


@pytest.fixture
def rag_duplo(bd_duplo, llm_duplo, cfg_teste):
    from app.rag import MotorRAG

    return MotorRAG(cfg_teste, bd_duplo, llm_duplo)


@pytest.fixture
def agente_duplo(bd_duplo, llm_duplo, rag_duplo, cfg_teste):
    from app.agente import AgenteAtendimento

    return AgenteAtendimento(cfg_teste, rag_duplo, llm_duplo, bd_duplo)
