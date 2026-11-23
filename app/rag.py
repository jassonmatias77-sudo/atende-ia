"""Indexação e recuperação de conhecimento.

DIFERENÇA FACE AO PORTFÓLIO ANTERIOR
    Antes: a similaridade era calculada em Python, comparando o vector da
    pergunta com TODOS os vectores em memória (força bruta, O(n)).

    Agora: a busca é delegada ao PostgreSQL com índice HNSW (O(log n)).
    A aplicação não vê os vectores — só recebe os resultados ordenados.
    É a diferença entre um protótipo e um sistema que escala.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import Config, config as cfg_global
from app.database import BaseDados, Documento
from app.llm import ClienteOllama, ModeloNaoEncontrado, OllamaIndisponivel

log = logging.getLogger(__name__)


def chunkar(texto: str, tamanho: int = 500, sobreposicao: int = 100) -> list[str]:
    """Divide o texto em pedaços com sobreposição.

    Quebra preferencialmente em fim de frase, para o chunk não começar a meio
    de uma ideia — o que degradaria o embedding.
    """
    if not texto or not texto.strip():
        return []
    texto = re.sub(r"\n{3,}", "\n\n", texto.strip())
    if len(texto) <= tamanho:
        return [texto]

    pedacos: list[str] = []
    inicio, n = 0, len(texto)

    while inicio < n:
        fim = min(inicio + tamanho, n)
        if fim < n:
            minimo = inicio + int(tamanho * 0.6)
            for marca in ("\n\n", ". ", "! ", "? ", "\n", " "):
                pos = texto.rfind(marca, minimo, fim)
                if pos > inicio:
                    fim = pos + len(marca)
                    break
        pedaco = texto[inicio:fim].strip()
        if pedaco:
            pedacos.append(pedaco)
        avanca = fim - sobreposicao
        inicio = avanca if avanca > inicio else fim

    return pedacos


@dataclass
class ResultadoBusca:
    """O que o RAG devolve ao agente."""

    documentos: list[Documento]
    confianca: float
    contexto: str

    @property
    def encontrou(self) -> bool:
        return bool(self.documentos) and self.confianca > 0


class MotorRAG:
    """Indexa documentação e recupera contexto relevante."""

    def __init__(self, cfg: Config | None = None,
                 bd: BaseDados | None = None,
                 llm: ClienteOllama | None = None) -> None:
        self.cfg = cfg or cfg_global()
        self.bd = bd or BaseDados(self.cfg)
        self.llm = llm or ClienteOllama(self.cfg)

    # ------------------------------------------------------------- indexação
    def indexar_texto(self, texto: str, fonte: str, titulo: str = "") -> int:
        """Indexa um texto. Devolve quantos chunks foram gravados."""
        pedacos = chunkar(texto)
        if not pedacos:
            return 0

        itens = []
        for i, p in enumerate(pedacos):
            emb = self.llm.embeddings(p)
            if not emb:
                log.warning("embedding vazio para chunk %d de %s", i, fonte)
                continue
            itens.append({
                "texto": p,
                "fonte": fonte,
                "titulo": titulo or fonte,
                "embedding": emb,
            })

        return self.bd.inserir_documentos(itens)

    def indexar_pasta(self, pasta: str | Path,
                      extensoes: tuple = (".md", ".txt")) -> dict:
        """Indexa todos os ficheiros de uma pasta. Devolve o resumo."""
        base = Path(pasta)
        resumo = {"ficheiros": 0, "chunks": 0, "falhas": 0, "nomes": []}

        if not base.exists():
            return resumo

        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in extensoes:
                continue
            try:
                conteudo = f.read_text(encoding="utf-8", errors="replace")
                n = self.indexar_texto(conteudo, f.name, f.stem.replace("_", " "))
                resumo["ficheiros"] += 1
                resumo["chunks"] += n
                resumo["nomes"].append(f.name)
            except Exception as e:  # noqa: BLE001
                log.warning("falha a indexar %s: %s", f.name, e)
                resumo["falhas"] += 1

        return resumo

    # ------------------------------------------------------------- recuperação
    def buscar(self, pergunta: str, top_k: int = 4) -> ResultadoBusca:
        """Recupera contexto. Nunca levanta — devolve vazio se falhar."""
        try:
            emb = self.llm.embeddings(pergunta)
        except (OllamaIndisponivel, ModeloNaoEncontrado) as e:
            log.warning("RAG indisponível: %s", e)
            return ResultadoBusca([], 0.0, "")

        if not emb:
            return ResultadoBusca([], 0.0, "")

        documentos = self.bd.buscar(emb, top_k=top_k)
        if not documentos:
            return ResultadoBusca([], 0.0, "")

        # A confiança é o score do MELHOR documento. Usar a média puniria
        # uma resposta certa só porque havia chunks irrelevantes no top-k.
        confianca = max(d.score for d in documentos)

        blocos = []
        for i, d in enumerate(documentos, 1):
            blocos.append("[%d] (fonte: %s, score %.2f)\n%s"
                          % (i, d.fonte, d.score, d.texto))

        return ResultadoBusca(documentos, confianca, "\n\n".join(blocos))

    # ------------------------------------------------------------------ estado
    def estado(self) -> dict:
        ok, msg = self.bd.saudavel()
        return {
            "base_ok": ok,
            "base_msg": msg,
            "documentos": self.bd.contar_documentos() if ok else 0,
            "fontes": self.bd.fontes() if ok else [],
            "llm": self.llm.diagnostico(),
        }
