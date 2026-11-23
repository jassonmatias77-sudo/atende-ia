"""Testes do chunking e do RAG (sem serviços externos)."""
from __future__ import annotations

from app.rag import MotorRAG, chunkar


# ------------------------------------------------------------------ chunking
def test_texto_curto_nao_dividido():
    assert chunkar("texto curto") == ["texto curto"]


def test_texto_vazio():
    assert chunkar("") == []
    assert chunkar("   ") == []


def test_texto_longo_dividido():
    pedacos = chunkar("Frase de teste. " * 100, 200, 40)
    assert len(pedacos) > 1
    assert all(len(p) <= 260 for p in pedacos)


def test_sobreposicao_preserva_continuidade():
    pedacos = chunkar("abcdefghij" * 60, 100, 30)
    assert len(pedacos) >= 2
    assert pedacos[0][-15:] in pedacos[1] or pedacos[0][-10:] in pedacos[1]


def test_sobreposicao_grande_nao_entra_em_loop():
    pedacos = chunkar("palavra " * 300, 50, 49)
    assert len(pedacos) < 600


def test_quebra_em_fim_de_frase():
    """O chunk não deve começar a meio de uma palavra."""
    texto = ("Primeira frase completa. " * 20).strip()
    pedacos = chunkar(texto, 100, 20)
    for p in pedacos[1:]:
        assert not p.startswith(" ")


# ----------------------------------------------------------------------- RAG
def test_indexar_e_buscar(rag_duplo):
    rag_duplo.indexar_texto("A politica de trocas permite 30 dias.",
                            "politica.md", "Politica")
    assert rag_duplo.bd.contar_documentos() >= 1

    r = rag_duplo.buscar("quantos dias para trocar?", top_k=1)
    assert r.documentos
    assert r.confianca > 0
    assert "politica.md" in r.contexto


def test_indexar_pasta_conta_ficheiros(rag_duplo, tmp_path):
    (tmp_path / "a.md").write_text("conteudo A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("conteudo B", encoding="utf-8")
    (tmp_path / "c.log").write_text("ignorar", encoding="utf-8")

    resumo = rag_duplo.indexar_pasta(tmp_path)
    assert resumo["ficheiros"] == 2       # .log fica de fora
    assert resumo["falhas"] == 0


def test_buscar_sem_documentos(rag_duplo):
    r = rag_duplo.buscar("qualquer coisa")
    assert r.documentos == []
    assert r.confianca == 0.0
    assert not r.encontrou


def test_confianca_e_o_melhor_score(rag_duplo):
    """A confiança não pode ser a média: um chunk irrelevante no top-k
    não deve derrubar uma resposta certa."""
    rag_duplo.indexar_texto("entrega em 10 a 15 dias", "entrega.md")
    rag_duplo.indexar_texto("assunto completamente diferente", "outro.md")
    r = rag_duplo.buscar("prazo de entrega", top_k=3)
    melhor = max(d.score for d in r.documentos)
    assert abs(r.confianca - melhor) < 1e-9


def test_rag_indisponivel_devolve_vazio(llm_duplo, bd_duplo, cfg_teste):
    """Se o LLM cair, o RAG devolve vazio em vez de rebentar."""
    llm_duplo.falhar = True
    rag = MotorRAG(cfg_teste, bd_duplo, llm_duplo)
    r = rag.buscar("pergunta qualquer")
    assert r.documentos == []
    assert not r.encontrou


def test_embedding_dimensao_errada_recusado(bd_duplo, llm_duplo):
    """Um vector com dimensão errada não pode entrar na base."""
    n = bd_duplo.inserir_documentos([{"texto": "x", "fonte": "f",
                                      "embedding": [0.1] * 10}])
    assert n == 0
    assert bd_duplo.contar_documentos() == 0
