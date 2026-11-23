"""Cria o schema da base de dados (Atende-IA).

USO
    python init_db.py

POR QUE UM SCRIPT E NAO UM FLAG
    O `app/database.py` e' uma biblioteca (classes e funcoes). Meter a CLI
    dentro dele obrigaria a importar psycopg2 para qualquer consumidor do
    modulo, mesmo quem so' quer a classe. Um script separado mantem a
    biblioteca limpa e a inicializacao explicita.

O QUE FAZ
    1. le a configuracao (o .env tem de estar preenchido)
    2. testa a ligacao
    3. cria as tabelas e o indice HNSW (idempotente: pode correr varias vezes)
    4. verifica o resultado lendo o catalogo do Postgres (nao confia no "OK")

Exemplo de saida:
    [OK] ligacao        Postgres ok - pgvector 0.8.7
    [OK] schema         6/6 objectos
    [OK] tabelas        atende_conversas, atende_documentos, atende_usuarios
    [OK] indice HNSW    atende_docs_hnsw_idx
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.config import config
        from app.database import BaseDados
    except Exception as e:  # noqa: BLE001
        print("ERRO ao importar a aplicacao: %s: %s" % (type(e).__name__, e))
        print("Confirme que corre a partir da raiz do projecto.")
        return 1

    try:
        cfg = config()
    except Exception as e:  # noqa: BLE001
        print("ERRO na configuracao: %s" % e)
        print("\nCopie .env.example para .env e preencha os valores.")
        print("Falta frequentemente: PG_PASSWORD e ATENDE_JWT_SECRET")
        return 1

    bd = BaseDados(cfg)

    falhas = 0

    def check(nome: str, ok: bool, detalhe: str = "") -> None:
        nonlocal falhas
        if not ok:
            falhas += 1
        print("[%s] %-16s %s" % ("OK" if ok else "FALHA", nome, detalhe))

    # 1) ligacao --------------------------------------------------------
    try:
        bd.ligar()
        ok, msg = bd.saudavel()
        check("ligacao", ok, msg)
        if not ok:
            return 1
    except Exception as e:  # noqa: BLE001
        check("ligacao", False, "%s: %s" % (type(e).__name__, e))
        print("\nVerifique PG_HOST/PG_PORT/PG_USER/PG_PASSWORD no .env.")
        return 1

    # 2) schema ---------------------------------------------------------
    try:
        r = bd.criar_schema()
    except Exception as e:  # noqa: BLE001
        check("schema", False, "%s: %s" % (type(e).__name__, e))
        return 1

    # criar_schema pode devolver dict com contagens; aceita as duas formas
    if isinstance(r, dict):
        total = sum(int(v) for v in r.values() if isinstance(v, (int, float)))
        check("schema", total > 0, "%d operacao(oes)" % total)
    else:
        check("schema", True, str(r)[:60])

    # 3) prova: ler o catalogo -----------------------------------------
    try:
        with bd.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name LIKE %s "
                "ORDER BY table_name", (cfg.base.prefixo + "%",))
            tabelas = [row["table_name"] for row in cur.fetchall()]
        check("tabelas", len(tabelas) >= 3, ", ".join(tabelas) or "nenhuma")

        with bd.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = %s AND indexdef ILIKE %s",
                (cfg.base.prefixo + "documentos", "%hnsw%"))
            idx = [row["indexname"] for row in cur.fetchall()]
        check("indice HNSW", bool(idx), ", ".join(idx) or "AUSENTE")

        with bd.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            row = cur.fetchone()
        check("pgvector", row is not None,
              ("v" + str(row["extversion"])) if row else "extensao ausente")
    except Exception as e:  # noqa: BLE001
        check("verificacao", False, "%s: %s" % (type(e).__name__, e))

    print("\n" + "=" * 60)
    if falhas == 0:
        print("RESULTADO: base pronta. Arranque com:")
        print("  uvicorn app.api:app --reload --port 8100")
    else:
        print("RESULTADO: %d falha(s) — corrija antes de arrancar." % falhas)
    return 0 if falhas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
