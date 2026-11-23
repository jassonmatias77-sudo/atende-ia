"""Cria o primeiro administrador (Atende-IA).

O PROBLEMA QUE ISTO RESOLVE
    A rota POST /auth/utilizador exige papel de administrador. Se a base esta'
    vazia, nao ha' admin — logo nao ha' forma de criar utilizadores pela API.
    E' o classico "ovo e a galinha": quem clona o repositorio fica bloqueado.

A SOLUCAO
    Um script de linha de comandos que escreve directamente na base, com a
    mesma funcao (ServicoAuth.criar_utilizador) que a API usa. Sem caminho
    alternativo: se a API mudar a regra, isto acompanha.

SEGURANCA
    A senha NAO vem por argumento (ficaria no historico do PowerShell).
    Vem por variavel de ambiente ou por input interactivo.

USO
    python criar_admin.py
    # ou
    $env:ATENDE_ADMIN_EMAIL="eu@exemplo.com"; $env:ATENDE_ADMIN_SENHA="..."
    python criar_admin.py
"""
from __future__ import annotations

import getpass
import os
import sys


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    try:
        from app.auth import ErroAuth, ServicoAuth
        from app.config import config
    except Exception as e:  # noqa: BLE001
        print("ERRO ao importar: %s: %s" % (type(e).__name__, e))
        return 1

    try:
        cfg = config()
    except Exception as e:  # noqa: BLE001
        print("ERRO na configuracao: %s" % e)
        print("Copie .env.example para .env e preencha (PG_PASSWORD, ATENDE_JWT_SECRET).")
        return 1

    # ------------------------------------------------------------ credenciais
    email = os.environ.get("ATENDE_ADMIN_EMAIL", "").strip()
    senha = os.environ.get("ATENDE_ADMIN_SENHA", "")

    if not email:
        try:
            email = input("E-mail do administrador: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ncancelado")
            return 1

    if not senha:
        try:
            s1 = getpass.getpass("Senha (nao aparece): ")
            s2 = getpass.getpass("Repita a senha: ")
        except (EOFError, KeyboardInterrupt):
            print("\ncancelado")
            return 1
        if s1 != s2:
            print("ERRO: as senhas nao coincidem.")
            return 1
        senha = s1

    if not email or len(email) < 3:
        print("ERRO: e-mail invalido.")
        return 1
    if len(senha) < 8:
        # a API aceita 6; aqui exigimos mais porque e' a conta com mais poder
        print("ERRO: use pelo menos 8 caracteres (conta de administrador).")
        return 1

    # ------------------------------------------------------------ criar
    auth = ServicoAuth(cfg)
    try:
        uid = auth.criar_utilizador(email, senha, papel="admin")
    except ErroAuth as e:
        print("ERRO de autenticacao: %s" % e)
        return 1
    except Exception as e:  # noqa: BLE001
        print("ERRO ao gravar: %s: %s" % (type(e).__name__, e))
        print("A base esta' acessivel? Corra: python init_db.py")
        return 1

    print("\n[OK] administrador criado/atualizado")
    print("     id    : %s" % uid)
    print("     e-mail: %s" % email.lower().strip())
    print("     papel : admin")
    print("\nEntre com:")
    print('  POST /auth/token  {"email": "%s", "senha": "..."}' % email.lower().strip())

    # prova: autentica imediatamente para nao deixar duvida
    try:
        tok = auth.autenticar(email, senha)
        print("\n[OK] login testado — token de %d caracteres emitido." % len(tok))
    except Exception as e:  # noqa: BLE001
        print("\n[AVISO] gravou mas o login falhou: %s" % e)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
