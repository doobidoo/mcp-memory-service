"""Regressão: filtrar por tipo/tag DEPOIS de recuperar só `n_results` esvazia o resultado.

`retrieve_memories` pede `n_results` ao storage e só então aplica os filtros de
`tags`/`memory_type`. Quando o filtro casa com memórias que não estão entre os
`n_results` mais próximos, o resultado sai vazio — e é o que faz `mistake_note_search`
devolver `{"notes": [], "count": 0}` com 295 notas de erro no acervo.

Deve FALHAR em `upstream/main` (RED) e passar com a correção (GREEN).
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.services.memory_service import MemoryService


class _StubStorage:
    """Storage mínimo: devolve o que foi pedido, respeitando `n_results`."""

    def __init__(self, resultados: list[MemoryQueryResult]):
        self._resultados = resultados
        self.chamadas: list[int] = []

    async def retrieve(self, query: str, n_results: int = 10):  # noqa: D401
        self.chamadas.append(n_results)
        return self._resultados[:n_results]


def _resultados() -> list[MemoryQueryResult]:
    itens = []
    for i in range(10):
        tipo = "mistake" if i == 9 else "observation"
        mem = Memory(
            content=f"memoria {i} do tipo {tipo}",
            content_hash=f"hash-{i}",
            memory_type=tipo,
        )
        itens.append(MemoryQueryResult(memory=mem, relevance_score=1.0 - i * 0.05))
    return itens


def test_filtro_por_tipo_encontra_memoria_fora_da_janela_inicial():
    storage = _StubStorage(_resultados())
    service = MemoryService(storage)

    resultado: dict[str, Any] = asyncio.run(
        service.retrieve_memories("porta errada no systemd", n_results=5, memory_type="mistake")
    )

    assert resultado["count"] == 1, (
        "filtro por memory_type nao encontrou a nota de erro que existe fora dos 5 "
        f"primeiros resultados; storage recebeu n_results={storage.chamadas} e o "
        f"resultado veio com {resultado['count']} itens"
    )
    assert resultado["memories"][0]["content"].startswith("memoria 9")


def test_resposta_respeita_n_results_mesmo_com_muitos_casando():
    """Over-fetch não pode devolver resposta maior que o pedido."""
    itens = []
    for i in range(10):
        mem = Memory(
            content=f"nota de erro {i}",
            content_hash=f"m-{i}",
            memory_type="mistake",
        )
        itens.append(MemoryQueryResult(memory=mem, relevance_score=1.0 - i * 0.05))

    storage = _StubStorage(itens)
    service = MemoryService(storage)

    resultado = asyncio.run(
        service.retrieve_memories("erro recorrente", n_results=5, memory_type="mistake")
    )

    assert resultado["count"] == 5, (
        "a resposta deve ser truncada em n_results depois do filtro; veio "
        f"{resultado['count']} itens (storage recebeu n_results={storage.chamadas})"
    )


def test_filtro_por_tag_sozinho_tambem_amplia_a_busca():
    """Filtro só por tag sofre do mesmo problema e precisa do mesmo over-fetch."""
    itens = []
    para_marcar = {3, 8}
    for i in range(10):
        mem = Memory(
            content=f"memoria {i}",
            content_hash=f"t-{i}",
            tags=["lancamento"] if i in para_marcar else ["outra"],
            memory_type="note",
        )
        itens.append(MemoryQueryResult(memory=mem, relevance_score=1.0 - i * 0.05))

    storage = _StubStorage(itens)
    service = MemoryService(storage)

    resultado = asyncio.run(
        service.retrieve_memories("planejamento", n_results=5, tags=["lancamento"])
    )

    assert resultado["count"] == 2, (
        "filtro por tag perdeu as memorias marcadas fora da janela inicial; veio "
        f"{resultado['count']} itens (storage recebeu n_results={storage.chamadas})"
    )
