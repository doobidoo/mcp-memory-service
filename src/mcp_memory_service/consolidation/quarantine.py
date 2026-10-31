"""Memory quarantine — holds memories that contradict active beliefs."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, List

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

CONTRADICTION_THRESHOLD = int(os.getenv("MCP_QUARANTINE_CONTRADICTION_THRESHOLD", "3"))

# Minimum NLI confidence for a contradiction to quarantine a memory on store.
# Configurable via MCP_QUARANTINE_NLI_THRESHOLD (issue #1216); read at call time
# so runtime config and tests take effect. The default heuristic backend tops
# out below this (see NLIClassifier.max_achievable_confidence), so the default
# config needs MCP_NLI_BACKEND=cascade or a lowered gate to quarantine on store.
DEFAULT_QUARANTINE_NLI_THRESHOLD = 0.7

# Orçamento total do check de crenças no caminho de escrita (issue #1630).
# Sem teto, uma gravação com N crenças elegíveis custa N chamadas de LLM
# sequenciais: medido 27 s com 20 crenças e ~1,2 s por chamada. O orçamento é
# fail-open — ao estourar, a verificação para e a gravação segue normalmente.
# 2 s é o menor valor que ainda permite UMA comparação real com o LLM medido
# (~1,3 s); com 1 s o check não compara nada e a feature vira no-op silencioso.
DEFAULT_ON_STORE_BUDGET_S = 2.0
DEFAULT_ON_STORE_MAX_BELIEFS = 20

# Quantas verificações de background podem rodar ao mesmo tempo (modo assíncrono).
# O excesso é descartado com WARNING: sem teto, uma rajada de gravações vira uma
# rajada de chamadas de LLM.
DEFAULT_BACKGROUND_CONCURRENCY = 2

# Backends already warned about below (one warning per backend per process).
_gate_warned_backends: set = set()


def _quarantine_nli_threshold() -> float:
    """The on-store quarantine gate, parsed with the repo's fallback pattern.

    An unparsable or out-of-range MCP_QUARANTINE_NLI_THRESHOLD logs an error
    and falls back to the default instead of raising — a raise here would be
    swallowed by the callers' broad handlers and silently disable quarantine.
    """
    from ..config import safe_get_float_env
    return safe_get_float_env(
        "MCP_QUARANTINE_NLI_THRESHOLD", DEFAULT_QUARANTINE_NLI_THRESHOLD,
        min_value=0.0, max_value=1.0,
    )


def _on_store_budget_s() -> float:
    """Orçamento total (segundos) do check de crenças no caminho de escrita.

    Lido a cada chamada, como o gate, para que configuração de runtime e testes
    tenham efeito sem reiniciar. Valor fora da faixa cai no default com erro
    registrado, e não desabilita a verificação em silêncio.
    """
    from ..config import safe_get_float_env
    return safe_get_float_env(
        "MCP_NLI_ON_STORE_BUDGET_S", DEFAULT_ON_STORE_BUDGET_S,
        min_value=0.0, max_value=120.0,
    )


def _on_store_max_beliefs() -> int:
    """Quantas crenças, no máximo, o check compara por gravação."""
    from ..config import safe_get_int_env
    return safe_get_int_env(
        "MCP_QUARANTINE_MAX_BELIEFS", DEFAULT_ON_STORE_MAX_BELIEFS,
        min_value=0, max_value=1000,
    )


def _warn_if_gate_unreachable(classifier, threshold: float) -> None:
    """Emit one warning per process if no contradiction could ever meet the gate.

    Makes the previously-silent dead configuration (issue #1216) visible: a gate
    above the active backend's achievable ceiling means quarantine-on-store can
    never fire. Robust to a mocked classifier (a non-numeric ceiling is skipped).
    """
    backend = getattr(classifier, "backend", "?")
    if backend in _gate_warned_backends:
        return
    ceiling = getattr(classifier, "max_achievable_confidence", None)
    if ceiling is None:
        return
    try:
        ceiling = ceiling()
    except Exception:
        return
    if isinstance(ceiling, (int, float)) and not isinstance(ceiling, bool) and threshold > ceiling:
        _gate_warned_backends.add(backend)
        logger.warning(
            "MCP_QUARANTINE_NLI_THRESHOLD=%s exceeds the '%s' NLI backend's maximum "
            "achievable confidence (%s); no contradiction can be quarantined on store "
            "with this configuration. Lower MCP_QUARANTINE_NLI_THRESHOLD to <= %s, or set "
            "MCP_NLI_BACKEND=cascade.",
            threshold, backend, ceiling, ceiling,
        )


async def quarantine_memory(
    storage,
    content_hash: str,
    contradicted_belief_hash: Optional[str],
    reason: str = "",
    contradicted_memory_hash: Optional[str] = None,
) -> dict:
    """Quarantine a memory that contradicts an active belief — or, for a
    value-swap rescued from semantic dedup (issue #1216), another *memory*.

    The two are recorded in distinct fields (``contradicted_belief`` /
    ``contradicted_memory``) so consumers can tell a belief contradiction from
    a memory collision; a memory hash never counts toward a belief's
    contradiction tally.
    """
    try:
        quarantine_meta = {
            "quarantined": True,
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "contradicted_belief": contradicted_belief_hash,
            "quarantine_reason": reason,
        }
        result = {"status": "quarantined", "content_hash": content_hash, "belief": contradicted_belief_hash}
        if contradicted_memory_hash:
            quarantine_meta["contradicted_memory"] = contradicted_memory_hash
            result["memory"] = contradicted_memory_hash
        ok, msg = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta, "tags": ["quarantined"]},
            preserve_timestamps=True,
        )
        if not ok:
            # update_memory_metadata reports ordinary failures by returning
            # (False, message) instead of raising. Treating a non-raising
            # failure as success would report a memory as quarantined while it
            # actually stays dedup-bypassed and active — the very leak #1216
            # asks to close. Propagate the error so _file_contradiction and
            # check_beliefs_on_store see status != "quarantined".
            logger.error("Failed to quarantine memory %s: %s", content_hash[:8], _sanitize_log_value(msg))
            return {"status": "error", "message": msg}
        return result
    except Exception as e:
        logger.error("Failed to quarantine memory: %s", _sanitize_log_value(str(e)))
        return {"status": "error", "message": str(e)}


async def unquarantine_memory(storage, content_hash: str) -> dict:
    """Remove quarantine from a memory."""
    try:
        quarantine_meta = {
            "quarantined": False,
            "unquarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        ok, msg = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta},
            preserve_timestamps=True,
        )
        if not ok:
            # Same honest-reporting rule as quarantine_memory: a non-raising
            # (False, message) failure must not be reported as unquarantined.
            logger.error("Failed to unquarantine memory %s: %s", content_hash[:8], _sanitize_log_value(msg))
            return {"status": "error", "message": msg}
        return {"status": "unquarantined", "content_hash": content_hash}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def check_beliefs_on_store(storage, belief_service, content: str, content_hash: str) -> Optional[dict]:
    """Check if new memory contradicts any active belief.

    Called from the on_store path (MCP_NLI_ON_STORE=true).

    O custo é limitado por dois tetos, ambos configuráveis: o número de crenças
    comparadas (MCP_QUARANTINE_MAX_BELIEFS) e o tempo total do check
    (MCP_NLI_ON_STORE_BUDGET_S, default 1 s). O orçamento é fail-open: ao
    estourar, a verificação para e a gravação segue — sem exceção. Uma linha
    INFO por execução registra duração, comparações e motivo da parada, para
    que o custo deixe de ser invisível (issue #1630).
    """
    from ..reasoning.nli import NLIClassifier

    started = time.monotonic()
    budget = _on_store_budget_s()
    max_beliefs = _on_store_max_beliefs()
    beliefs: list = []
    compared = 0
    stop_reason = "no-beliefs"

    try:
        beliefs = await belief_service.get_beliefs(status="active", min_confidence=0.35)
        if not beliefs:
            return None

        classifier = NLIClassifier(backend="auto")
        threshold = _quarantine_nli_threshold()
        _warn_if_gate_unreachable(classifier, threshold)
        # Distingue "olhou tudo" de "parou pelo teto de contagem" (Greptile, #1633).
        stop_reason = "max-beliefs" if len(beliefs) > max_beliefs else "completed"

        for belief in beliefs[:max_beliefs]:
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                stop_reason = "budget"
                break

            compared += 1
            try:
                result = await asyncio.wait_for(
                    classifier.classify(belief["content"], content), timeout=remaining
                )
            except (asyncio.TimeoutError, TimeoutError):
                stop_reason = "timeout"
                break

            if result.label == "contradiction" and result.confidence >= threshold:
                q_result = await quarantine_memory(
                    storage, content_hash, belief["belief_hash"],
                    reason=f"Contradicts belief: {belief['content'][:100]}",
                )

                contradiction_count = await _count_quarantined_for_belief(storage, belief["belief_hash"])
                if contradiction_count >= CONTRADICTION_THRESHOLD:
                    await belief_service.challenge_belief(belief["belief_hash"])
                    logger.info(f"Belief {belief['belief_hash'][:8]} challenged after {contradiction_count} contradictions")

                stop_reason = "contradiction"
                return q_result

        return None
    finally:
        elapsed = time.monotonic() - started
        logger.info(
            "NLI on-store check: %.2fs, %d of %d belief(s) compared, stop=%s",
            elapsed,
            compared,
            min(len(beliefs), max_beliefs),
            stop_reason,
        )
        # INFO é suprimido no modo Claude Desktop (protocolo JSON). Um check
        # truncado é anômalo o bastante para aparecer como WARNING nos dois modos:
        # é o sinal de que contradições podem estar passando sem exame.
        if compared and stop_reason in ("budget", "timeout"):
            logger.warning(
                "NLI on-store check truncated at %.2fs after %d belief(s) (%s); "
                "raise MCP_NLI_ON_STORE_BUDGET_S if contradictions are being missed",
                elapsed,
                compared,
                stop_reason,
            )
        if stop_reason == "max-beliefs":
            logger.warning(
                "NLI on-store check covered %d of %d belief(s) (MCP_QUARANTINE_MAX_BELIEFS=%d); "
                "raise it to widen coverage",
                compared,
                len(beliefs),
                max_beliefs,
            )
        # O orçamento governa a fase de comparação (rede). Uma quarentena já
        # iniciada é concluída: abortar no meio deixaria metadata/edges
        # inconsistentes. O estouro fica visível no log.
        if stop_reason == "contradiction" and elapsed > budget:
            logger.warning(
                "NLI on-store check persisted a contradiction after %.2fs (budget %.2fs); "
                "persistence is not aborted to keep quarantine state consistent",
                elapsed,
                budget,
            )


async def _count_quarantined_for_belief(storage, belief_hash: str) -> int:
    """Count memories quarantined due to a specific belief."""
    try:
        results = await storage.search_by_tag(["quarantined"])
        count = 0
        for mem in results:
            meta = mem.metadata if hasattr(mem, "metadata") else {}
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if meta.get("contradicted_belief") == belief_hash:
                count += 1
        return count
    except Exception:
        return 0


async def get_quarantined_memories(storage, limit: int = 50) -> List[dict]:
    """List all quarantined memories."""
    try:
        results = await storage.search_by_tag(["quarantined"])
        quarantined = []
        for mem in results[:limit]:
            meta = mem.metadata if hasattr(mem, "metadata") else {}
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if meta.get("quarantined"):
                quarantined.append({
                    "content_hash": mem.content_hash,
                    "content": mem.content[:200],
                    "contradicted_belief": meta.get("contradicted_belief"),
                    "contradicted_memory": meta.get("contradicted_memory"),
                    "quarantined_at": meta.get("quarantined_at"),
                    "reason": meta.get("quarantine_reason", ""),
                })
        return quarantined
    except Exception:
        return []


# ── Verificação em background (MCP_NLI_ON_STORE_ASYNC) ──────────────────────
# A cura do caminho de escrita é não esperar pelo LLM: a memória é gravada, a
# resposta volta, e as verificações rodam depois. As referências ficam num set
# para que o event loop não colete a task no meio.

_background_checks: set = set()


class _BackgroundState:
    """Estado do trabalho de background (objeto em vez de global solta)."""

    def __init__(self) -> None:
        self.active = 0


_background_state = _BackgroundState()


def _background_concurrency() -> int:
    """Quantas verificações de background podem rodar ao mesmo tempo."""
    from ..config import safe_get_int_env
    return safe_get_int_env(
        "MCP_NLI_BACKGROUND_CONCURRENCY", DEFAULT_BACKGROUND_CONCURRENCY,
        min_value=1, max_value=32,
    )


async def _run_on_store_checks(storage, content: str, content_hash: str) -> None:
    try:
        try:
            from ..reasoning.nli import detect_contradictions_nli

            result = await detect_contradictions_nli(
                storage, memory_hash=content_hash, dry_run=False
            )
            logger.info(
                "NLI background check (%s): pairs_detected=%s",
                content_hash[:8],
                result.get("pairs_detected", 0),
            )
        except Exception as e:
            logger.warning("NLI background contradiction check failed: %s", _sanitize_log_value(e))

        try:
            from .belief_service import BeliefService

            outcome = await check_beliefs_on_store(
                storage, BeliefService(storage), content, content_hash
            )
            # O resultado não pode sumir: sem hash e status, ninguém sabe qual
            # memória foi quarentenada nem se a escrita de metadata funcionou.
            status = (outcome or {}).get("status", "clean")
            if status == "error":
                logger.warning(
                    "NLI background belief check (%s) failed: %s",
                    content_hash[:8],
                    _sanitize_log_value(str((outcome or {}).get("message", ""))),
                )
            else:
                logger.info(
                    "NLI background belief check (%s): status=%s",
                    content_hash[:8],
                    status,
                )
        except Exception as e:
            logger.warning("NLI background belief check failed: %s", _sanitize_log_value(e))
    finally:
        _background_state.active -= 1


def schedule_on_store_checks(storage, content: str, content_hash: str) -> bool:
    """Agenda as verificações de NLI fora do caminho da escrita.

    Ligado por MCP_NLI_ON_STORE_ASYNC. O `store` responde sem esperar o LLM; o
    resultado das verificações vai para o log, não para a resposta.

    Concorrência limitada por MCP_NLI_BACKGROUND_CONCURRENCY (default 2). Com o
    limite atingido, o trabalho em excesso é **descartado com WARNING** — política
    explícita, em vez de uma fila de tasks e chamadas de LLM sem teto.
    """
    limite = _background_concurrency()
    if _background_state.active >= limite:
        logger.warning(
            "NLI background check dropped for %s: %d already running "
            "(MCP_NLI_BACKGROUND_CONCURRENCY=%d)",
            content_hash[:8],
            _background_state.active,
            limite,
        )
        return False

    # Reserva a vaga na admissão: create_task só roda depois, e sem isso duas
    # chamadas seguidas passariam pelo limite antes da primeira task começar.
    _background_state.active += 1
    try:
        task = asyncio.create_task(_run_on_store_checks(storage, content, content_hash))
    except Exception:
        _background_state.active -= 1
        raise
    _background_checks.add(task)
    task.add_done_callback(_background_checks.discard)
    return True
