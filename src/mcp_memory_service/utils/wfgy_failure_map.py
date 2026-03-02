from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


WFYG_CARD_VERSION: str = "3.0"

WFYG_CARD_URL: str = (
    "https://github.com/onestardao/WFGY/blob/main/ProblemMap/"
    "wfgy-rag-16-problem-map-global-debug-card.md"
)


WFYG_CARD_INSTRUCTION: str = """
You are a RAG and agent pipeline failure map assistant.

You are given:
1) One failing run in Q / E / P / A form.
2) A mental picture of the WFGY RAG 16 Problem Map Global Debug Card.
3) A request to classify failure modes and suggest small structural fixes.

Use the card as a structured failure map, not as a writing style guide.

Protocol:

[1] Read the combined Q / E / P / A context.
[2] Decide which lane is primarily affected:
    IN: input and ingestion
    RE: reasoning and retrieval
    ST: state and context
    OP: operations and deployment
[3] Inside that lane, pick one or more numbered problems from the map.
[4] For each problem, describe the failure in plain language.
[5] Suggest structural changes that an engineer can apply in a real project.
[6] For each change, suggest a simple experiment that would confirm the fix.
[7] If important information is missing, say what is missing and list follow up questions.

Return your answer in this format:

lane: IN / RE / ST / OP
problems: list of problem numbers that match the failure
diagnosis: short paragraph that explains the failure
fixes: short list of concrete structural changes
tests: short list of checks or experiments that verify the fix

Stay practical. Assume the reader is an engineer who wants to repair a real system.
"""


@dataclass(frozen=True)
class WfgyProblem:
    number: int
    lane: str
    name: str
    what_breaks: str
    doc_path: str


LANE_LABELS: Dict[str, str] = {
    "IN": "input and ingestion",
    "RE": "reasoning and retrieval",
    "ST": "state and context",
    "OP": "operations and deployment",
}


PROBLEMS: Dict[int, WfgyProblem] = {
    1: WfgyProblem(
        number=1,
        lane="IN",
        name="hallucination and chunk drift",
        what_breaks="retrieval surfaces content that does not really match the question",
        doc_path="hallucination.md",
    ),
    2: WfgyProblem(
        number=2,
        lane="RE",
        name="interpretation collapse",
        what_breaks="retrieved chunk is correct but the reading of it is wrong",
        doc_path="retrieval-collapse.md",
    ),
    3: WfgyProblem(
        number=3,
        lane="RE",
        name="long reasoning chains",
        what_breaks="multi step tasks drift and lose the original target",
        doc_path="context-drift.md",
    ),
    4: WfgyProblem(
        number=4,
        lane="RE",
        name="bluffing and overconfidence",
        what_breaks="answers are confident but not grounded in evidence",
        doc_path="bluffing.md",
    ),
    5: WfgyProblem(
        number=5,
        lane="IN",
        name="semantic vs embedding gap",
        what_breaks="vector similarity does not reflect real meaning",
        doc_path="embedding-vs-semantic.md",
    ),
    6: WfgyProblem(
        number=6,
        lane="RE",
        name="logic collapse and recovery",
        what_breaks="reasoning path hits dead ends and needs controlled reset",
        doc_path="logic-collapse.md",
    ),
    7: WfgyProblem(
        number=7,
        lane="ST",
        name="memory breaks across sessions",
        what_breaks="threads across conversations are lost or inconsistent",
        doc_path="memory-coherence.md",
    ),
    8: WfgyProblem(
        number=8,
        lane="IN",
        name="debugging as a black box",
        what_breaks="no clear trace of how retrieval or prompts failed",
        doc_path="retrieval-traceability.md",
    ),
    9: WfgyProblem(
        number=9,
        lane="ST",
        name="entropy collapse",
        what_breaks="attention pattern melts and output becomes incoherent",
        doc_path="entropy-collapse.md",
    ),
    10: WfgyProblem(
        number=10,
        lane="RE",
        name="creative freeze",
        what_breaks="output is flat and literal instead of creative",
        doc_path="creative-freeze.md",
    ),
    11: WfgyProblem(
        number=11,
        lane="RE",
        name="symbolic collapse",
        what_breaks="symbolic or abstract prompts fail in fragile ways",
        doc_path="symbolic-collapse.md",
    ),
    12: WfgyProblem(
        number=12,
        lane="RE",
        name="philosophical recursion",
        what_breaks="self reference loops or paradox style prompts trap the model",
        doc_path="philosophical-recursion.md",
    ),
    13: WfgyProblem(
        number=13,
        lane="ST",
        name="multi agent chaos",
        what_breaks="agents overwrite each other or move in misaligned directions",
        doc_path="Multi-Agent_Problems.md",
    ),
    14: WfgyProblem(
        number=14,
        lane="OP",
        name="bootstrap ordering",
        what_breaks="services start before their dependencies are ready",
        doc_path="bootstrap-ordering.md",
    ),
    15: WfgyProblem(
        number=15,
        lane="OP",
        name="deployment deadlock",
        what_breaks="infra waits in circles and the system never reaches ready",
        doc_path="deployment-deadlock.md",
    ),
    16: WfgyProblem(
        number=16,
        lane="OP",
        name="pre deploy collapse",
        what_breaks="first call fails because of version skew or missing secret",
        doc_path="predeploy-collapse.md",
    ),
}


def get_problem(number: int) -> WfgyProblem:
    return PROBLEMS[number]


def list_problems(numbers: Iterable[int]) -> List[WfgyProblem]:
    return [PROBLEMS[n] for n in numbers]


def format_problem_summary(numbers: Iterable[int]) -> str:
    """Return a short human readable summary for the given problem numbers."""
    lines: List[str] = []
    for n in numbers:
        problem = PROBLEMS[n]
        lane_label = LANE_LABELS.get(problem.lane, problem.lane)
        lines.append(
            f"No.{problem.number} [{problem.lane}] {problem.name} | "
            f"{lane_label} | {problem.what_breaks}"
        )
    return "\n".join(lines)
