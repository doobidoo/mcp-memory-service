# Hermes

Esta página foi movida para **[`agents/hermes/README.md`](../../agents/hermes/README.md)**.

A documentação pública do Hermes como cliente do `mcp-memory-service` agora vive junto das
demais integrações, em inglês, no caminho `agents/<harness>/README.md` — a convenção
estabelecida pelo upstream no PR #1618 (mergeado em 21/out/2026).

Motivo da consolidação: a versão anterior desta página estava **fork-only**, em português, e
divergia do produto em três pontos medidos — chamava o instalador de `instalar.py` quando o
arquivo mergeado é `scripts/hermes/install.py`, fixava a porta de uma única deployment (3202)
quando a porta é configuração, e repetia o conteúdo que subiu ao upstream.

O mapa por camadas (1 a 5) de cada harness sobe ao upstream em `agents/README.md` (PR #1617, em
revisão) e, enquanto ele não mergeia, a versão viva é a wiki
[Memory-Portability-Map](https://github.com/doobidoo/mcp-memory-service/wiki/Memory-Portability-Map).
