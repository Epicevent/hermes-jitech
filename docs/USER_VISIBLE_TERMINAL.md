# Hermes user-visible RAG terminal

## Terminal

The terminal is a real oc20 user turn in which Hermes is instructed to build an
index from the corpus mounted in that slot, then answers a scoped or unscoped
question through slot-local retrieval, dense search, GPU reranking, bounded
context, the existing provider handoff, and content-free receipts. Turning RAG
off (and restarting while it is off) must produce zero retrieval, index-open,
GPU, context-injection, and provider-handoff activity.

## Layered state

- **Source:** `codex/kwrag-product-native-no-opsctl` at the current worktree
  head; this change owns the Hermes caller/tool bridge only.
- **Build:** no new image or wrapper has been built by this change.
- **Install:** no Agent OPS or oc20 mutation has been performed.
- **Runtime:** current product-native wheel exposes the slot-local index API;
  direct oc20 UI evidence shows an index-build turn reporting 21,558 segments,
  but a later explicit verification turn reported that `kwrag_search` was not
  available. This is runtime evidence of a partial/older deployment, not a
  completed retrieval turn.
- **Actual turn:** not terminal until an oc20 UI turn produces the linked
  search/context/provider receipts and the OFF/restart negative.

## What this change closes

The sequential dashboard path now invokes `kwrag_search` through the active
agent seam, so a verified result can reach the existing provider-consumption
path. The terminal request accepts source-neutral optional scope; explicit
source/room scope is validated and never silently replaced by Kakao.

## First non-substitutable blocker

The current deployed oc20 image does not expose the `kwrag_search` tool, so the
local caller bridge is not in the product runtime yet. In addition, the
currently embedded KWRAG wheel does not advertise a source-neutral runtime
scope or a measured natural-language room catalog/router. Non-Kakao source
requests therefore fail closed until the server-side KWRAG agent publishes that
runtime API and matching wheel. These are real source/build/runtime blockers,
not reasons to invent a Hermes shim or claim a live RAG turn.

## Required positive and negative evidence

Positive evidence must show an explicit `kwrag_index_build` call, a non-empty
active index under the oc20 Workspace, a real dense→GPU-rerank search, bounded
context, provider response, and content-free receipts joined by turn/build ID.
Negative evidence must show RAG-off and post-restart RAG-off with all retrieval
and handoff counters at zero. Synthetic provider or registry-only tool tests do
not satisfy the terminal.

## Next runnable action

Run the focused Hermes tests and hand the exact source/API gap to the server
KWRAG agent at `ji` → `remote_usr2` → `kwrag-two-canary:0.0`. Do not push,
merge, build an image, install, or run an oc20 canary from this worktree.
