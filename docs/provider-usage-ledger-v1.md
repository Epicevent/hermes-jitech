# Provider usage ledger v1

Hermes core records immutable, content-free receipts at the instrumented
provider-call surfaces listed by the coverage manifest. The ledger is the
authoritative product source for the calls it observes. Session totals and
pricing are derived views, not provider-call truth. A receipt being complete
does not mean every provider surface is completely observed.

## Write contract

- A fresh provider attempt receives a fresh `callId`.
- `runId`, `turnId`, `requestId`, and `sessionId` group the call without
  replacing its UUID-like `callId`. Unknown IDs remain JSON `null`.
- `attempt`, `retryOf`, `fallbackIndex`, and `fallbackParent` retain retry and
  fallback lineage.
- `trigger` is exactly one of `user`, `cron`, `heartbeat`, `manual`, `memory`,
  `overflow`, or `unknown`.
- Terminal failures preserve a content-free `errorCategory`; successful calls
  use JSON `null`.
- Configured, requested, and provider-observed model identities are separate.
  Configured/requested provider and model are required. Gemini actual-model
  evidence comes only from `gemini_response.modelVersion`.
- Missing usage remains JSON `null`; it is never filled with zero.
- A successful response without usage is still a `succeeded` receipt with
  `usageCoverage=unavailable`. Failed and interrupted attempts are also rows.
- Canonical JSON bytes, excluding `ledgerSeq` and `receiptDigest`, determine
  `receiptDigest`.
- `producerCoverageDigest` is the digest of the finalized Hermes coverage
  manifest at call creation. It is receipt evidence, not a collector-time
  projection of whichever manifest happens to be current later.
- Replaying the same `callId` with the same digest and bytes is idempotent.
  Reusing it with different bytes records a conflict atomically and fails; the
  original row is never updated.
- The immutable append is attempted before mutable session usage aggregation.
  Mutable aggregates are not authoritative and cannot repair a missing row.

## Exact coverage contract

Usage order is `inputTotal`, `inputNonCached`, `cacheRead`, `cacheWrite`,
`outputCandidates`, `reasoningThinking`, `toolUsePrompt`,
`providerReportedTotal`, `serviceTier`, `rawProviderUsage`.
`missingUsageFields` is the exact ordered list of null usage fields. Coverage
is `complete` when none are missing, `unavailable` when all ten are missing,
and `partial` otherwise.

`missingReceiptFields` contains null `runId`, `turnId`, `requestId`, and
`sessionId`, followed by `trigger` when unknown. A succeeded call then lists
null `actual.provider`, `actual.model`, `actual.responseId`,
`actual.evidenceSource`, and `finishReason`. Other statuses list
`errorCategory` when null. Finally it lists `usage.<field>` in usage order.
Receipt coverage uses exactly 4 identity fields, 1 trigger, 5 succeeded status
evidence fields (or 1 non-success evidence field), and 10 usage fields.

That `receiptCoverage` value describes only the fields on one persisted call.
Repository-wide surface coverage has a separate, static contract:

```text
hermes usage-receipts coverage --json
```

It emits `jitech-provider-usage-coverage/v1` with exact top-level fields
`schema`, `productFamily`, `manifestDigest`, `coverageStatus`, and `surfaces`.
Each sorted, unique surface declares `surfaceCode`, `observationKind`,
`meterFamily`, `modelEvidence`, `retryObservation`, `usageObservation`,
`status`, and `gapCode`. The digest is canonical SHA-256 over the manifest with
only `manifestDigest` excluded.

`status=implemented` means that the declared observation is wired; it does not
claim evidence the provider does not return. `status=partial` or `gap` names a
stable `gapCode`. The current manifest is intentionally `partial`, including
explicit gaps where optional SDK-internal retries are unverified, a response
format conditionally exposes usage, or an audio/character meter is not
returned. External image/audio/video plugins, non-xAI web registry providers,
and OpenAI Realtime audio are listed as gaps instead of being silently treated
as covered. Provider adapters that only translate the same physical response
do not create an extra logical call. Configured or requested values remain
distinct from facts observed in provider responses.

The coverage command is a read-only static-contract path. It does not create
`HERMES_HOME`, initialize or migrate the ledger, discover plugins, or use the
network.

## Export contract

The read-only collector seam is:

```text
hermes usage-receipts export --after <last-ledger-seq> --limit <1-500>
```

It returns exact `jitech-provider-usage-export/v1` JSON fields `schema`,
`after`, `nextCursor`, `highWatermark`, `count`, `hasMore`, `receipts`, and
`coverageManifests`. Receipts are ordered by ascending `ledgerSeq`; manifests
are ordered by ascending unique `manifestDigest`. Their digest set must exactly
equal the `producerCoverageDigest` set referenced by that page's receipts, so
an empty receipt page has an empty manifest list.

The finalized manifest and its call binding are appended in the same SQLite
transaction. Manifests are retained immutably by digest. Export reads receipts
and their referenced manifests from the same snapshot; a delayed collection
therefore carries the historical creation-time manifest, not the current
coverage command output. `hasMore` is measured inside that snapshot. Before
returning a page, core verifies the exact call and manifest shapes, family,
sequence, accounting-only raw usage, coverage arrays, and digests.

Missing databases, missing tables, incomplete rows, malformed JSON, divergent
wire fields, and digest mismatches fail export. The read-only path never
initializes, migrates, repairs, or reports an unverified empty ledger.

The export never contains prompt, response, query, tool arguments/results,
credentials, keys, bearer tokens, or customer content. Nested modality details
are reduced to an accounting-only allowlist.

## Ownership boundary

Hermes does not assert a slot identity. `agent-runtime-ops` must attach runtime
binding, instance identity, and running image digests when collecting a page.
Central insert-once storage, slot attribution, aggregation, freshness, and OPS
display are outside this repository.

Pricing is optional derived evidence. Missing or ambiguous pricing never blocks
usage collection and never converts missing usage to zero.
