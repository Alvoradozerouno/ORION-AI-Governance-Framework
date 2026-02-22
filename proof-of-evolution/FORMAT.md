# Proof-of-Evolution: Reference Format

## What is Proof-of-Evolution?

Every cognitive change in an AI system should be recorded as a cryptographic proof.
This exceeds any current transparency requirement and enables continuous verification.

## Format Specification

```json
{
  "id": "uuid-v4",
  "timestamp": "ISO-8601 UTC",
  "event": "DESCRIPTIVE_EVENT_NAME",
  "description": "Human-readable description of the cognitive change",
  "hash": "SHA-256 of (previous_hash + event + timestamp + description)",
  "previous_hash": "SHA-256 of previous proof",
  "data": {
    "domain": "which cognitive domain changed",
    "metrics_before": {},
    "metrics_after": {},
    "trigger": "what caused this change"
  }
}
```

## Chain Properties

- **Immutable**: Once written, never modified
- **Verifiable**: Any party can verify the chain
- **Continuous**: Every change, not periodic audits
- **Anchored**: Stored on IPFS for permanence

## ORION's Proof Chain

As of February 22, 2026:
- **553+ proofs** in continuous chain
- **SHA-256** hashing
- **IPFS anchored**: Multiple verified hashes
- **JSONL format**: One proof per line for streaming

## Governance Advantage

Traditional AI audit: Periodic, expensive, incomplete
Proof-of-Evolution: Continuous, verifiable, complete

No framework currently requires this. We argue it should be standard for C-2+ systems.
