# Architecture — QBO Gold Layer (In-Place Refactor)

## Context

This module represents an **in-place architectural extraction** from a legacy ~1500-line finance transformation class.

Originally designed as a monolithic processor, the system is being progressively decomposed into **explicit, composable scripts** while maintaining uninterrupted production delivery.

This refactor is intentionally evolutionary rather than disruptive.

The primary objective is **restoring structural clarity without introducing operational risk.**

## Architectural Intent

The gold layer is responsible for:

> Converting standardized financial data into **analytics-ready, contract-aligned datasets** for downstream reporting systems (primarily Power BI).

Design priorities:

- Preserve production stability
- Extract reusable business logic
- Externalize contracts
- Reduce cognitive load
- Enable future orchestration (Spark / distributed compute)
- Avoid rewrite risk

This is **constraint-aware architecture**, not greenfield design.

## Module Layout

gold/  
├── _helpers.py  
├── finance_logic.py  
├── finance_budget.py  
├── finance_summary.py  
└── user_inputs.py  

utils/  
└── filesystem.py

json_configs/  
├── contracts/  
├── io/  
└── state/

## Design Pattern: Controlled Decomposition

Instead of rewriting the system, the strategy is:

> **Decompose → stabilize → iterate**

Each extracted script represents a **business capability boundary**.

### Helpers

`_helpers.py`

Contains shared transformation primitives:

- Pillar classification  
- Commodity standardization  
- Account rerouting  

These functions are intentionally stateless and reusable.

---

### Core Finance Transformation

`finance_logic.py`

Primary entrypoint for transforming standardized QBO P&L data into a curated gold dataset.

Responsibilities include:

- Account reclassification  
- Location normalization  
- Pillar derivation  
- Sign correction  

This module defines the **semantic shape** of financial truth used across downstream consumers.

---

### Budget Consolidation

`finance_budget.py`

Composes budget inputs with actual financials into a Power BI–ready structure.

Key outputs include:

- Unified monthly view  
- FX normalization  
- Contract-aligned identifiers  

This module acts as a bridge between **planned** and **observed** financial states.

## External Contracts

Business rules are not embedded blindly in code.

Instead, mappings and schemas live in:

`json_configs/contracts/`

Examples:

- Account mappings  
- Financial facts  
- Routing contracts  

This separation enables:

- Faster policy updates  
- Reduced code churn  
- Auditable rule changes  

> Contracts represent **organizational memory**, not just configuration.

## State Layer

`json_configs/state/fx.json`


Holds slowly changing financial context (e.g., FX rates).

This is treated as **controlled mutable state**, distinct from static contracts.

## I/O Boundary

Filesystem access is abstracted via:

`utils/filesystem.py`


Why this matters:

If storage changes later (cloud / object store), transformation logic remains untouched.

This preserves a critical invariant:

> **Business logic should not know where data lives.**

## Current Tradeoffs (Explicit)

This architecture intentionally accepts several temporary constraints:

- No workflow orchestrator yet  
- Limited test coverage  
- Logging not fully structured  
- Execution remains script-driven  

These are acknowledged — not accidental.

The current priority is **structural extraction without operational regression.**

## Architectural Trajectory

This decomposition unlocks several future paths:

- Distributed execution (Spark / Ray)
- Formal data contracts
- Test harness around business logic
- Pipeline orchestration
- Cloud-native storage
- Observability

Importantly:

> The hardest step is not scaling.

> The hardest step is **escaping the monolith without breaking reality.**

That step is now underway.

## Engineering Philosophy Behind This Refactor

This work reflects a core belief:

> Systems should evolve through controlled structural improvements rather than episodic rewrites.

Rewrites reset knowledge.

Evolution compounds it.

The goal is not architectural purity.

The goal is **durable clarity under real-world constraints.**

## Security & Data Policy

This repository intentionally contains **code only**.

All organization-specific materials are excluded, including (but not limited to):
- `json_configs/io/*` (paths, storage routing, environment-specific I/O)
- `json_configs/contracts/*` (account mappings, business rules, facts)
- any real datasets, report outputs, or customer/company identifiers

These inputs are treated as **private deployment artifacts** and are injected at runtime via local configuration (not version control).

