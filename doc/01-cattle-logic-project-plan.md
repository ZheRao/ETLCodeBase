# Cattle Logic Split — Concise Implementation Specification

## 1. Objective

Separate cattle financial activity into two operational pillars:
- **Cattle-CowCalf**
- **Cattle-Feedlot**

For locations that contain mixed operations (Airdrie, Eddystone, Waldeck), transactions are separated using:
1. **Explicit class labels when available**
2. **Inventory-based percentage split when labels are missing or invalid**

This ensures accurate pillar attribution while preserving financial totals.

## 2. Core Invariants

These rules must always remain true.

### 2.1 Conservation of Value

Global totals must remain unchanged.
```text
sum(all transactions before split)
=
sum(all transactions after split)
```

### 2.2 Net-Zero at Original Location

For split locations:
```text
original_location + synthetic_rows = 0
```

Example:
```text
+ Airdrie (H)
+ Airdrie (HD)
- Airdrie
```

### 2.3 Traceability

Every generated row must reference the original transaction.

Required fields:
- record_layer
    - `ORIGINAL`
    - `SYNTHETIC_SPLIT`
    - `SYNTHETIC_OFFSET`
- classification_source
    - `CLASS_LABEL`
    - `CATTLE_INVENTORY_SPLIT`

## 3. Locations Subject to Split

Only the following locations are split:
- Airdrie
- Eddystone
- Waldeck

All other locations remain unchanged.

## 4. Classification Logic

Before applying the split algorithm, transactions are partitioned into two groups.

### 4.1 Dataset Partition
```text
transactions
    │
    ├─ labeled_transactions
    │
    └─ unlabeled_transactions
```

### 4.2 Labeled Transactions

Criteria:
```text
class contains {"Feedlot", "Cow/Calf"}
```

Processing:
```text
pillar = class mapping
record_layer = CLASS_DIRECT
classification_source = CLASS_LABEL
```

No synthetic rows are generated.

### 4.3 Unlabeled Transactions

Criteria:
```text
class is NULL
or {"Feedlot", "Cow/Calf"} not in class
```

Processing:

Inventory-based split is applied.

Generated rows:
```text
+ Loc (H)  → CowCalf
+ Loc (HD) → Feedlot
- Loc      → offset
```

Metadata:
```text
record_layer = SYNTHETIC_SPLIT / SYNTHETIC_OFFSET
classification_source = CATTLE_INVENTORY
```

## 5. Processing Pipeline

The pipeline executes in the following order.
```text
1. Load transactions

2. Filter split locations
      {Airdrie, Eddystone, Waldeck}

3. Partition dataset
      labeled_transactions
      unlabeled_transactions

4. Process labeled transactions
      map class → pillar

5. Process unlabeled transactions
      apply inventory split

6. Combine datasets
      labeled + synthetic

7. Write output dataset
```

## 6. Data Quality Rules

Only class values that includes one of these words are valid signals.
```text
Feedlot
Cow/Calf
```

All other values are treated as unlabeled.

Examples treated as unlabeled:
```text
"Corporate:Cattle (Header):*Cattle"
"Corporate:Grain (Header):Grain"
"Corporate:Produce (Header):*Produce"
NULL
```