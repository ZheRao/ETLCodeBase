# Overall Scope and Plan

## A) Hard invariants for Cattle Logic split

### A1) Data model invariants (must remain true everywhere)

1. **Universal Location → Pillar mapping remains deterministic**.  
After the change, *every* row still maps to exactly one pillar via location (including the new synthetic locations).

2. **Net-zero at original location for split locations**.  
For {Airdrie, Eddystone, Waldeck}: after split, the *original* location should net to **0** for any metric being split (amount, quantity, etc.), because you create:
    - +txn to `Loc (H)` (CowCalf)
    - +txn to `Loc (HD)` (Feedlot)
    - -offset txn to `Loc` (original)  
This is your “no double count” invariant.

3. **Conservation of value (global totals unchanged)**.  
For any time window and slice **above location** (pillar totals, company totals), the sums must match pre-split:
    - `sum(all locations, amount)` unchanged
    - `sum(by pillar)` unchanged (except **distribution** between CowCalf vs Feedlot changes)

4. **Idempotency invariant (non-negotiable)**.  
Re-running the split must not duplicate rows. You need a stable key strategy:
    - Original transaction row has a stable `txn_id` (or composite key)
    - Split rows deterministically derive new IDs (e.g., `txn_id|SPLIT|H` / `txn_id|SPLIT|HD` / `txn_id|SPLIT|OFFSET`)
    - If the split is rerun, it should upsert/replace, not append duplicates.

5. **Traceability / audit invariant**.  
Every generated row must reference the original via something like:
    - `source_txn_id`
    - `split_version`
    - `split_pct_cc`, `split_pct_fl`
    - `split_basis_date` (inventory snapshot date)

5. **Percentages are well-defined + bounded**.
    - `0 ≤ pct ≤ 1`
    - `pct_cc + pct_fl = 1` (within rounding tolerance)
    - If inventory is missing / zero, you must have a deterministic fallback (e.g., 100% to one pillar, or “no split applied + flag row”).

---
### A2) Business-rule invariants (your “contract”)

7. **Split basis is inventory proportion after unit normalization to head-days**.  
Your example implies:
    - Convert heads → head-days (heads * days in period or a defined conversion window)
    - Compute percent from head-days  
This implies you must specify the **time basis** (“days in period” vs “as-of snapshot”). If you don’t, you’ll get silent disagreements later.

8. **Only three locations are subject to split**.  
Everything else remains location→pillar as-is.

9. **New synthetic locations are the only place where CowCalf vs Feedlot separation happens**.  
Meaning: reports should not infer CowCalf/Feedlot from anything else (account, class, memo, etc.). It’s purely location-driven.

## B) Impacts on semantic model + dashboards (Executive + HR) - Budget also (next)

### B1) What WILL change (expected)

1. **Pillar totals for Cattle-CowCalf vs Cattle-Feedlot will shift** (for affected transactions).  
That’s the point.

2. **Location-level visuals will change**.
    - Original Airdrie/Eddystone/Waldeck should drop toward **0** for the split measures.
    - New locations `Airdrie (H)/(HD)` etc. will appear and carry the value.

---
### B2) What might silently break (high risk)

4. **Any dashboard filters or slicers with hard-coded location lists** (or custom sorting) will miss the new locations.  
Symptoms: “numbers don’t match” or “Airdrie disappeared.”

5. **Any measures that rely on “location count”** (like averages per location) will change because you added synthetic locations.

6. **Any mapping tables / relationships keyed by location name** will break if they assume a stable set of locations.

7. **HR / QBO Time alignment becomes inconsistent**.  
If exec dashboard compares labor (QBO Time) vs financials (QBO), you’ll now be comparing:
    - Financials: split CowCalf vs Feedlot
    - Labor: not split (or forced to one side)  
This creates misleading efficiency KPIs unless handled explicitly.

---
### B3) Dashboard-by-dashboard quick impact checklist

For each dashboard page, scan for:
- Any visual grouped by **Location** → will change
- Any visual grouped by **Pillar** → will change within cattle pillars
- Any measure using **distinctcount(Location)** → will change
- Any slicer using **static list** → will break
- Any cross-source KPI (QBO vs QBO Time) → **red flag**

Lean in — core growth: build a “Split Reconciliation” page that proves invariants (net zero old loc, totals conserved). This will save you from endless stakeholder debates later.

## C) Quarantine plan (dev parallel to daily pipeline, no pollution)

### C1) Folder + dataset isolation (cleanest approach)

Create a fully parallel namespace:

**Storage / Lake / Files**
- `.../gold/` (prod stays untouched)
- `.../gold__cattle_split_dev/` (ALL new outputs go here)
- `.../semantic_models/` (prod)
- `.../semantic_models__cattle_split_dev/` (copied + modified)

**Datasets / Tables naming**
- Prod: `fact_qbo_transactions`
- Dev: `fact_qbo_transactions__cattle_split_dev`
- Same for dims / bridges where needed.

**Power BI / Semantic**

- Duplicate PBIX / dataset:
    - `Exec Dashboard` → `Exec Dashboard (Cattle Split Dev)`
    - `HR Dashboard` → `HR Dashboard (Cattle Split Dev)`
- Point dev dashboards ONLY at dev semantic model.

---
### C2) Pipeline quarantine rules (non-negotiable)

1. **No shared tables**. Dev never writes into prod schemas/folders.
2. **No shared refresh triggers**. Dev refresh runs on manual / separate schedule.
3. **Immutable inputs snapshot for reproducibility**.  
For early days: freeze a snapshot of QBO transactions and inventory inputs so results don’t drift while you debug.

---
### C3) Deployment gating (how you merge safely later)

You want explicit promotion stages:
- **DEV**: assumed % split; static snapshot data
- **UAT**: CSV-driven %; refreshed on demand
- **PROD**: dynamic inventory-driven %; integrated with daily pipeline

Each gate requires passing the reconciliation suite (below).

## D) Implementation plan (tight + testable)

### Step 1 — Split algorithm with assumed percentage

Deliverables:
- Function that takes transactions and outputs 3-row expansion for impacted locations.
- Deterministic IDs + source tracing fields.

Must-pass tests:
- For each impacted original txn:` H + HD + OFFSET = original` (amount/qty)
- For each day: company total unchanged
- Old locations net to 0 after split (for split measures)

### Step 2 — Update location→pillar mapping

- Add six new locations:
    - `Airdrie (H)` → Cattle-CowCalf
    - `Airdrie (HD)` → Cattle-Feedlot
    - same for Eddystone, Waldeck

- Keep original mappings intact (even if now “should net to 0”) to avoid breaking historical references.

### Step 3 — CSV % ingestion (manual control)

- CSV contract must include:
    - location, pct_cc, pct_fl, effective_start_date, effective_end_date (or snapshot_date)

- Validation rules:
    - pct bounds, sum to 1, no overlaps, no gaps (or deterministic fallback)

### Step 4 — Dynamic % from most recent inventory

- Define “most recent” precisely:
    - latest inventory snapshot date available at runtime

- Define head-days conversion window (this is where fights happen later if undefined):
    - If you use “as-of snapshot”, heads → head-days needs a days assumption (e.g., next 30 days). Write it down.

Park — high future value: build a small “Split % history” dimension table so you can audit which % was used for each transaction date.

### Step 5 — Dev semantic model + dashboard clone
- Clone semantic model
- Add reconciliation measures + pages
- Validate the executive + HR dashboards side-by-side:
    - Prod totals vs Dev totals at company level must match (conservation)
    - Pillar distribution differences must be explainable + consistent

## E) QBO Time decision frame (don’t let this stay fuzzy)

You already see the core problem: **financials split; labor not**.

Here’s the blunt decision logic:

### Option 1: Force all QBO Time into one side (CowCalf OR Feedlot)
- Pros: simple; immediate; no fake precision
- Cons: makes labor efficiency KPIs wrong for the other side; exec comparisons become misleading

### Option 2: Split QBO Time hours by the same % as QBO transactions
- Pros: consistent story across finance + labor
- Cons: still an assumption; for Eddystone grain/cattle mix it’s structurally wrong

My recommended invariant-driven approach:
- For **Airdrie + Waldeck**: Option 2 is defensible (split by %).
- For **Eddystone**: you must **flag contamination** because QBO Time cannot separate grain vs cattle.
    - Practical: either exclude Eddystone labor from CowCalf/Feedlot KPIs, or park it into an “Unallocated (Eddystone)” bucket so you don’t lie with numbers.

Skip — low yield: pretending Eddystone labor cleanly maps to cattle split without any separability signal. That will create executive mistrust later.

## F) Reconciliation suite (the thing that makes this project “safe”)

Create 4 automated checks that run every dev refresh:
- **Conservation check**: totals match prod (by date, by company)
- **Net-zero check**: old locations net to 0 for split measures
- **Row lineage check**: every generated row has source_txn_id + split_version
- **Idempotency check**: rerun produces identical row counts + identical IDs

If these pass, you can confidently iterate without fear of hidden dashboard drift.


# Additional Considerations

## 1) “Original vs Synthetic” column is a core control lever

Do it. Make it a **string enum** (not boolean) and treat it as part of the model contract.

### Recommended design

Create **two** fields (this scales cleanly when you add more synthetic layers later):

- `record_layer` (string)
    - `"ORIGINAL"`
    - `"SYNTHETIC_SPLIT"`
    - `SYNTHETIC_OFFSET"`"
    - (future) `"SYNTHETIC_EBITDA_ALLOC"`, `"SYNTHETIC_UNCLASS_ALLOC"`, etc.

- `synthetic_group` (string, nullable)
    - For the cattle split, set to something stable like: `"CATTLE_SPLIT_V1"`
    - Helps you filter “just this synthetic family” without accidentally killing future families.

And keep the lineage fields:
- `source_txn_id`
- `synthetic_id` (your deterministic derived id)
- `split_version`, `pct_cc`, `pct_fl`, `basis_date`

### Power BI behavior you want

You’re basically giving BI an **on/off switch** and a **debug lens**:
- Default visuals include everything (`record_layer` all)
- A “Raw view” toggle page filter: `record_layer = "ORIGINAL"`
- A “Synthetic audit” page: `record_layer <> "ORIGINAL"`

**Invariant this creates**: You can always reproduce pre-split reporting by filtering to ORIGINAL.  
That’s huge for stakeholder trust.

One caution: if you ever build measures like “Total Amount”, make sure the default measure includes synthetic (because synthetic + offset preserves totals). 
But “Raw Total” should explicitly filter ORIGINAL.

## 2) Budget blast radius is real — treat it as a separate reconciliation track

Yes: your split touches **actuals** (QBO) and **labor** (QBO Time) and *now budget* (by location). 
Budget is typically *more brittle* because it’s planned at a specific grain (location) and stakeholders care a lot about continuity.

### Budget-specific invariants you should define now

1. **Budget totals at company level must remain unchanged** (obvious but must be tested).

2. **Budget location grain must be explicitly declared**:

    - Is budget intended to be reported at:
        - original locations only? (Airdrie stays Airdrie)
        - split locations? (Airdrie becomes Airdrie(H)/(HD))
        - both? (requires a bridge)

3. **Budget vs Actual comparability rule must be consistent**.  
If actuals move from Airdrie → Airdrie(H)/(HD) but budget stays at Airdrie, your variance will look “broken” unless you provide a mapping to compare apples-to-apples.

---
### Practical modeling options (choose later, but design now to support)

- **Option B1: Keep budget at original locations; roll synthetic actuals back up for variance**
    - Budget stays stable
    - Create a reporting “Location (Budget)” field where both Airdrie(H) and Airdrie(HD) map back to Airdrie
    - Best for continuity

- **Option B2: Split budget using same percentages**
    - Budget aligns to CowCalf/Feedlot pillars
    - But % is either assumed or dynamic—budget folks may hate moving targets unless you lock % for the budget year

- **Option B3: Maintain dual views**
- Provide both “Budget Location” and “Operational Location”
- Exec can view variance in either frame

Lean in — core growth: build a Location Bridge Dimension early:
- `location_operational` = {Airdrie, Airdrie(H), Airdrie(HD), …}
- `location_budget` = {Airdrie, …}
- `pillar_operational`  
This single bridge makes QBO/QBO Time/Budget reconciliation possible without hacks.

## 3) Quarantine just became mandatory, not optional

Your “double insurance” framing is correct, but tighten it:

### The two insurances
1. **Data-layer insurance**: `record_layer` lets you “turn off synthetic” and prove nothing was destroyed.
2. **Environment insurance**: dev datasets + dev semantic model + dev dashboards prevent contaminating prod pipelines and budget reporting.

### Add one more insurance (the missing one)

3. **Metric insurance**: reconciliation suite must include budget comparisons.  
Add tests like:
    - Company total budget unchanged (dev vs prod)
    - Budget variance at company level unchanged when record_layer="ORIGINAL" (this proves no accidental join/key breakage)
    - If you implement a budget bridge: aggregated budget location variances remain stable