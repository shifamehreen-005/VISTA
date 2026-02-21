# OSHA Intelligence Engine — Build Spec
**Owner:** Person 3 (OSHA Brain)
**Estimated time:** 6-8 hours
**Priority:** HIGH — this is our data moat

---

## Context: What Is VESTA?

VESTA is our hackathon project — a **safety AI agent for construction sites**. It watches video from a worker's hardhat camera, detects hazards (holes, exposed wiring, forklifts, etc.), and remembers where they are even when the camera looks away.

The system has 4 modules:
1. **Optical Flow** — tracks how the camera is moving (rotation, translation)
2. **Gemini Flash Detector** — sends video frames to Google's Gemini AI to detect hazards
3. **Hazard Registry** — a persistent map of all hazards, updated with camera movement
4. **YOUR MODULE: OSHA Engine** — gives VESTA real-world data to back up its warnings

Without your module, VESTA can say *"There's a floor hole to your left."*
With your module, VESTA can say *"There's a floor hole to your left. In OSHA records, falls through floor openings caused an average of 32 days away from work. Example: A carpenter fell 12 feet through an unguarded opening, fracturing both legs."*

**Your module makes the difference between a tech demo and a hackathon winner.**

---

## What You're Building

A fast, queryable intelligence engine over **18,694 real OSHA construction incident records** that powers VESTA's hazard warnings with real-world data.

You are replacing the current placeholder (`vesta/utils/osha_lookup.py` — a basic keyword search over pandas that's slow and dumb) with a proper search engine backed by SQLite.

**You work entirely inside `osha_engine/`.** You do not need to touch any other code. We plug your module in during integration.

---

## Glossary (Quick Reference)

| Term | Meaning |
|------|---------|
| **OSHA** | Occupational Safety and Health Administration — US federal agency that tracks workplace injuries |
| **NAICS code** | Industry classification number. Construction = codes starting with `23` |
| **DAFW** | Days Away From Work — how many days an injured worker couldn't work |
| **DJTR** | Days of Job Transfer/Restriction — worker reassigned to lighter duties |
| **FTS5** | Full-Text Search 5 — SQLite's built-in fast text search engine. Works like Google search on your data |
| **OIICS** | Occupational Injury and Illness Classification System — standardized codes for injury types |
| **RAG** | Retrieval-Augmented Generation — fancy way of saying "search the database, then feed results to the AI" |
| **Hazard Registry** | VESTA's internal map of all detected hazards (a Python dictionary). Your engine enriches it with OSHA data |

---

## The Data You Have

All files are in `data/osha/` (relative to the project root). There are **two types**:

### Case Detail Files (YOUR PRIMARY DATA)
These have per-incident narratives — the gold.

| File | Rows | Encoding | Notes |
|------|------|----------|-------|
| `ITA Case Detail Data 2024 through 08-31-2025.csv` | ~686K | UTF-8 (mostly, has some latin-1 bytes) | Newer format |
| `ITA Case Detail Data 2023 through 12-31-2023OIICS.csv` | ~890K | Latin-1 | Has extra OIICS prediction columns |

**Construction incidents (NAICS code starts with `23`):** exactly **18,694** across both files.

**Skip the duplicate files** — anything with " 2.csv" in the name is an exact copy.

#### Key Columns You'll Use

| Column | What It Is | Example |
|--------|-----------|---------|
| `naics_code` | Industry code. **Filter to `23*` for construction.** | `236220` |
| `industry_description` | Human-readable industry | `"Commercial building construction"` |
| `job_description` | Worker's role | `"Mason"`, `"Electrician"`, `"Laborer"` |
| `date_of_incident` | When it happened | `"09/29/2024"` |
| `incident_outcome` | 1=Death, 2=Days away, 3=Transfer, 4=Other | `"2"` |
| `dafw_num_away` | Days Away From Work (integer) | `"77"` |
| `djtr_num_tr` | Days of Job Transfer/Restriction | `"0"` |
| `type_of_incident` | 1=Injury, 2=Skin disorder, 3=Respiratory, etc. | `"1"` |
| `NEW_NAR_WHAT_HAPPENED` | **THE NARRATIVE** — what happened in plain English | `"While unloading boxes employee fell..."` |
| `NEW_NAR_BEFORE_INCIDENT` | What worker was doing before | `"Unloading butane boxes from container"` |
| `NEW_INCIDENT_LOCATION` | Where on site | `"Warehouse unloading area"` |
| `NEW_NAR_INJURY_ILLNESS` | What injury occurred | `"Lip laceration and facial fracture"` |
| `NEW_NAR_OBJECT_SUBSTANCE` | Object/substance involved | `"Corner edge of butane boxes and concrete floor"` |
| `NEW_INCIDENT_DESCRIPTION` | Short summary | (often a shorter version of WHAT_HAPPENED) |
| `date_of_death` | Non-empty = fatality | (empty or date) |

The 2023 OIICS file also has these bonus columns (ML-predicted OIICS codes):
- `nature_title_pred` — Injury nature (e.g., "Fractures")
- `part_title_pred` — Body part (e.g., "Lower leg")
- `event_title_pred` — Event type (e.g., "Fall to lower level")
- `source_title_pred` — Source (e.g., "Floor opening")

### Summary Files (SECONDARY — for aggregate stats)
| File | What |
|------|------|
| `ITA 300A Summary Data 2024...csv` | Establishment-level annual totals (deaths, DAFW, etc.) |
| `ITA 300A Summary Data 2023...csv` | Same, older year |
| `ITA Data CY 2016-2022.csv` (7 files) | Older annual summaries |

Use these for aggregate stats ("X deaths in construction in 2023") but the Case Detail files are your main source.

---

## What You're Delivering

### File Structure

```
osha_engine/
├── __init__.py
├── indexer.py          ← Step 1: CSV → SQLite with FTS5
├── rag_retriever.py    ← Step 2: Hazard label → top-k narratives
├── risk_scorer.py      ← Step 3: Registry → site risk score
└── stats.py            ← Step 4: Pre-computed category stats
```

---

### Module 1: `indexer.py` — Build the Database

**What:** One-time script that reads both Case Detail CSVs and loads construction incidents into SQLite with full-text search.

**Why:** The CSVs are ~1GB combined. Pandas scanning them on every query takes seconds. SQLite FTS5 returns results in milliseconds.

**Requirements:**
- Read both Case Detail CSVs (handle encoding: try UTF-8, fall back to latin-1)
- Filter to **construction only**: `naics_code LIKE '23%'`
- Skip rows where `NEW_NAR_WHAT_HAPPENED` is empty or `[REDACTED]`
- Create a SQLite database at `data/osha/osha_construction.db`
- Schema:

```sql
CREATE TABLE incidents (
    id INTEGER PRIMARY KEY,
    naics_code TEXT,
    industry_description TEXT,
    job_description TEXT,
    date_of_incident TEXT,
    incident_outcome INTEGER,    -- 1=death, 2=DAFW, 3=transfer, 4=other
    days_away INTEGER,           -- from dafw_num_away
    days_transfer INTEGER,       -- from djtr_num_tr
    is_fatality BOOLEAN,         -- date_of_death is not empty
    what_happened TEXT,          -- NEW_NAR_WHAT_HAPPENED
    before_incident TEXT,        -- NEW_NAR_BEFORE_INCIDENT
    location TEXT,               -- NEW_INCIDENT_LOCATION
    injury_illness TEXT,         -- NEW_NAR_INJURY_ILLNESS
    object_substance TEXT,       -- NEW_NAR_OBJECT_SUBSTANCE
    description TEXT,            -- NEW_INCIDENT_DESCRIPTION
    -- OIICS predictions (from 2023 file, NULL for 2024)
    nature_pred TEXT,            -- nature_title_pred
    body_part_pred TEXT,         -- part_title_pred
    event_pred TEXT,             -- event_title_pred
    source_pred TEXT             -- source_title_pred
);

-- FTS5 virtual table for fast text search
CREATE VIRTUAL TABLE incidents_fts USING fts5(
    what_happened,
    before_incident,
    injury_illness,
    object_substance,
    description,
    content='incidents',
    content_rowid='id'
);
```

**Interface:**
```python
def build_index(force_rebuild: bool = False) -> str:
    """
    Build the SQLite index from CSVs.
    Returns path to the database file.
    Skips if database already exists (unless force_rebuild=True).
    """
```

**Test:** After indexing, `SELECT COUNT(*) FROM incidents` should return ~18,694 (± a few hundred after filtering empty narratives).

---

### Module 2: `rag_retriever.py` — Narrative Retrieval

**What:** Given a hazard label (e.g., "Floor Hole") and category (e.g., "Fall Hazard"), find the top-k most relevant real OSHA incident narratives.

**Requirements:**
- Use SQLite FTS5 `MATCH` queries for fast full-text search
- Expand the hazard label into search terms using the keyword mapping (already defined in `vesta/utils/osha_lookup.py` — `CATEGORY_KEYWORDS` dict — copy or import it)
- Rank results by: (a) FTS5 relevance score, (b) severity (fatalities first, then high-DAFW), (c) recency
- Return structured results with all narrative fields + metadata

**Interface (this is the contract with VESTA — do not change the signature):**
```python
from dataclasses import dataclass

@dataclass
class OSHAIncident:
    what_happened: str
    injury: str
    object_substance: str
    location: str
    days_away: int
    is_fatality: bool
    job_description: str
    date: str
    relevance_score: float

def retrieve_narratives(
    hazard_label: str,
    hazard_category: str = "",
    max_results: int = 3,
) -> list[OSHAIncident]:
    """
    Retrieve the most relevant OSHA incident narratives for a hazard.

    Args:
        hazard_label: What VESTA detected (e.g., "Floor Hole", "Exposed Wiring")
        hazard_category: OSHA category (e.g., "Fall Hazard", "Electrocution")
        max_results: Number of narratives to return

    Returns:
        List of OSHAIncident, sorted by relevance (most relevant first)
    """

def format_for_agent(incidents: list[OSHAIncident]) -> str:
    """
    Format retrieved incidents into a text block suitable for injection
    into the VESTA agent's context.
    """
```

**Example output for `retrieve_narratives("Floor Hole", "Fall Hazard")`:**
```
OSHAIncident(
    what_happened="Employee was walking across the 2nd floor when he stepped into
                   an uncovered floor opening and fell 12 feet to the ground level",
    injury="Fractured pelvis and both wrists",
    object_substance="Floor opening, concrete floor",
    location="2nd floor of building under construction",
    days_away=89,
    is_fatality=False,
    job_description="Carpenter",
    date="04/15/2024",
    relevance_score=8.5,
)
```

---

### Module 3: `risk_scorer.py` — Site Risk Score

**What:** Given the current set of hazards in VESTA's registry, compute a composite **Site Risk Score (0-100)** based on historical frequency and severity of those hazard types.

**Requirements:**

1. **Category Weight Table** — Pre-compute from the database:
   - For each OSHA hazard category, calculate:
     - `frequency`: How many construction incidents match this category
     - `avg_days_away`: Average days away from work
     - `fatality_rate`: % of incidents that were fatal
     - `severity_score`: Weighted combo of the above (you define the formula)

2. **Site Risk Score** — Given a list of active hazards from the registry:
   ```
   site_score = Σ (hazard_severity_score × hazard_confidence) / max_possible × 100
   ```
   Clamp to 0-100. Higher = more dangerous.

3. **Risk Level Buckets:**
   - 0-25: LOW (green)
   - 26-50: MODERATE (yellow)
   - 51-75: HIGH (orange)
   - 76-100: CRITICAL (red)

**Interface:**
```python
@dataclass
class SiteRiskReport:
    score: int                  # 0-100
    level: str                  # "LOW", "MODERATE", "HIGH", "CRITICAL"
    top_risks: list[dict]       # Top 3 hazards driving the score
    recommendation: str         # One-line safety recommendation

def compute_site_risk(hazards: list[dict]) -> SiteRiskReport:
    """
    Compute site risk from a list of hazard dicts.
    Each dict has: label, category, severity, confidence
    (This is what HazardEntry.to_dict() returns.)
    """
```

---

### Module 4: `stats.py` — Pre-Computed Stats

**What:** Pre-compute headline statistics that VESTA can cite instantly without querying the DB every time.

**Requirements:**

Build a JSON file or Python dict at index time with:

```python
STATS = {
    "total_construction_incidents": 18694,
    "total_fatalities": ???,          # COUNT where is_fatality = true
    "avg_days_away_all": ???,         # AVG(days_away) where days_away > 0
    "by_category": {
        "Fall Hazard": {
            "count": ???,
            "fatalities": ???,
            "avg_days_away": ???,
            "pct_of_total": ???,
            "top_objects": ["ladder", "scaffold", "roof"],  # from object_substance
            "top_jobs": ["Laborer", "Carpenter", "Roofer"],  # from job_description
            "example_stat": "Falls account for X% of construction fatalities with an average of Y days away from work."
        },
        # ... repeat for each category
    },
    "by_year": {
        "2023": {"incidents": ???, "fatalities": ???},
        "2024": {"incidents": ???, "fatalities": ???},
    }
}
```

**Interface:**
```python
def get_category_stats(category: str) -> dict:
    """Get pre-computed stats for a hazard category."""

def get_headline_stat(category: str) -> str:
    """
    One-liner stat suitable for the agent to cite.
    Example: "Falls from height: 4,231 construction incidents, 187 fatalities,
             avg 34 days away from work (OSHA 2023-2024)"
    """
```

---

## How Your Code Plugs Into VESTA

Your engine replaces the current `vesta/utils/osha_lookup.py`. The VESTA agent calls two functions:

1. **During detection** — When a hazard is added to the registry:
   ```python
   # In vesta/agent/vesta_agent.py, after adding a hazard:
   entry.osha_narrative = format_for_agent(
       retrieve_narratives(entry.label, entry.category)
   )
   ```

2. **During query** — When a user asks about hazards, the agent's `get_osha_context` tool calls:
   ```python
   def tool_get_osha_context(self, hazard_label, category=""):
       incidents = retrieve_narratives(hazard_label, category)
       return format_for_agent(incidents)
   ```

3. **For the site score** — Called by the UI or agent on demand:
   ```python
   report = compute_site_risk(registry.get_summary()["hazards"])
   ```

**You don't need to modify any VESTA core files.** Just build the `osha_engine/` package. We'll wire it in during integration.

---

## Build Order

| Step | Time | What | Test |
|------|------|------|------|
| 1 | 1-2h | `indexer.py` — CSV → SQLite | `SELECT COUNT(*) FROM incidents` ≈ 18,694 |
| 2 | 2-3h | `rag_retriever.py` — FTS5 search | `retrieve_narratives("ladder fall")` returns real incidents |
| 3 | 1-2h | `risk_scorer.py` — scoring formula | `compute_site_risk([{"category": "Fall Hazard", ...}])` returns 0-100 |
| 4 | 1h | `stats.py` — pre-compute headline numbers | `get_headline_stat("Fall Hazard")` returns a citation-ready string |

---

## Setup (Do This First)

```bash
# 1. Clone the repo and cd into it
git clone <repo-url>
cd Ironsite

# 2. Install dependencies (you only need pandas and sqlite3, which is built into Python)
pip install pandas

# 3. Verify you can see the data files
ls data/osha/
# You should see 13 CSV files. The two you care about:
#   ITA Case Detail Data 2024 through 08-31-2025.csv        (~392MB)
#   ITA Case Detail Data 2023 through 12-31-2023OIICS.csv   (~624MB)

# 4. Create your working directory
mkdir -p osha_engine
touch osha_engine/__init__.py

# 5. Build your indexer (Step 1), then test it:
python -m osha_engine.indexer
# Expected output: "Indexed ~18,000 construction incidents into data/osha/osha_construction.db"

# 6. Test the full pipeline after each step:
python -c "from osha_engine.rag_retriever import retrieve_narratives; print(retrieve_narratives('floor hole', 'Fall Hazard'))"
python -c "from osha_engine.stats import get_headline_stat; print(get_headline_stat('Fall Hazard'))"
python -c "from osha_engine.risk_scorer import compute_site_risk; print(compute_site_risk([{'label': 'Floor Hole', 'category': 'Fall Hazard', 'severity': 'high', 'confidence': 0.9}]))"
```

---

## Gotchas

1. **Encoding:** The 2023 CSV has Latin-1 encoded bytes. Use `open(f, encoding='latin-1')` or `pd.read_csv(f, encoding='latin-1')`. The 2024 CSV is mostly UTF-8 but also has stray Latin-1 bytes — safest to use `encoding='latin-1'` for both.

2. **Duplicate files:** `ITA Case Detail Data 2023 through 12-31-2023OIICS 2.csv` is an exact duplicate. Skip any file with ` 2.csv` in the name.

3. **Redacted narratives:** Some `NEW_NAR_WHAT_HAPPENED` values are just `[REDACTED]`. Skip these — they have no useful content.

4. **OIICS columns:** Only the 2023 file has `nature_title_pred`, `part_title_pred`, `event_title_pred`, `source_title_pred`. For 2024 rows, these will be NULL. Your retriever should still work without them.

5. **NAICS filter:** Construction = NAICS codes starting with `23`. This gives you the 18,694 construction-specific incidents. Index ALL of them, but you could add a filter parameter to `retrieve_narratives` to optionally include non-construction incidents too.

6. **`days_away` is a string in the CSV.** Cast to int, default to 0 if empty/non-numeric.

---

## How to Verify You're Done (Checklist)

Run each of these. If they all work, you're good.

```bash
# ✅ 1. Database exists and has the right row count
python -c "
import sqlite3
conn = sqlite3.connect('data/osha/osha_construction.db')
count = conn.execute('SELECT COUNT(*) FROM incidents').fetchone()[0]
print(f'Incidents: {count}')
assert 17000 < count < 20000, f'Expected ~18,694, got {count}'
print('PASS: Database looks good')
"

# ✅ 2. Full-text search returns real results
python -c "
from osha_engine.rag_retriever import retrieve_narratives
results = retrieve_narratives('floor hole', 'Fall Hazard')
assert len(results) > 0, 'No results returned'
assert results[0].days_away >= 0, 'days_away should be a number'
assert len(results[0].what_happened) > 20, 'Narrative too short'
print(f'Got {len(results)} results. Top result: {results[0].what_happened[:100]}...')
print('PASS: Retriever works')
"

# ✅ 3. Risk scorer returns a score 0-100
python -c "
from osha_engine.risk_scorer import compute_site_risk
report = compute_site_risk([
    {'label': 'Floor Hole', 'category': 'Fall Hazard', 'severity': 'critical', 'confidence': 0.9},
    {'label': 'Exposed Wire', 'category': 'Electrocution', 'severity': 'high', 'confidence': 0.7},
])
assert 0 <= report.score <= 100, f'Score {report.score} out of range'
assert report.level in ('LOW', 'MODERATE', 'HIGH', 'CRITICAL'), f'Bad level: {report.level}'
print(f'Site Risk: {report.score}/100 ({report.level})')
print(f'Recommendation: {report.recommendation}')
print('PASS: Risk scorer works')
"

# ✅ 4. Stats return a human-readable one-liner
python -c "
from osha_engine.stats import get_headline_stat
stat = get_headline_stat('Fall Hazard')
assert len(stat) > 30, 'Stat too short'
print(f'Headline: {stat}')
print('PASS: Stats work')
"
```

---

## What "Great" Looks Like

- **Search latency under 50ms** (SQLite FTS5 makes this easy)
- `retrieve_narratives("exposed wiring")` returns 3 real electrocution incidents with days-away data
- `get_headline_stat("Fall Hazard")` returns something like: *"4,231 fall incidents in construction (2023-2024). 187 fatalities. Average 34 days away from work. Most common: ladders, scaffolds, roofs."*
- `compute_site_risk(...)` returns a score that makes intuitive sense — a site with a fall hazard + electrical hazard scores higher than one with just debris on the floor
- The whole thing runs from a single ~15MB SQLite file — no pandas needed at query time, no API keys needed, works fully offline

---

## Questions?

If something in this doc is unclear, message the group chat before guessing. The interfaces (function signatures) are the contract — if you want to change them, check with the team first so we don't break integration.
