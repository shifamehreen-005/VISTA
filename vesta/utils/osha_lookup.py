"""
OSHA Narrative Lookup Utility

Searches the OSHA ITA Case Detail CSVs to find the most relevant incident
narratives for a given hazard type. Used to inject real-world context into
VESTA's warnings.
"""

import os
from pathlib import Path
from functools import lru_cache

import pandas as pd

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "osha"

# Keywords that map hazard categories to OSHA narrative search terms
CATEGORY_KEYWORDS = {
    "Fall Hazard": ["fall", "fell", "height", "scaffold", "ladder", "roof", "edge", "hole", "trench"],
    "Struck-By Hazard": ["struck", "hit", "falling object", "forklift", "vehicle", "crane", "load"],
    "Caught-In/Between": ["caught", "crushed", "pinch", "trench collapse", "machinery", "rotating"],
    "Electrocution": ["electrocut", "shock", "electrical", "power line", "wiring", "voltage"],
    "Overhead Hazard": ["overhead", "clearance", "beam", "suspended", "crane", "boom"],
    "Trip/Slip Hazard": ["trip", "slip", "fell same level", "debris", "uneven", "wet"],
    "Chemical/Respiratory": ["chemical", "inhal", "dust", "fume", "respirat", "asbestos", "silica"],
    "Confined Space": ["confined space", "manhole", "tank", "pit", "oxygen", "ventilat"],
    "Fire/Explosion": ["fire", "explosion", "burn", "welding", "flammab", "gas cylinder"],
    "PPE Violation": ["hard hat", "helmet", "harness", "safety glass", "vest", "protective"],
}


@lru_cache(maxsize=1)
def _load_case_detail_data() -> pd.DataFrame | None:
    """
    Load the OSHA case detail CSVs (the ones with narrative text).
    Cached so we only load once.
    """
    case_files = sorted(DATA_DIR.glob("ITA Case Detail*.csv"))
    if not case_files:
        print(f"[OSHA] No case detail files found in {DATA_DIR}")
        return None

    frames = []
    for f in case_files:
        # Skip duplicate files
        if " 2.csv" in f.name:
            continue
        try:
            df = pd.read_csv(
                f,
                low_memory=False,
                encoding="latin-1",
                usecols=lambda c: c.lower() in [
                    "new_nar_what_happened", "new_nar_injury_illness",
                    "new_nar_object_substance", "new_incident_description",
                    "naics_code", "dafw_num_away", "job_description",
                    "date_of_incident", "date_of_death",
                ],
                dtype=str,
            )
            frames.append(df)
            print(f"[OSHA] Loaded {len(df)} records from {f.name}")
        except Exception as e:
            print(f"[OSHA] Warning: Could not load {f.name}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Normalize column names to lowercase
    combined.columns = [c.lower().strip() for c in combined.columns]

    return combined


def search_narratives(
    hazard_category: str,
    hazard_label: str = "",
    max_results: int = 3,
) -> list[dict]:
    """
    Search OSHA case detail narratives for incidents matching a hazard type.

    Args:
        hazard_category: OSHA category (e.g., "Fall Hazard")
        hazard_label: Specific hazard label for refined search
        max_results: Maximum number of narratives to return

    Returns:
        List of dicts with narrative text and metadata
    """
    df = _load_case_detail_data()
    if df is None:
        return []

    # Find the narrative column
    narrative_col = None
    for col in ["new_nar_what_happened", "new_incident_description"]:
        if col in df.columns:
            narrative_col = col
            break

    if narrative_col is None:
        return []

    # Filter to construction only (NAICS 23*)
    if "naics_code" in df.columns:
        df = df[df["naics_code"].fillna("").str.startswith("23")]

    # Build search keywords
    keywords = CATEGORY_KEYWORDS.get(hazard_category, [])
    if hazard_label:
        keywords.extend(hazard_label.lower().split())

    if not keywords:
        return []

    # Filter rows where narrative contains any keyword
    mask = pd.Series(False, index=df.index)
    narrative_lower = df[narrative_col].fillna("").str.lower()
    for kw in keywords[:5]:  # Limit to 5 keywords for performance
        mask = mask | narrative_lower.str.contains(kw, na=False)

    matches = df[mask].head(max_results * 3)  # Get extra, then pick best

    if matches.empty:
        return []

    # Score by keyword density and pick top results
    results = []
    for _, row in matches.iterrows():
        narrative = str(row.get(narrative_col, ""))
        if len(narrative) < 20:
            continue

        score = sum(1 for kw in keywords if kw in narrative.lower())
        results.append({
            "narrative": narrative[:500],
            "injury": str(row.get("new_nar_injury_illness", "Unknown")),
            "object": str(row.get("new_nar_object_substance", "Unknown")),
            "days_away": str(row.get("dafw_num_away", "N/A")),
            "job": str(row.get("job_description", "Unknown")),
            "score": score,
        })

    # Sort by relevance score and return top results
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:max_results]


def get_risk_context(hazard_category: str, hazard_label: str = "") -> str:
    """
    Get a formatted risk context string for a hazard, suitable for
    injection into the agent's response.
    """
    narratives = search_narratives(hazard_category, hazard_label)

    if not narratives:
        return f"OSHA Category: {hazard_category}. No matching incident narratives found."

    parts = [f"OSHA Category: {hazard_category}"]
    parts.append(f"Found {len(narratives)} similar incidents in OSHA records:")

    for i, n in enumerate(narratives, 1):
        parts.append(
            f"  Incident {i}: {n.get('job', 'Worker')} — "
            f"Injury: {n.get('injury', 'Unknown')}. "
            f"Days away from work: {n['days_away']}. "
            f"What happened: \"{n['narrative'][:200]}...\""
        )

    return "\n".join(parts)
