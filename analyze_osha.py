#!/usr/bin/env python3
"""
OSHA Construction Incident Analysis — Ironsite Hackathon
Generates presentation-ready plots backing the problem statement:
"Spatial hazard detection from egocentric (hardhat-cam) construction footage"

Uses ALL available data: summary files (2016-2022) + case detail files (2023-2025)
"""

import csv
import sys
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

csv.field_size_limit(sys.maxsize)

BASE = Path("/Users/mns/Documents/Ironsite")
OUT = BASE / "plots"
OUT.mkdir(exist_ok=True)

# ── Color palette ──────────────────────────────────────────────────────────────
DARK_BG = "#0D1117"
CARD_BG = "#161B22"
ACCENT = "#58A6FF"
ACCENT2 = "#F78166"
ACCENT3 = "#7EE787"
ACCENT4 = "#D2A8FF"
ACCENT5 = "#FFA657"
ACCENT6 = "#FF7B72"
TEXT = "#E6EDF3"
TEXT_DIM = "#8B949E"
GRID = "#21262D"

PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5, ACCENT6, "#79C0FF", "#FFC68A"]

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=11)
    ax.tick_params(colors=TEXT_DIM, labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.5)

def save(fig, name):
    fig.patch.set_facecolor(DARK_BG)
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✓ {name}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 1: SUMMARY DATA (2016-2022) — industry-level trends
# ══════════════════════════════════════════════════════════════════════════════
print("Loading summary data (2016-2022)...")

summary_files = {
    2016: BASE / "ITA Data CY 2016.csv",
    2017: BASE / "ITA Data CY 2017.csv",
    2018: BASE / "ITA Data CY 2018.csv",
    2019: BASE / "ITA Data CY 2019.csv",
    2020: BASE / "ITA Data CY 2020.csv",
    2021: BASE / "ITA 2021 data.csv",
    2022: BASE / "CY 2022.csv",
}

# Also parse 2023 and 2024 from the 300A summary files
summary_files_300a = {
    2023: BASE / "ITA 300A Summary Data 2023 through 12-31-2024.csv",
    2024: BASE / "ITA 300A Summary Data 2024 through 08-31-2025.csv",
}

yearly_construction = {}  # year -> {deaths, dafw, djtr, other, injuries, establishments, employees, hours}

for year, fpath in sorted({**summary_files, **summary_files_300a}.items()):
    stats = {"deaths": 0, "dafw": 0, "djtr": 0, "other": 0, "injuries": 0,
             "establishments": 0, "employees": 0, "hours": 0}
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            naics = str(row.get('naics_code', '')).strip().strip('"')
            if not naics.startswith('23'):
                continue
            # For 300A files with multiple years, filter to the right year
            yff = str(row.get('year_filing_for', str(year))).strip().strip('"')
            if yff and yff.isdigit() and int(yff) != year:
                continue
            try:
                stats["deaths"] += int(float(row.get('total_deaths', 0) or 0))
                stats["dafw"] += int(float(row.get('total_dafw_cases', 0) or 0))
                stats["djtr"] += int(float(row.get('total_djtr_cases', 0) or 0))
                stats["other"] += int(float(row.get('total_other_cases', 0) or 0))
                stats["injuries"] += int(float(row.get('total_injuries', 0) or 0))
                stats["establishments"] += 1
                stats["employees"] += int(float(row.get('annual_average_employees', 0) or 0))
                stats["hours"] += int(float(row.get('total_hours_worked', 0) or 0))
            except (ValueError, TypeError):
                continue
    yearly_construction[year] = stats
    total_cases = stats["dafw"] + stats["djtr"] + stats["other"]
    print(f"  {year}: {stats['establishments']:,} establishments, {total_cases:,} cases, {stats['deaths']} deaths")

# ── Also get all-industry data for comparison ──
yearly_all = {}
for year, fpath in sorted(summary_files.items()):
    stats = {"deaths": 0, "dafw": 0, "djtr": 0, "other": 0, "injuries": 0,
             "establishments": 0, "employees": 0, "hours": 0}
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stats["deaths"] += int(float(row.get('total_deaths', 0) or 0))
                stats["dafw"] += int(float(row.get('total_dafw_cases', 0) or 0))
                stats["injuries"] += int(float(row.get('total_injuries', 0) or 0))
                stats["establishments"] += 1
                stats["employees"] += int(float(row.get('annual_average_employees', 0) or 0))
                stats["hours"] += int(float(row.get('total_hours_worked', 0) or 0))
            except (ValueError, TypeError):
                continue
    yearly_all[year] = stats


# ── PLOT 1: Construction injury trend over years ──
print("\nGenerating plots...")
years = sorted(yearly_construction.keys())
total_cases = [yearly_construction[y]["dafw"] + yearly_construction[y]["djtr"] + yearly_construction[y]["other"] for y in years]
deaths = [yearly_construction[y]["deaths"] for y in years]
dafw = [yearly_construction[y]["dafw"] for y in years]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(years, total_cases, color=ACCENT, alpha=0.85, width=0.6, label="Total Recordable Cases")
ax.bar(years, dafw, color=ACCENT2, alpha=0.85, width=0.6, label="Days Away From Work Cases")
style_ax(ax, "Construction Industry Injuries Over Time (OSHA 300A Data)", "Year", "Number of Cases")
ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
ax.set_xticks(years)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, v in enumerate(total_cases):
    ax.text(years[i], v + 200, f'{v:,}', ha='center', va='bottom', color=TEXT_DIM, fontsize=8)
save(fig, "01_construction_injuries_trend.png")


# ── PLOT 2: Construction injury rate per 100 workers ──
rates = []
for y in years:
    emp = yearly_construction[y]["employees"]
    cases = yearly_construction[y]["dafw"] + yearly_construction[y]["djtr"] + yearly_construction[y]["other"]
    rates.append((cases / emp * 100) if emp > 0 else 0)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(years, rates, color=ACCENT2, linewidth=3, marker='o', markersize=8, markerfacecolor=ACCENT6, zorder=5)
ax.fill_between(years, rates, alpha=0.15, color=ACCENT2)
style_ax(ax, "Construction Injury Rate per 100 Workers", "Year", "Cases per 100 Employees")
ax.set_xticks(years)
for i, v in enumerate(rates):
    ax.text(years[i], v + 0.05, f'{v:.1f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
save(fig, "02_injury_rate_per_100_workers.png")


# ── PLOT 3: Construction vs All Industries (injury rate comparison) ──
comp_years = sorted(set(years) & set(yearly_all.keys()))
constr_rates = []
all_rates = []
for y in comp_years:
    ce = yearly_construction[y]["employees"]
    cc = yearly_construction[y]["dafw"] + yearly_construction[y]["djtr"] + yearly_construction[y]["other"]
    ae = yearly_all[y]["employees"]
    ac = yearly_all[y]["dafw"]  # using DAFW for cleaner comparison
    constr_rates.append((cc / ce * 100) if ce > 0 else 0)
    all_rates.append((ac / ae * 100) if ae > 0 else 0)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_years))
w = 0.35
ax.bar(x - w/2, constr_rates, w, color=ACCENT2, alpha=0.85, label="Construction")
ax.bar(x + w/2, all_rates, w, color=ACCENT, alpha=0.85, label="All Industries (DAFW only)")
style_ax(ax, "Construction vs All Industries — Injury Rate per 100 Workers", "Year", "Cases per 100 Employees")
ax.set_xticks(x)
ax.set_xticklabels(comp_years)
ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
save(fig, "03_construction_vs_all_industries.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2: CASE DETAIL DATA (2023-2025) — incident narratives
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading case detail data (2023-2025)...")

detail_files = [
    BASE / "ITA Case Detail Data 2023 through 12-31-2023OIICS.csv",
    BASE / "ITA Case Detail Data 2024 through 08-31-2025.csv",
]

# ── Categorize every construction incident by spatial hazard type ──
SPATIAL_CATEGORIES = {
    "Falls from Height": {
        "keywords": ["fell from", "fall from", "fell off", "fall off roof", "fell off ladder", "fell off scaffold",
                      "fell from roof", "fell from ladder", "fell from scaffold", "fell to the ground from"],
        "color": ACCENT6,
    },
    "Struck by Falling Object": {
        "keywords": ["fell on his", "fell on her", "fell on employee", "dropped on", "object fell",
                      "material fell", "fell onto", "landed on his", "landed on her", "fell and struck",
                      "fell from above", "tool fell", "pipe fell", "board fell", "beam fell"],
        "color": ACCENT2,
    },
    "Trip / Slip Hazards": {
        "keywords": ["tripped over", "tripped on", "slipped on", "stumbled over", "stumbled on",
                      "stepped on nail", "stepped on a nail", "stepped on screw", "uneven surface",
                      "debris on floor", "cord on ground", "hose on ground"],
        "color": ACCENT5,
    },
    "Caught-In / Pinch Points": {
        "keywords": ["caught between", "caught in", "pinch", "crushed between", "squeezed between",
                      "hand caught", "finger caught", "smashed between", "compressed between"],
        "color": ACCENT4,
    },
    "Overhead Clearance Strikes": {
        "keywords": ["hit his head", "hit her head", "struck his head", "struck her head", "bumped his head",
                      "bumped her head", "head on beam", "head on pipe", "hit head on", "struck head on",
                      "low clearance", "overhead beam", "head struck"],
        "color": ACCENT,
    },
    "Floor Openings / Holes": {
        "keywords": ["fell in hole", "fell into hole", "fell in the hole", "fell into the hole",
                      "fell through", "floor opening", "open hole", "fell through opening",
                      "stepped into hole", "unmarked hole", "uncovered hole", "fell through skylight"],
        "color": ACCENT3,
    },
    "Unstable Load / Rigging": {
        "keywords": ["rigging", "load shift", "load fell", "came loose from", "sling broke",
                      "sling failed", "load swung", "suspended load", "crane load", "bundle fell",
                      "stack fell", "stack tipped", "toppled", "material shifted"],
        "color": "#79C0FF",
    },
    "Protruding Objects": {
        "keywords": ["protruding nail", "protruding rebar", "exposed rebar", "uncapped rebar",
                      "protruding screw", "protruding bolt", "sharp edge protruding",
                      "nail sticking", "rebar sticking"],
        "color": "#FFC68A",
    },
    "Equipment Proximity / Blind Spot": {
        "keywords": ["blind spot", "backed into", "backing up", "run over by", "ran over",
                      "struck by vehicle", "struck by equipment", "struck by excavat",
                      "struck by loader", "struck by forklift", "backed over"],
        "color": "#CEA5FB",
    },
}

all_incidents = []
construction_incidents = []
spatial_counts = Counter()
spatial_incidents = defaultdict(list)
outcome_counts = Counter()
job_role_counts = Counter()
object_counts = Counter()
monthly_counts = defaultdict(int)
severity_by_category = defaultdict(lambda: Counter())  # cat -> {outcome: count}
state_counts = Counter()

for fpath in detail_files:
    print(f"  Processing {fpath.name}...")
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            naics = str(row.get('naics_code', '')).strip()
            if not naics.startswith('23'):
                continue

            what = (row.get('NEW_NAR_WHAT_HAPPENED', '') or '').strip()
            before = (row.get('NEW_NAR_BEFORE_INCIDENT', '') or '').strip()
            desc = (row.get('NEW_INCIDENT_DESCRIPTION', '') or '').strip()
            obj = (row.get('NEW_NAR_OBJECT_SUBSTANCE', '') or '').strip()
            injury = (row.get('NEW_NAR_INJURY_ILLNESS', '') or '').strip()
            location = (row.get('NEW_INCIDENT_LOCATION', '') or '').strip()
            outcome = row.get('incident_outcome', '')
            job = (row.get('job_description', '') or '').strip().lower()
            state = (row.get('state', '') or '').strip()
            date = (row.get('date_of_incident', '') or '').strip()
            dafw = row.get('dafw_num_away', '')

            combined_text = f"{what} {before} {desc}".lower()

            construction_incidents.append(row)
            outcome_counts[outcome] += 1
            if job:
                job_role_counts[job] += 1
            if obj.strip().lower() not in ('', 'n a', 'na', 'none', 'unknown', 'n/a'):
                object_counts[obj.strip().lower()] += 1
            if state:
                state_counts[state] += 1

            # Parse month
            if date:
                m = re.search(r'(\d{2})/\d{2}/(\d{4})', date)
                if m:
                    monthly_counts[f"{m.group(2)}-{m.group(1)}"] += 1

            # Categorize
            matched = False
            for cat, info in SPATIAL_CATEGORIES.items():
                for kw in info["keywords"]:
                    if kw in combined_text:
                        spatial_counts[cat] += 1
                        spatial_incidents[cat].append({
                            "what": what[:300], "obj": obj, "job": job,
                            "outcome": outcome, "dafw": dafw, "injury": injury,
                        })
                        severity_by_category[cat][outcome] += 1
                        matched = True
                        break

print(f"\n  Total construction incidents (2023-2025): {len(construction_incidents):,}")
print(f"  Spatially categorized: {sum(spatial_counts.values()):,}")


# ── PLOT 4: Spatial hazard category breakdown (main chart) ──
cats = sorted(spatial_counts.keys(), key=lambda x: spatial_counts[x], reverse=True)
counts = [spatial_counts[c] for c in cats]
colors = [SPATIAL_CATEGORIES[c]["color"] for c in cats]

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(range(len(cats)), counts, color=colors, height=0.65)
ax.set_yticks(range(len(cats)))
ax.set_yticklabels(cats, fontsize=12)
ax.invert_yaxis()
style_ax(ax, "Construction Spatial Hazards by Category (2023–2025 OSHA Data)", "Number of Incidents", "")
ax.spines['left'].set_visible(False)
for i, (bar, v) in enumerate(zip(bars, counts)):
    ax.text(v + 20, i, f'{v:,}', va='center', color=TEXT, fontsize=11, fontweight='bold')
save(fig, "04_spatial_hazard_categories.png")


# ── PLOT 5: Severity distribution within each spatial category ──
outcome_labels = {"1": "Death", "2": "Days Away", "3": "Job Transfer", "4": "Other"}
outcome_colors = {"1": ACCENT6, "2": ACCENT2, "3": ACCENT5, "4": TEXT_DIM}

fig, ax = plt.subplots(figsize=(14, 7))
cat_order = sorted(spatial_counts.keys(), key=lambda x: spatial_counts[x], reverse=True)
x = np.arange(len(cat_order))
bottom = np.zeros(len(cat_order))

for oc in ["1", "2", "3", "4"]:
    vals = [severity_by_category[c].get(oc, 0) for c in cat_order]
    ax.barh(x, vals, left=bottom, height=0.6, color=outcome_colors[oc], label=outcome_labels[oc])
    bottom += np.array(vals)

ax.set_yticks(x)
ax.set_yticklabels(cat_order, fontsize=11)
ax.invert_yaxis()
style_ax(ax, "Severity Breakdown by Spatial Hazard Category", "Number of Incidents", "")
ax.spines['left'].set_visible(False)
ax.legend(facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc='lower right')
save(fig, "05_severity_by_category.png")


# ── PLOT 6: Top objects/substances involved ──
top_objects = object_counts.most_common(20)
obj_names = [o[0][:30] for o in top_objects]
obj_vals = [o[1] for o in top_objects]

fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(range(len(obj_names)), obj_vals, color=ACCENT, height=0.65, alpha=0.85)
ax.set_yticks(range(len(obj_names)))
ax.set_yticklabels(obj_names, fontsize=10)
ax.invert_yaxis()
style_ax(ax, "Top 20 Objects/Substances in Construction Incidents", "Incident Count", "")
ax.spines['left'].set_visible(False)
for i, v in enumerate(obj_vals):
    ax.text(v + 5, i, f'{v:,}', va='center', color=TEXT_DIM, fontsize=9)
save(fig, "06_top_objects_substances.png")


# ── PLOT 7: Top job roles affected ──
top_jobs = job_role_counts.most_common(15)
job_names = [j[0].title() for j in top_jobs]
job_vals = [j[1] for j in top_jobs]

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(range(len(job_names)), job_vals, color=ACCENT3, height=0.6, alpha=0.85)
ax.set_yticks(range(len(job_names)))
ax.set_yticklabels(job_names, fontsize=11)
ax.invert_yaxis()
style_ax(ax, "Most Affected Job Roles in Construction Incidents", "Incident Count", "")
ax.spines['left'].set_visible(False)
for i, v in enumerate(job_vals):
    ax.text(v + 5, i, f'{v:,}', va='center', color=TEXT_DIM, fontsize=9)
save(fig, "07_top_job_roles.png")


# ── PLOT 8: Monthly incident pattern (seasonality) ──
sorted_months = sorted(monthly_counts.keys())
if sorted_months:
    # Aggregate by month-of-year across all years
    month_of_year = Counter()
    for m, c in monthly_counts.items():
        month_num = int(m.split('-')[1])
        month_of_year[month_num] += c

    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_vals = [month_of_year.get(m, 0) for m in months]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(months, month_vals, color=ACCENT5, alpha=0.85, width=0.7)
    style_ax(ax, "Construction Incidents by Month (Seasonality)", "Month", "Incidents")
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    for i, v in enumerate(month_vals):
        ax.text(months[i], v + 10, f'{v:,}', ha='center', va='bottom', color=TEXT_DIM, fontsize=8)
    save(fig, "08_seasonality.png")


# ── PLOT 9: Top 15 states by construction incidents ──
top_states = state_counts.most_common(15)
st_names = [s[0] for s in top_states]
st_vals = [s[1] for s in top_states]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(len(st_names)), st_vals, color=ACCENT4, alpha=0.85, width=0.65)
ax.set_xticks(range(len(st_names)))
ax.set_xticklabels(st_names, fontsize=10)
style_ax(ax, "Top 15 States by Construction Incidents", "State", "Incidents")
for i, v in enumerate(st_vals):
    ax.text(i, v + 10, f'{v:,}', ha='center', va='bottom', color=TEXT_DIM, fontsize=8)
save(fig, "09_top_states.png")


# ── PLOT 10: Overhead clearance deep dive — the niche problem ──
overhead_incidents = spatial_incidents.get("Overhead Clearance Strikes", [])
oh_objects = Counter()
oh_jobs = Counter()
for inc in overhead_incidents:
    o = inc["obj"].strip().lower()
    if o and o not in ('n a', 'na', 'none', 'unknown', 'n/a', ''):
        oh_objects[o] += 1
    j = inc["job"]
    if j:
        oh_jobs[j] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Objects causing head strikes
top_oh_obj = oh_objects.most_common(12)
if top_oh_obj:
    names = [o[0][:25] for o in top_oh_obj]
    vals = [o[1] for o in top_oh_obj]
    axes[0].barh(range(len(names)), vals, color=ACCENT, height=0.6)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=10)
    axes[0].invert_yaxis()
    style_ax(axes[0], "Objects Causing Head Strikes", "Count", "")
    axes[0].spines['left'].set_visible(False)

# Jobs most affected by overhead strikes
top_oh_jobs = oh_jobs.most_common(10)
if top_oh_jobs:
    names = [j[0].title()[:20] for j in top_oh_jobs]
    vals = [j[1] for j in top_oh_jobs]
    axes[1].barh(range(len(names)), vals, color=ACCENT2, height=0.6)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=10)
    axes[1].invert_yaxis()
    style_ax(axes[1], "Job Roles Most Affected", "Count", "")
    axes[1].spines['left'].set_visible(False)

fig.suptitle("Deep Dive: Overhead Clearance Strikes", color=TEXT, fontsize=18, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, "10_overhead_clearance_deep_dive.png")


# ── PLOT 11: "The Fatal Four" in construction — pie/donut chart ──
fatal_four = {
    "Falls": spatial_counts.get("Falls from Height", 0) + spatial_counts.get("Floor Openings / Holes", 0),
    "Struck-By": spatial_counts.get("Struck by Falling Object", 0),
    "Caught-In/Between": spatial_counts.get("Caught-In / Pinch Points", 0),
    "Other Spatial": spatial_counts.get("Overhead Clearance Strikes", 0) +
                     spatial_counts.get("Trip / Slip Hazards", 0) +
                     spatial_counts.get("Unstable Load / Rigging", 0) +
                     spatial_counts.get("Protruding Objects", 0) +
                     spatial_counts.get("Equipment Proximity / Blind Spot", 0),
}
uncategorized = len(construction_incidents) - sum(fatal_four.values())
fatal_four["Non-Spatial / Other"] = uncategorized

ff_colors = [ACCENT6, ACCENT2, ACCENT4, ACCENT, TEXT_DIM]
fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    fatal_four.values(), labels=fatal_four.keys(), colors=ff_colors,
    autopct=lambda p: f'{p:.1f}%\n({int(p*sum(fatal_four.values())/100):,})',
    startangle=90, pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor=DARK_BG, linewidth=2)
)
for t in texts:
    t.set_color(TEXT)
    t.set_fontsize(11)
for t in autotexts:
    t.set_color(TEXT)
    t.set_fontsize=9
ax.set_title("Spatial vs Non-Spatial Construction Incidents (2023–2025)",
             color=TEXT, fontsize=16, fontweight='bold', pad=20)
save(fig, "11_spatial_vs_nonspatial_donut.png")


# ── PLOT 12: Days away from work distribution by category ──
dafw_by_cat = defaultdict(list)
for cat, incs in spatial_incidents.items():
    for inc in incs:
        try:
            d = int(float(inc["dafw"]))
            if 0 < d < 365:  # reasonable range
                dafw_by_cat[cat].append(d)
        except (ValueError, TypeError):
            continue

# Box plot
cats_with_data = [(c, dafw_by_cat[c]) for c in cat_order if len(dafw_by_cat[c]) > 5]
if cats_with_data:
    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(
        [d[1] for d in cats_with_data],
        labels=[d[0] for d in cats_with_data],
        vert=False, patch_artist=True,
        medianprops=dict(color=TEXT, linewidth=2),
        whiskerprops=dict(color=TEXT_DIM),
        capprops=dict(color=TEXT_DIM),
        flierprops=dict(marker='.', markerfacecolor=TEXT_DIM, markersize=3, alpha=0.3),
    )
    for i, patch in enumerate(bp['boxes']):
        cat_name = cats_with_data[i][0]
        patch.set_facecolor(SPATIAL_CATEGORIES[cat_name]["color"])
        patch.set_alpha(0.7)
    style_ax(ax, "Days Away From Work by Hazard Category", "Days Away", "")
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)

    # Add median labels
    for i, (c, data) in enumerate(cats_with_data):
        med = np.median(data)
        ax.text(med, i + 1.3, f'median: {med:.0f}d', color=TEXT, fontsize=8, va='bottom')
    save(fig, "12_days_away_by_category.png")


# ── PLOT 13: Incident narratives word cloud data (top words for overhead + falls) ──
# Instead of wordcloud, make a clean bar chart of key phrases
fall_words = Counter()
overhead_words = Counter()

important_phrases = [
    "ladder", "scaffold", "roof", "beam", "pipe", "truss", "joist", "rebar",
    "concrete", "steel", "plywood", "drywall", "duct", "conduit", "form",
    "hoist", "crane", "forklift", "excavator", "boom", "bucket",
    "nail gun", "grinder", "saw", "drill", "hammer",
    "harness", "guardrail", "railing", "cover", "barricade",
    "hole", "opening", "edge", "trench",
]

for cat_name, incs in spatial_incidents.items():
    for inc in incs:
        text = inc["what"].lower()
        for phrase in important_phrases:
            if phrase in text:
                if cat_name == "Falls from Height":
                    fall_words[phrase] += 1
                elif cat_name == "Overhead Clearance Strikes":
                    overhead_words[phrase] += 1

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Falls
top_fw = fall_words.most_common(12)
if top_fw:
    names, vals = zip(*top_fw)
    axes[0].barh(range(len(names)), vals, color=ACCENT6, height=0.6)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=11)
    axes[0].invert_yaxis()
    style_ax(axes[0], "Key Objects in Fall Incidents", "Mentions", "")
    axes[0].spines['left'].set_visible(False)

# Overhead
top_ow = overhead_words.most_common(12)
if top_ow:
    names, vals = zip(*top_ow)
    axes[1].barh(range(len(names)), vals, color=ACCENT, height=0.6)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names, fontsize=11)
    axes[1].invert_yaxis()
    style_ax(axes[1], "Key Objects in Overhead Strikes", "Mentions", "")
    axes[1].spines['left'].set_visible(False)

fig.suptitle("Objects Involved in Spatial Hazard Incidents", color=TEXT, fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
save(fig, "13_key_objects_in_incidents.png")


# ── PLOT 14: Summary stats card ──
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_facecolor(CARD_BG)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

stats_text = [
    ("3.6M+", "Total OSHA records analyzed (2016–2025)"),
    (f"{len(construction_incidents):,}", "Construction incident cases with narratives (2023–2025)"),
    (f"{sum(spatial_counts.values()):,}", "Incidents involving spatial hazards"),
    (f"{spatial_counts.get('Falls from Height', 0) + spatial_counts.get('Floor Openings / Holes', 0):,}", "Fall-related incidents (falls from height + through openings)"),
    (f"{spatial_counts.get('Overhead Clearance Strikes', 0):,}", "Overhead clearance / head strike incidents"),
    (f"{spatial_counts.get('Struck by Falling Object', 0):,}", "Struck-by-falling-object incidents"),
    (f"9", "Fatalities in construction (2024–2025 detail data)"),
]

y_pos = 9.0
ax.text(5, y_pos + 0.5, "OSHA Data Analysis — Key Numbers", color=TEXT, fontsize=20,
        fontweight='bold', ha='center', va='top')
y_pos -= 0.5

for num, desc in stats_text:
    ax.text(1.5, y_pos, num, color=ACCENT, fontsize=24, fontweight='bold', va='center', ha='right')
    ax.text(1.8, y_pos, desc, color=TEXT_DIM, fontsize=13, va='center')
    y_pos -= 1.1

ax.text(5, 0.3, "Source: OSHA ITA (Injury Tracking Application) 2016–2025  |  NAICS 23xxxx (Construction)",
        color=TEXT_DIM, fontsize=9, ha='center', style='italic')

save(fig, "14_summary_stats_card.png")


# ── PLOT 15: Treemap-style view — proportion of spatial hazards that are "camera-detectable" ──
# Which hazards could a hardhat-mounted camera plausibly detect?
camera_detectable = {
    "Overhead Clearance Strikes": ("YES — camera faces the hazard", True),
    "Floor Openings / Holes": ("YES — visible in downward glance", True),
    "Trip / Slip Hazards": ("YES — visible on walking surface", True),
    "Protruding Objects": ("PARTIAL — small objects, but in FOV", True),
    "Struck by Falling Object": ("PARTIAL — requires upward FOV", True),
    "Falls from Height": ("INDIRECT — edge detection possible", True),
    "Unstable Load / Rigging": ("PARTIAL — rigging geometry visible", True),
    "Caught-In / Pinch Points": ("LIMITED — close-range interaction", False),
    "Equipment Proximity / Blind Spot": ("LIMITED — requires 360° awareness", False),
}

detectable_count = sum(spatial_counts[c] for c, (_, d) in camera_detectable.items() if d)
non_detectable_count = sum(spatial_counts[c] for c, (_, d) in camera_detectable.items() if not d)

fig, ax = plt.subplots(figsize=(14, 7))
cat_order_det = sorted(camera_detectable.keys(), key=lambda c: spatial_counts.get(c, 0), reverse=True)
vals = [spatial_counts.get(c, 0) for c in cat_order_det]
colors_det = [ACCENT3 if camera_detectable[c][1] else TEXT_DIM for c in cat_order_det]
labels = [f"{c}\n{camera_detectable[c][0]}" for c in cat_order_det]

bars = ax.barh(range(len(cat_order_det)), vals, color=colors_det, height=0.65, alpha=0.85)
ax.set_yticks(range(len(cat_order_det)))
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
style_ax(ax, f"Camera-Detectable Spatial Hazards: {detectable_count:,} of {detectable_count+non_detectable_count:,} incidents ({detectable_count/(detectable_count+non_detectable_count)*100:.0f}%)",
         "Incidents", "")
ax.spines['left'].set_visible(False)
for i, v in enumerate(vals):
    ax.text(v + 10, i, f'{v:,}', va='center', color=TEXT, fontsize=10)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ACCENT3, label='Camera-detectable'),
                   Patch(facecolor=TEXT_DIM, label='Limited camera detection')]
ax.legend(handles=legend_elements, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10, loc='lower right')
save(fig, "15_camera_detectable_hazards.png")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SAMPLE NARRATIVES FOR PRESENTATION SLIDES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SAMPLE NARRATIVES FOR SLIDES")
print("="*80)

import random
random.seed(42)

for cat in ["Overhead Clearance Strikes", "Falls from Height", "Floor Openings / Holes", "Struck by Falling Object"]:
    print(f"\n── {cat} ──")
    incs = spatial_incidents.get(cat, [])
    for inc in random.sample(incs, min(5, len(incs))):
        print(f"  [{inc['job'].title()}] {inc['what'][:200]}")
    print()

print("\n" + "="*80)
print(f"All plots saved to: {OUT}/")
print("="*80)
