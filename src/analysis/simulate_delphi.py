#!/usr/bin/env python3
"""Synthesises a two-round Delphi expert-validation panel for the eight
thematic clusters in the keyword network. Output is *clearly labelled
simulation*; the instrument is the same one that would be administered to
real experts.

The simulation seeds eight panellists with realistic profile distributions,
draws Round-1 ratings from a beta distribution centred on a per-cluster
plausible consensus, runs aggregation, and re-draws outliers in Round-2
toward the panel median.

Outputs:
    data/processed/delphi_simulation.json
    supplementary/delphi_simulation_results.tex   (LaTeX fragment)
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Synthetic panel.
# ---------------------------------------------------------------------------

PANEL = [
    {"id": "E1", "profile": "Tenured CS-education professor",   "country": "USA",        "experience_yrs": 22},
    {"id": "E2", "profile": "Industry game-dev practitioner",   "country": "UK",         "experience_yrs": 14},
    {"id": "E3", "profile": "Mid-career game-design lecturer",  "country": "Germany",    "experience_yrs": 12},
    {"id": "E4", "profile": "Educational technology researcher","country": "Netherlands","experience_yrs": 18},
    {"id": "E5", "profile": "Curriculum designer (HE)",         "country": "Canada",     "experience_yrs": 16},
    {"id": "E6", "profile": "Early-career game-pedagogy PhD",   "country": "Brazil",     "experience_yrs":  4},
    {"id": "E7", "profile": "Serious-games industry consultant","country": "Finland",    "experience_yrs": 11},
    {"id": "E8", "profile": "K-12 CS-education trainer",        "country": "Australia",  "experience_yrs":  9},
]

# Eight clusters from data/processed/thematic_map.json (the actual clusters).
CLUSTERS = [
    {"id": "C1", "label": "Game development",                    "size": 54, "quadrant": "motor",                  "consensus": 4.6},
    {"id": "C3", "label": "Serious games / game-based learning", "size": 79, "quadrant": "motor",                  "consensus": 4.7},
    {"id": "C5", "label": "Education / video games (generic)",   "size": 64, "quadrant": "motor",                  "consensus": 4.0},
    {"id": "C7", "label": "Software engineering",                "size": 31, "quadrant": "basic_transversal",      "consensus": 3.6},
    {"id": "C2", "label": "Educational game / motivation",       "size": 33, "quadrant": "emerging_or_declining",  "consensus": 3.8},
    {"id": "C6", "label": "Virtual reality / Unity",             "size": 31, "quadrant": "emerging_or_declining",  "consensus": 3.9},
    {"id": "C0", "label": "Workshop / digital game AI",          "size":  2, "quadrant": "emerging_or_declining",  "consensus": 2.4},
    {"id": "C4", "label": "Higher education / game curriculum",  "size": 15, "quadrant": "niche",                  "consensus": 4.0},
]

# Ten Likert items from the instrument body. Items 5 and 6 are reverse-coded
# (high rating = MORE missing keyword / MORE over-merged), so consensus shifts
# slightly downward for items 5–6 on well-defined clusters.
ITEMS = list(range(1, 11))
REVERSE_CODED = {5, 6}

# ---------------------------------------------------------------------------
# Synthetic ratings.
# ---------------------------------------------------------------------------


def _sample_likert(centre: float, dispersion: float = 0.6) -> int:
    """Sample a 1-5 Likert response centred near ``centre``.

    Uses a discretised truncated normal to keep ratings realistic; the
    dispersion parameter controls inter-expert disagreement.
    """
    raw = rng.normal(loc=centre, scale=dispersion)
    return int(np.clip(round(raw), 1, 5))


def _round_one(cluster: Dict[str, Any]) -> Dict[str, List[int]]:
    """Per-expert ratings for the 10 items on one cluster."""
    base = cluster["consensus"]
    matrix: Dict[str, List[int]] = {}
    for expert in PANEL:
        # Slight per-expert bias: senior experts shade higher; juniors lower.
        bias = 0.15 if expert["experience_yrs"] > 15 else (-0.15 if expert["experience_yrs"] < 8 else 0.0)
        ratings = []
        for item in ITEMS:
            centre = base + bias
            if item in REVERSE_CODED:
                # Reverse-coded items: well-defined clusters score LOW (nothing missing / merged).
                centre = (6 - base) + (-bias)
            ratings.append(_sample_likert(centre))
        matrix[expert["id"]] = ratings
    return matrix


def _round_two(round_one: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """For each item, identify outliers (>= 2 from median) and have those
    panellists revise toward the median; ~30% of outliers actually move."""
    new_matrix = {eid: list(r) for eid, r in round_one.items()}
    n_items = len(ITEMS)
    for item_idx in range(n_items):
        col = np.array([round_one[e["id"]][item_idx] for e in PANEL])
        med = float(np.median(col))
        for i, e in enumerate(PANEL):
            if abs(col[i] - med) >= 2 and rng.random() < 0.30:
                # Revise halfway toward the median (rounded).
                new_val = int(round((col[i] + med) / 2))
                new_matrix[e["id"]][item_idx] = new_val
    return new_matrix


# ---------------------------------------------------------------------------
# Aggregation + reliability statistics.
# ---------------------------------------------------------------------------


def _aggregate(matrix: Dict[str, List[int]]) -> Dict[str, Any]:
    arr = np.array([matrix[e["id"]] for e in PANEL])  # 8 × 10
    out = {
        "median_per_item": [float(np.median(arr[:, i])) for i in range(arr.shape[1])],
        "iqr_per_item": [float(np.percentile(arr[:, i], 75) - np.percentile(arr[:, i], 25))
                          for i in range(arr.shape[1])],
        "pct_ge_4_per_item": [float(np.mean(arr[:, i] >= 4)) for i in range(arr.shape[1])],
    }
    return out


def _cohen_kappa_binary(a: List[int], b: List[int]) -> float:
    """Cohen's κ on binary classifications (treat rating ≥ 4 as positive)."""
    a_pos = np.array([x >= 4 for x in a], dtype=int)
    b_pos = np.array([x >= 4 for x in b], dtype=int)
    n = len(a_pos)
    p_a = float(a_pos.mean())
    p_b = float(b_pos.mean())
    p_o = float((a_pos == b_pos).mean())
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e >= 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def _krippendorff_alpha_ordinal(matrix: np.ndarray) -> float:
    """Krippendorff's α with an ordinal-distance metric.

    ``matrix`` is rater × item.
    Computes observed disagreement / expected disagreement under chance.
    """
    n_raters, n_items = matrix.shape
    flat_values = matrix.flatten()
    if len(flat_values) < 2:
        return 1.0

    # Observed disagreement: average squared rater-pair difference per item.
    Do_sum = 0.0
    n_pairs = 0
    for j in range(n_items):
        col = matrix[:, j]
        for i, k in combinations(range(n_raters), 2):
            Do_sum += (col[i] - col[k]) ** 2
            n_pairs += 1
    Do = Do_sum / max(n_pairs, 1)

    # Expected disagreement: average squared difference across all rating pairs.
    De_sum = 0.0
    pair_count = 0
    for i, j in combinations(range(len(flat_values)), 2):
        De_sum += (flat_values[i] - flat_values[j]) ** 2
        pair_count += 1
    De = De_sum / max(pair_count, 1)

    if De == 0:
        return 1.0
    return 1 - Do / De


def _bootstrap_alpha(matrix: np.ndarray, n_bootstrap: int = 500) -> Dict[str, float]:
    n_raters = matrix.shape[0]
    boot_alphas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_raters, size=n_raters)
        sample = matrix[idx, :]
        boot_alphas.append(_krippendorff_alpha_ordinal(sample))
    arr = np.array(boot_alphas)
    return {
        "alpha": float(_krippendorff_alpha_ordinal(matrix)),
        "ci_2_5": float(np.percentile(arr, 2.5)),
        "ci_97_5": float(np.percentile(arr, 97.5)),
    }


# ---------------------------------------------------------------------------
# Driver — simulate, aggregate, write JSON + LaTeX.
# ---------------------------------------------------------------------------


def simulate_all() -> Dict[str, Any]:
    out = {"panel": PANEL, "clusters": []}
    for cluster in CLUSTERS:
        r1 = _round_one(cluster)
        r2 = _round_two(r1)

        agg1 = _aggregate(r1)
        agg2 = _aggregate(r2)

        # Item 10 = global retain/revise judgement → Cohen's κ pairwise.
        kappas = []
        for a, b in combinations(PANEL, 2):
            ka = _cohen_kappa_binary(r2[a["id"]], r2[b["id"]])
            kappas.append(ka)
        kappa_mean = float(np.mean(kappas))
        kappa_min = float(np.min(kappas))
        kappa_max = float(np.max(kappas))

        # Krippendorff's α on the full 8 × 10 matrix.
        m_arr = np.array([r2[e["id"]] for e in PANEL])
        alpha_stats = _bootstrap_alpha(m_arr, n_bootstrap=300)

        # Retention determination — clear if median item 10 >= 4.
        retain = agg2["median_per_item"][9] >= 4

        out["clusters"].append({
            "cluster": cluster,
            "round1": r1,
            "round2": r2,
            "round1_aggregate": agg1,
            "round2_aggregate": agg2,
            "cohen_kappa_item10": {
                "mean": kappa_mean, "min": kappa_min, "max": kappa_max, "n_pairs": len(kappas),
            },
            "krippendorff_alpha": alpha_stats,
            "retention_decision": "RETAIN" if retain else "REVISE",
        })
    return out


def latex_for_results(sim: Dict[str, Any]) -> str:
    """Format the simulation output as a self-contained LaTeX fragment."""
    lines = [
        "% Auto-generated by simulate_delphi.py — DO NOT EDIT BY HAND.",
        "% This is SIMULATED Delphi data; see disclosure block.",
        "",
        "\\section*{Simulation: Round-1 ratings (synthetic panel)}",
        "",
        "We simulate a two-round Delphi study on the eight thematic clusters using a synthetic 8-expert panel. The simulation seeds Round-1 ratings from a per-cluster Likert-centred normal with experience-weighted bias, and Round-2 from outlier-driven moves toward the median. \\textbf{All numbers below are simulated; they exist to demonstrate the instrument's analytical output and to scaffold a real Delphi study, not to validate the clusters themselves.}",
        "",
        "\\subsection*{Synthetic panel composition}",
        "",
        "\\begin{tabular}{@{}lll r@{}}",
        "\\toprule",
        "\\textbf{ID} & \\textbf{Profile} & \\textbf{Country} & \\textbf{Yr exp.} \\\\",
        "\\midrule",
    ]
    for e in sim["panel"]:
        lines.append(f"{e['id']} & {e['profile']} & {e['country']} & {e['experience_yrs']} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "", ""])

    lines.extend([
        "\\subsection*{Per-cluster Round-2 aggregates and reliability}",
        "",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\begin{tabular}{@{}lll rrrl@{}}",
        "\\toprule",
        "\\textbf{Cluster} & \\textbf{Quadrant} & \\textbf{Item-10 med.} & \\textbf{IQR} & "
        "\\textbf{$\\bar{\\kappa}$} & \\textbf{$\\alpha$ (95\\% CI)} & \\textbf{Decision} \\\\",
        "\\midrule",
    ])
    for c in sim["clusters"]:
        cl = c["cluster"]
        med10 = c["round2_aggregate"]["median_per_item"][9]
        iqr10 = c["round2_aggregate"]["iqr_per_item"][9]
        ks = c["cohen_kappa_item10"]
        a = c["krippendorff_alpha"]
        decision = c["retention_decision"]
        # LaTeX-escape the label
        label = cl['label'].replace("&", "\\&")
        quad = cl["quadrant"].replace("_", "\\_")
        lines.append(
            f"{cl['id']} {label} & {quad} & {med10:.1f} & {iqr10:.1f} & "
            f"{ks['mean']:.2f} & {a['alpha']:.2f} ({a['ci_2_5']:.2f}, {a['ci_97_5']:.2f}) & {decision} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    lines.extend([
        "",
        "\\subsection*{Round-1 raw rating matrix (cluster C3, motor: \\textit{serious games / game-based learning}) --- illustrative}",
        "",
        "\\begin{tabular}{@{}l rrrrrrrrrr@{}}",
        "\\toprule",
        "\\textbf{Expert} & \\multicolumn{10}{c}{\\textbf{Item 1--10 (1--5 Likert)}} \\\\",
        "\\midrule",
    ])
    illustrative = next(c for c in sim["clusters"] if c["cluster"]["id"] == "C3")
    for e in PANEL:
        ratings = illustrative["round1"][e["id"]]
        lines.append(f"{e['id']} & " + " & ".join(str(r) for r in ratings) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    lines.extend([
        "",
        "\\subsection*{Disclosure}",
        "",
        "The figures above are produced by \\texttt{src/analysis/simulate\\_delphi.py} (random seed 42). The instrument is real and reusable; the panel and the ratings are synthetic. We include this output to (i)~illustrate the kind of reliability statistics the instrument delivers (Cohen's $\\kappa$ on the binary retain/revise judgement; Krippendorff's $\\alpha$ on the full Likert vector with 95\\% bootstrap CI from 300 resamples), and (ii)~scaffold the format of a real Delphi study without claiming validation. Any prescriptive use of the cluster labels in this manuscript should be preceded by an actual expert panel.",
    ])

    return "\n".join(lines)


def main() -> None:
    sim = simulate_all()
    out_json = ROOT / "data" / "processed" / "delphi_simulation.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(sim, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {out_json}")

    out_tex = ROOT / "supplementary" / "delphi_simulation_results.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(latex_for_results(sim), encoding="utf-8")
    print(f"Wrote {out_tex}")


if __name__ == "__main__":
    main()
