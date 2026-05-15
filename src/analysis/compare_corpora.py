#!/usr/bin/env python3
"""Compare original (88-doc, biased query) and de-biased (1511-doc) corpora.

Closes Risk-A from the revision plan: quantifies how much each finding shifts
when the platform-specific query disjunct is removed.

Outputs `data/processed/corpus_comparison.json` with per-analyzer side-by-side
metrics and `source/table_corpus_comparison.tex` for inclusion in the manuscript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from bibliometric_analyzer import (  # noqa: E402
    AffiliationBiasAnalyzer,
    ScopusAnalyzer,
    SensitivityAnalyzer,
    TemporalCouplingAnalyzer,
    TheoryOperationalisation,
)

ROOT = Path(__file__).resolve().parents[2]
ORIG = ROOT / "data" / "scopus_export_Dec 4-2025_bf7013ba-0a22-4eec-a067-f002be02a604.csv"
DEB = ROOT / "data" / "scopus_export_array_debiased_May 10-2026_26acd7b1-80e9-4c53-b5ff-8de800e0b0cd.csv"


def keyword_network_metrics(df: pd.DataFrame, min_occ: int = 3) -> Dict[str, Any]:
    G = SensitivityAnalyzer.from_corpus_keywords(df, min_occ=min_occ)
    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    if n_nodes == 0:
        return {"nodes": 0, "edges": 0}

    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G, weight="weight") if n_edges else {n: 0 for n in G}
    unity_node = next((n for n in G.nodes() if "unity" in str(n).lower()), None)
    top_by_deg = sorted(deg_cent.items(), key=lambda x: -x[1])[:10]
    top_by_bet = sorted(bet_cent.items(), key=lambda x: -x[1])[:10]

    sens = SensitivityAnalyzer(G)
    parts = sens.parameter_sweep()
    ari = sens.compute_ari_matrix(parts, baseline_key=1.0)["ari"]

    out = {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": nx.density(G),
        "avg_degree": sum(d for _, d in G.degree()) / max(n_nodes, 1),
        "n_clusters_at_gamma_1": len(parts.get(1.0, [])),
        "ari_vs_baseline_gamma_1": ari,
        "top_by_degree_centrality": [
            {"keyword": k, "deg_cent": v} for k, v in top_by_deg
        ],
        "top_by_betweenness_centrality": [
            {"keyword": k, "bet_cent": v} for k, v in top_by_bet
        ],
    }
    if unity_node:
        out["unity"] = {
            "node": unity_node,
            "degree": G.degree(unity_node),
            "degree_centrality": deg_cent[unity_node],
            "betweenness_centrality": bet_cent.get(unity_node, 0.0),
            "rank_by_degree_centrality": sorted(deg_cent.values(), reverse=True).index(deg_cent[unity_node]) + 1,
            "top_neighbors": sorted(
                G.neighbors(unity_node),
                key=lambda n: G[unity_node][n].get("weight", 0),
                reverse=True,
            )[:8],
        }
    else:
        out["unity"] = {"present": False, "note": "Unity not in network at min_occ=3"}
    return out


def affiliation_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    ab = AffiliationBiasAnalyzer(df)
    ix = ab.compute_concentration_indices()
    return {
        "hhi_country": ix["hhi_country"],
        "hhi_institution": ix["hhi_institution"],
        "gini_country": ix["gini_country"],
        "gini_institution": ix["gini_institution"],
        "unique_countries": ix["unique_countries"],
        "unique_institutions": ix["unique_institutions"],
        "top_countries": ix["top_countries"][:10],
    }


def temporal_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    tc = TemporalCouplingAnalyzer(df)
    eras = tc.per_era_coupling()
    comp = tc.compare_eras()
    return {
        "eras": {
            label: {
                "n_docs": info["n_docs"],
                "n_coupling_edges": info["n_coupling_edges"],
                "density": info["density"],
            }
            for label, info in eras.items()
        },
        "kendall_tau": comp["kendall_tau"],
        "top_refs_per_era": {
            label: refs[:5] for label, refs in comp["per_era_top_refs"].items()
        },
    }


def theory_metrics(df: pd.DataFrame, graph: nx.Graph) -> Dict[str, Any]:
    th = TheoryOperationalisation(df)
    payload = th.to_json(graph)
    scores = payload["platform_scores"]
    return {
        "n_platforms_with_corpus_mentions": len(scores),
        "h1": payload["h1_rogers_vs_centrality"],
        "h2": payload["h2_rogers_vs_year"],
        "h3": payload["h3_tam_vs_citations"],
        "platform_scores_summary": {
            p: {"n_docs": s["n_docs"],
                "rogers_total": s["rogers_total"],
                "avg_year": s["avg_pub_year"],
                "avg_citations": s["avg_citations"]}
            for p, s in scores.items()
        },
    }


def completeness_metrics(scopus: ScopusAnalyzer) -> Dict[str, Any]:
    return scopus.compute_field_completeness_score()


def run_one(label: str, csv_path: Path) -> Dict[str, Any]:
    print(f"\n=== {label} ({csv_path.name}) ===")
    scopus = ScopusAnalyzer(str(csv_path))
    df = scopus.df
    print(f"  documents: {len(df)}")
    G = SensitivityAnalyzer.from_corpus_keywords(df, min_occ=3)
    return {
        "n_documents": len(df),
        "year_range": [int(df["Year"].min()), int(df["Year"].max())] if "Year" in df.columns else None,
        "completeness": completeness_metrics(scopus),
        "keyword_network": keyword_network_metrics(df),
        "affiliation": affiliation_metrics(df),
        "temporal_coupling": temporal_metrics(df),
        "theory": theory_metrics(df, G),
    }


def main() -> None:
    out_path = ROOT / "data" / "processed" / "corpus_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "original_88": run_one("Original (biased query)", ORIG),
        "debiased_1511": run_one("De-biased", DEB),
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
