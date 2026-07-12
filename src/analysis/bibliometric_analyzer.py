#!/usr/bin/env python3
"""
Bibliometric Analyzer for VOSViewer Network Data and Scopus Exports

This module provides comprehensive bibliometric analysis including:
- VOSViewer map and network file parsing
- Network analysis using NetworkX (centrality, clustering, community detection)
- Scopus CSV data processing
- Visualization generation
- Results export for LaTeX integration

Author: Generated with Claude Opus 4.5 assistance
Date: 2026-01-09
"""

import os
import json
import csv
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings

# Scientific computing
import numpy as np
import pandas as pd

# Network analysis
import networkx as nx
from networkx.algorithms import community

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for academic figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Reproducibility — seed all stochastic algorithms (Louvain, async LPA, bootstrap).
# Required to make sensitivity-analysis comparisons valid (see SensitivityAnalyzer).
RANDOM_SEED = 42


def _set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python random and NumPy. Call at pipeline entry and before stochastic ops.

    Note: byte-identical clustering ALSO requires that graph node/edge insertion
    order be independent of ``PYTHONHASHSEED`` (Louvain is order-sensitive). We
    guarantee this by inserting nodes/edges in sorted order everywhere a graph is
    built from a Python ``set``/``dict`` (see ``from_corpus_keywords``), rather
    than relying on the hash seed.
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Keyword normalisation (co-word thesaurus + WordNet lemmatisation).
# Addresses Array reviewer R2 (minor #3): singular/plural + synonym fragmentation
# (e.g. "serious game" vs "serious games", "game" vs "games") splits what should
# be one node across several, inflating the node count and fragmenting clusters.
# Normalisation is deterministic — a fixed, version-controlled thesaurus file plus
# WordNet noun-lemmatisation of the head token — so canonical forms never depend on
# run order or hash seed. See ``keyword_thesaurus.json``.
# ---------------------------------------------------------------------------

_THESAURUS_PATH = Path(__file__).with_name('keyword_thesaurus.json')
_KW_WHITESPACE_RE = re.compile(r'\s+')
# Surrounding characters to trim from raw keywords (straight + curly quotes, dots).
_KW_STRIP_CHARS = '“”‘’"\'. '


def _load_thesaurus(path: Path = _THESAURUS_PATH) -> Dict[str, Any]:
    """Load the curated keyword thesaurus; degrade gracefully if absent."""
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'synonyms': {}, 'protect_plurals': set()}
    return {
        'synonyms': {str(k).strip().lower(): str(v) for k, v in data.get('synonyms', {}).items()},
        'protect_plurals': {str(w).lower() for w in data.get('protect_plurals', [])},
    }


_THESAURUS = _load_thesaurus()


def _get_lemmatizer():
    """Return a cached WordNet lemmatiser, or None if the corpus is unavailable."""
    cache = getattr(_get_lemmatizer, '_cache', 'unset')
    if cache != 'unset':
        return cache
    lem = None
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize('tests')  # force WordNet corpus load; raises if data missing
    except Exception:
        warnings.warn("WordNet lemmatiser unavailable; using rule-based plural "
                      "stripping for keyword normalisation (results may differ from "
                      "the WordNet-based reference run).")
        lem = None
    _get_lemmatizer._cache = lem
    return lem


def _rule_based_singular(word: str) -> str:
    """Deterministic, dependency-free fallback singulariser for a head token."""
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'
    if word.endswith(('ses', 'xes', 'zes', 'ches', 'shes')):
        return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 3:
        return word[:-1]
    return word


def _canonicalise_keyword(kw: str, thesaurus: Optional[Dict[str, Any]] = None) -> str:
    """Map a raw keyword to its canonical form.

    Steps (all deterministic): lower / strip / collapse-whitespace / strip quotes
    -> direct thesaurus lookup -> WordNet noun-lemmatise the head (last) token
    unless it is a protected non-plural (e.g. 'graphics', 'physics') -> re-check
    the thesaurus. Returns '' for empty input.
    """
    thes = thesaurus if thesaurus is not None else _THESAURUS
    synonyms = thes.get('synonyms', {})
    protect = thes.get('protect_plurals', set())

    s = _KW_WHITESPACE_RE.sub(' ', kw.strip().lower())
    s = s.strip(_KW_STRIP_CHARS)
    if not s:
        return ''
    if s in synonyms:
        return synonyms[s]

    tokens = s.split(' ')
    head = tokens[-1]
    if head and head not in protect:
        lem = _get_lemmatizer()
        singular = lem.lemmatize(head) if lem is not None else _rule_based_singular(head)
        if singular and singular != head:
            tokens[-1] = singular
    cand = ' '.join(tokens)
    return synonyms.get(cand, cand)


@dataclass
class VOSViewerNode:
    """Represents a node from VOSViewer map file."""
    id: int
    label: str
    x: float
    y: float
    cluster: int
    links: int
    total_link_strength: int
    occurrences: int
    avg_pub_year: Optional[float] = None
    avg_citations: Optional[float] = None
    avg_norm_citations: Optional[float] = None


@dataclass
class VOSViewerNetwork:
    """Represents a complete VOSViewer network."""
    name: str
    analysis_type: str
    unit_of_analysis: str
    nodes: Dict[int, VOSViewerNode]
    edges: List[Tuple[int, int, int]]  # (source_id, target_id, weight)
    graph: Optional[nx.Graph] = None


@dataclass
class NetworkMetrics:
    """Network-level metrics."""
    num_nodes: int
    num_edges: int
    density: float
    avg_clustering: float
    num_components: int
    modularity: float
    avg_degree: float
    diameter: Optional[int] = None


@dataclass
class NodeMetrics:
    """Node-level centrality metrics."""
    node_id: int
    label: str
    cluster: int
    degree: int
    betweenness: float
    closeness: float
    eigenvector: float
    pagerank: float


class VOSViewerParser:
    """Parser for VOSViewer map and network files."""

    # Analysis type mapping based on file naming convention
    ANALYSIS_TYPES = {
        1: ("Co-authorship", "Authors"),
        2: ("Co-authorship", "Organizations"),
        3: ("Co-authorship", "Countries"),
        4: ("Keyword Co-occurrence", "All Keywords"),
        5: ("Keyword Co-occurrence", "Author Keywords"),
        6: ("Keyword Co-occurrence", "Index Keywords"),
        7: ("Citation", "Documents"),
        8: ("Citation", "Sources/Journals"),
        9: ("Citation", "Authors"),
        10: ("Citation", "Organizations"),
        11: ("Citation", "Countries"),
        12: ("Bibliographic Coupling", "Documents"),
        13: ("Bibliographic Coupling", "Sources/Journals"),
        14: ("Bibliographic Coupling", "Authors"),
        15: ("Bibliographic Coupling", "Organizations"),
        16: ("Bibliographic Coupling", "Countries"),
        17: ("Co-citation", "Cited References"),
        18: ("Co-citation", "Cited Sources"),
        19: ("Co-citation", "Cited Authors"),
    }

    def __init__(self, map_dir: str, net_dir: str):
        self.map_dir = Path(map_dir)
        self.net_dir = Path(net_dir)

    def parse_map_file(self, filepath: Path) -> Dict[int, VOSViewerNode]:
        """Parse a VOSViewer map file."""
        nodes = {}

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)

            # Map column indices
            col_map = {col.lower().strip(): idx for idx, col in enumerate(header)}

            for row in reader:
                if not row or len(row) < 6:
                    continue

                try:
                    node = VOSViewerNode(
                        id=int(row[col_map.get('id', 0)]),
                        label=row[col_map.get('label', 1)],
                        x=float(row[col_map.get('x', 2)]),
                        y=float(row[col_map.get('y', 3)]),
                        cluster=int(row[col_map.get('cluster', 4)]),
                        links=int(row[col_map.get('weight<links>', 5)]) if len(row) > 5 else 0,
                        total_link_strength=int(row[col_map.get('weight<total link strength>', 6)]) if len(row) > 6 else 0,
                        occurrences=int(row[col_map.get('weight<occurrences>', 7)]) if len(row) > 7 else 0,
                        avg_pub_year=float(row[col_map.get('score<avg. pub. year>', 8)]) if len(row) > 8 and row[8] else None,
                        avg_citations=float(row[col_map.get('score<avg. citations>', 9)]) if len(row) > 9 and row[9] else None,
                        avg_norm_citations=float(row[col_map.get('score<avg. norm. citations>', 10)]) if len(row) > 10 and row[10] else None,
                    )
                    nodes[node.id] = node
                except (ValueError, IndexError) as e:
                    warnings.warn(f"Error parsing row in {filepath}: {e}")
                    continue

        return nodes

    def parse_net_file(self, filepath: Path) -> List[Tuple[int, int, int]]:
        """Parse a VOSViewer network file (edge list)."""
        edges = []

        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                if len(row) >= 2:
                    try:
                        source = int(row[0])
                        target = int(row[1])
                        weight = int(row[2]) if len(row) > 2 else 1
                        edges.append((source, target, weight))
                    except ValueError:
                        continue

        return edges

    def build_networkx_graph(self, nodes: Dict[int, VOSViewerNode],
                             edges: List[Tuple[int, int, int]]) -> nx.Graph:
        """Build a NetworkX graph from nodes and edges."""
        G = nx.Graph()

        # Add nodes with attributes
        for node_id, node in nodes.items():
            G.add_node(node_id, **asdict(node))

        # Add edges with weights
        for source, target, weight in edges:
            if source in nodes and target in nodes:
                G.add_edge(source, target, weight=weight)

        return G

    def parse_network(self, network_num: int) -> Optional[VOSViewerNetwork]:
        """Parse a complete network (map + net files) by number."""
        # Find map file
        map_file = self.map_dir / f"cosco{network_num}.txt"

        # Find net file (handling vosco vs cosco naming)
        net_file = self.net_dir / f"cosco{network_num}.txt"
        if not net_file.exists():
            net_file = self.net_dir / f"vosco{network_num}.txt"

        if not map_file.exists():
            warnings.warn(f"Map file not found: {map_file}")
            return None

        nodes = self.parse_map_file(map_file)
        edges = self.parse_net_file(net_file) if net_file.exists() else []

        # Get analysis type info
        analysis_type, unit = self.ANALYSIS_TYPES.get(network_num, ("Unknown", "Unknown"))

        graph = self.build_networkx_graph(nodes, edges) if nodes else None

        return VOSViewerNetwork(
            name=f"cosco{network_num}",
            analysis_type=analysis_type,
            unit_of_analysis=unit,
            nodes=nodes,
            edges=edges,
            graph=graph
        )

    def parse_all_networks(self, network_range: range = range(1, 20)) -> Dict[int, VOSViewerNetwork]:
        """Parse all available networks."""
        networks = {}

        for num in network_range:
            network = self.parse_network(num)
            if network:
                networks[num] = network
                print(f"Parsed network {num}: {network.analysis_type} - {network.unit_of_analysis} "
                      f"({len(network.nodes)} nodes, {len(network.edges)} edges)")

        return networks


class NetworkAnalyzer:
    """Analyzer for network metrics and community detection."""

    def __init__(self, network: VOSViewerNetwork):
        self.network = network
        self.graph = network.graph

    def calculate_network_metrics(self) -> NetworkMetrics:
        """Calculate network-level metrics."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return NetworkMetrics(0, 0, 0, 0, 0, 0, 0)

        G = self.graph

        # Basic metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)

        # Clustering
        avg_clustering = nx.average_clustering(G) if num_edges > 0 else 0

        # Components
        num_components = nx.number_connected_components(G)

        # Modularity using existing clusters (only if graph has edges)
        modularity = 0
        if num_edges > 0:
            clusters = defaultdict(set)
            for node_id, data in G.nodes(data=True):
                cluster_id = data.get('cluster', 0)
                clusters[cluster_id].add(node_id)

            partition = list(clusters.values())
            if len(partition) > 1:
                try:
                    modularity = nx.algorithms.community.modularity(G, partition)
                except (ZeroDivisionError, ValueError):
                    modularity = 0

        # Average degree
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0

        # Diameter (only for connected graphs)
        diameter = None
        if num_components == 1 and num_nodes > 1:
            try:
                diameter = nx.diameter(G)
            except:
                pass

        return NetworkMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_clustering=avg_clustering,
            num_components=num_components,
            modularity=modularity,
            avg_degree=avg_degree,
            diameter=diameter
        )

    def calculate_node_metrics(self) -> List[NodeMetrics]:
        """Calculate centrality metrics for all nodes."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return []

        G = self.graph

        # Calculate centralities
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight='weight')
        closeness = nx.closeness_centrality(G)

        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except:
            eigenvector = {n: 0 for n in G.nodes()}

        pagerank = nx.pagerank(G, weight='weight')

        metrics = []
        for node_id, data in G.nodes(data=True):
            metrics.append(NodeMetrics(
                node_id=node_id,
                label=data.get('label', str(node_id)),
                cluster=data.get('cluster', 0),
                degree=G.degree(node_id),
                betweenness=betweenness.get(node_id, 0),
                closeness=closeness.get(node_id, 0),
                eigenvector=eigenvector.get(node_id, 0),
                pagerank=pagerank.get(node_id, 0)
            ))

        return metrics

    def detect_communities(self, seed: int = RANDOM_SEED) -> Dict[str, List[set]]:
        """Detect communities using multiple algorithms. Seeded for reproducibility."""
        if self.graph is None or self.graph.number_of_nodes() < 3:
            return {}

        G = self.graph
        communities = {}

        # Louvain-like community detection (seeded).
        try:
            louvain_comms = community.louvain_communities(G, weight='weight', seed=seed)
            communities['louvain'] = louvain_comms
        except Exception:
            pass

        # Greedy modularity (deterministic given fixed graph node order).
        try:
            greedy_comms = community.greedy_modularity_communities(G, weight='weight')
            communities['greedy'] = list(greedy_comms)
        except Exception:
            pass

        # Asynchronous label propagation (seedable; sync label_propagation_communities is not).
        try:
            label_comms = community.asyn_lpa_communities(G, weight='weight', seed=seed)
            communities['label_propagation'] = list(label_comms)
        except Exception:
            pass

        return communities

    def get_cluster_summary(self) -> Dict[int, Dict]:
        """Get summary statistics for each cluster."""
        if self.graph is None:
            return {}

        clusters = defaultdict(list)
        for node_id, data in self.graph.nodes(data=True):
            clusters[data.get('cluster', 0)].append(data)

        summary = {}
        for cluster_id, nodes in clusters.items():
            labels = [n.get('label', '') for n in nodes]
            occurrences = [n.get('occurrences', 0) for n in nodes]
            avg_years = [n.get('avg_pub_year') for n in nodes if n.get('avg_pub_year')]

            summary[cluster_id] = {
                'num_nodes': len(nodes),
                'keywords': labels,
                'total_occurrences': sum(occurrences),
                'avg_year': np.mean(avg_years) if avg_years else None,
                'top_keywords': sorted(zip(labels, occurrences),
                                       key=lambda x: x[1], reverse=True)[:5]
            }

        return summary


class ScopusAnalyzer:
    """Analyzer for Scopus export CSV data."""

    def __init__(self, csv_path: str, year_max: Optional[int] = None):
        self.csv_path = Path(csv_path)
        # Optional upper year bound. Set to 2025 to exclude the incomplete 2026
        # partial year for the robustness variant (reviewers R1.3 / R2.2).
        self.year_max = year_max
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess Scopus CSV."""
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')

        # Clean column names
        df.columns = [col.strip() for col in df.columns]

        # Convert year to numeric
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        # Convert citations to numeric
        if 'Cited by' in df.columns:
            df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce').fillna(0).astype(int)

        # Optional year cap (excludes the incomplete final year for robustness).
        if self.year_max is not None and 'Year' in df.columns:
            df = df[df['Year'] <= self.year_max]

        return df

    def get_basic_stats(self) -> Dict:
        """Get basic dataset statistics."""
        stats = {
            'total_documents': len(self.df),
            'year_range': (self.df['Year'].min(), self.df['Year'].max()) if 'Year' in self.df.columns else None,
            'total_citations': self.df['Cited by'].sum() if 'Cited by' in self.df.columns else None,
            'avg_citations': self.df['Cited by'].mean() if 'Cited by' in self.df.columns else None,
        }

        # Count unique authors
        if 'Authors' in self.df.columns:
            all_authors = set()
            for authors in self.df['Authors'].dropna():
                for author in str(authors).split(';'):
                    all_authors.add(author.strip())
            stats['unique_authors'] = len(all_authors)

        # Count unique sources
        if 'Source title' in self.df.columns:
            stats['unique_sources'] = self.df['Source title'].nunique()

        # Extract countries from affiliations
        if 'Affiliations' in self.df.columns:
            countries = set()
            for aff in self.df['Affiliations'].dropna():
                # Extract country (last part after comma typically)
                parts = str(aff).split(';')
                for part in parts:
                    if ',' in part:
                        country = part.split(',')[-1].strip()
                        if len(country) > 2:
                            countries.add(country)
            stats['unique_countries'] = len(countries)

        return stats

    def get_publications_by_year(self) -> pd.DataFrame:
        """Get publication counts by year."""
        if 'Year' not in self.df.columns:
            return pd.DataFrame()

        return self.df.groupby('Year').size().reset_index(name='count')

    def compute_field_completeness_score(self) -> Dict[str, float]:
        """Fit competing growth models to the cumulative-publications curve.

        Compares linear, exponential, single-wave **Gompertz**, and two-wave
        **bi-logistic** (Meyer 1994) fits, reporting R² and AIC/BIC for each and
        selecting ``best_model`` by AIC. The Gompertz ``saturation_ratio`` =
        (latest cumulative) / (Gompertz asymptote).

        Addresses reviewer R1.3 (sparse field vs restrictive query) AND R1.6/R1.7:
        a single saturating curve can misread a disruption-driven regrowth as
        "maturity", so we test whether a multi-wave (bi-logistic) model is
        preferred and interpret the saturation ratio as an index-scope estimate
        rather than a maturity verdict.
        """
        nan = float('nan')
        empty = {'gompertz_r2': nan, 'linear_r2': nan, 'exponential_r2': nan,
                 'bilogistic_r2': nan, 'saturation_ratio': nan, 'best_model': None}
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            return empty

        df_y = self.get_publications_by_year().sort_values('Year')
        if df_y.empty or len(df_y) < 4:
            return empty

        years = df_y['Year'].astype(float).to_numpy()
        cum = df_y['count'].cumsum().astype(float).to_numpy()
        t = years - years.min()
        n = len(t)

        def ssr(observed, predicted):
            return float(np.sum((observed - predicted) ** 2))

        def r2_from_ssr(ss_res):
            ss_tot = float(np.sum((cum - cum.mean()) ** 2))
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        def aic_bic(ss_res, k):
            # Gaussian-likelihood AIC/BIC; k = model params, +1 for error variance.
            if not np.isfinite(ss_res) or ss_res <= 0 or n <= k + 1:
                return nan, nan
            ll_term = n * np.log(ss_res / n)
            return float(ll_term + 2 * (k + 1)), float(ll_term + (k + 1) * np.log(n))

        results: Dict[str, float] = {'cumulative_at_last_year': float(cum[-1]),
                                     'years_observed': int(n)}
        ssr_by_model: Dict[str, Tuple[float, int]] = {}

        # Linear: cum = a*t + b (2 params).
        linear_pred = np.polyval(np.polyfit(t, cum, 1), t)
        ssr_by_model['linear'] = (ssr(cum, linear_pred), 2)

        # Exponential: cum = a*exp(b*t) (2 params).
        def exp_model(x, a, b):
            return a * np.exp(np.clip(b * x, -50, 50))
        try:
            exp_p, _ = curve_fit(exp_model, t, cum, p0=(1.0, 0.1), maxfev=10000)
            ssr_by_model['exponential'] = (ssr(cum, exp_model(t, *exp_p)), 2)
        except (RuntimeError, ValueError):
            ssr_by_model['exponential'] = (nan, 2)

        # Gompertz: cum = K*exp(-b*exp(-c*t)) (3 params).
        def gompertz(x, K, b, c):
            return K * np.exp(-b * np.exp(-c * x))
        K_guess = max(cum[-1] * 1.5, 100.0)
        saturation = nan
        gompertz_K = nan
        try:
            g_p, _ = curve_fit(gompertz, t, cum, p0=(K_guess, 5.0, 0.2), maxfev=20000,
                               bounds=([cum[-1], 0.1, 0.01], [cum[-1] * 100, 50.0, 5.0]))
            ssr_by_model['gompertz'] = (ssr(cum, gompertz(t, *g_p)), 3)
            gompertz_K = float(g_p[0])
            saturation = float(cum[-1] / g_p[0]) if g_p[0] > 0 else nan
        except (RuntimeError, ValueError):
            ssr_by_model['gompertz'] = (nan, 3)

        # Bi-logistic: two additive logistic waves (Meyer 1994), 6 params.
        def bilogistic(x, L1, k1, t1, L2, k2, t2):
            w1 = L1 / (1.0 + np.exp(-np.clip(k1 * (x - t1), -50, 50)))
            w2 = L2 / (1.0 + np.exp(-np.clip(k2 * (x - t2), -50, 50)))
            return w1 + w2
        T = float(t.max())
        bilogistic_asymptote = nan
        try:
            bl_p, _ = curve_fit(
                bilogistic, t, cum,
                p0=(cum[-1] * 0.5, 0.4, T * 0.45, cum[-1] * 0.5, 0.4, T * 0.85),
                maxfev=40000,
                bounds=([0, 0.01, 0, 0, 0.01, 0],
                        [cum[-1] * 5, 5.0, T * 1.5, cum[-1] * 5, 5.0, T * 1.5]),
            )
            ssr_by_model['bilogistic'] = (ssr(cum, bilogistic(t, *bl_p)), 6)
            bilogistic_asymptote = float(bl_p[0] + bl_p[3])
        except (RuntimeError, ValueError):
            ssr_by_model['bilogistic'] = (nan, 6)

        # R² + AIC/BIC per model; pick best by AIC.
        best_model, best_aic = None, np.inf
        for name, (ss, k) in ssr_by_model.items():
            results[f'{name}_r2'] = float(r2_from_ssr(ss)) if np.isfinite(ss) else nan
            a, b = aic_bic(ss, k)
            results[f'{name}_aic'] = a
            results[f'{name}_bic'] = b
            if np.isfinite(a) and a < best_aic:
                best_aic, best_model = a, name

        results['saturation_ratio'] = saturation
        results['gompertz_asymptote'] = gompertz_K
        results['bilogistic_asymptote'] = bilogistic_asymptote
        results['best_model'] = best_model
        results['delta_aic_bilogistic_minus_gompertz'] = (
            results['bilogistic_aic'] - results['gompertz_aic']
            if np.isfinite(results.get('bilogistic_aic', nan))
            and np.isfinite(results.get('gompertz_aic', nan)) else nan)
        return results

    def get_top_authors(self, n: int = 10) -> pd.DataFrame:
        """Get top authors by publication count and citations."""
        if 'Authors' not in self.df.columns:
            return pd.DataFrame()

        author_stats = defaultdict(lambda: {'publications': 0, 'citations': 0})

        for _, row in self.df.iterrows():
            if pd.isna(row['Authors']):
                continue
            authors = str(row['Authors']).split(';')
            citations = row.get('Cited by', 0) or 0

            for author in authors:
                author = author.strip()
                if author:
                    author_stats[author]['publications'] += 1
                    author_stats[author]['citations'] += citations

        df_authors = pd.DataFrame([
            {'Author': k, 'Publications': v['publications'], 'Citations': v['citations']}
            for k, v in author_stats.items()
        ])

        return df_authors.nlargest(n, 'Publications')

    def get_top_sources(self, n: int = 10) -> pd.DataFrame:
        """Get top sources/journals by publication count."""
        if 'Source title' not in self.df.columns:
            return pd.DataFrame()

        source_counts = self.df.groupby('Source title').agg({
            'Title': 'count',
            'Cited by': 'sum'
        }).reset_index()
        source_counts.columns = ['Source', 'Publications', 'Total Citations']

        return source_counts.nlargest(n, 'Publications')

    def get_keyword_frequency(self, keyword_col: str = 'Author Keywords') -> Dict[str, int]:
        """Get keyword frequency distribution."""
        if keyword_col not in self.df.columns:
            return {}

        keyword_counts = defaultdict(int)

        for keywords in self.df[keyword_col].dropna():
            for kw in str(keywords).split(';'):
                kw = _canonicalise_keyword(kw)
                if kw:
                    keyword_counts[kw] += 1

        # Sort by count desc, then key asc, for a deterministic ordering on ties.
        return dict(sorted(keyword_counts.items(), key=lambda x: (-x[1], x[0])))


# Country-name normalisation. Scopus exports use heterogeneous spellings
# (USA / United States / U.S.A.); collapse them so HHI/Gini are not inflated.
_COUNTRY_ALIASES = {
    'usa': 'United States', 'us': 'United States',
    'united states of america': 'United States', 'united states': 'United States',
    'uk': 'United Kingdom', 'great britain': 'United Kingdom',
    'britain': 'United Kingdom', 'united kingdom': 'United Kingdom',
    'south korea': 'Korea, Republic of', 'republic of korea': 'Korea, Republic of',
    'korea': 'Korea, Republic of',
    'czech republic': 'Czechia',
    'russia': 'Russian Federation',
    'viet nam': 'Vietnam',
    'iran': 'Iran, Islamic Republic of',
}


def _normalise_country(raw: str) -> str:
    """Normalise a country string against common Scopus spelling variants.

    Strips trailing punctuation and collapses interior periods so that
    'U.S.A.' / 'USA' / 'U.S.A' all map to 'United States'.
    """
    s = (raw or '').strip().rstrip('.').strip()
    key = s.lower().replace('.', '').strip()
    return _COUNTRY_ALIASES.get(key, s)


class AffiliationBiasAnalyzer:
    """Audits geographic and institutional concentration of the source corpus.

    Tests the reviewer hypothesis (R1.5) that the dual ACM/IEEE intellectual
    pillar is a side-effect of geographic affiliation clustering rather than
    genuine intellectual convergence.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def extract_author_institution_country_triples(self) -> List[Dict[str, Any]]:
        """Parse Scopus 'Authors with affiliations' into (doc_id, author, institution, country) rows.

        Format: 'Surname, First, Institution, City, State, Country; ...'
        """
        col = 'Authors with affiliations'
        if col not in self.df.columns:
            return []

        triples = []
        for idx, raw in self.df[col].dropna().items():
            for chunk in str(raw).split(';'):
                parts = [p.strip() for p in chunk.split(',') if p.strip()]
                if len(parts) < 4:
                    # Need at least Surname, First, Institution, Country.
                    continue
                author = f"{parts[0]}, {parts[1]}"
                institution = parts[2]
                country = _normalise_country(parts[-1])
                triples.append({
                    'doc_id': int(idx),
                    'author': author,
                    'institution': institution,
                    'country': country,
                })
        return triples

    def compute_concentration_indices(self) -> Dict[str, Any]:
        """Herfindahl-Hirschman Index and Gini for institution and country distributions.

        HHI is on a 0-1 scale; HHI > 0.25 conventionally indicates high concentration.
        Gini is on a 0-1 scale; 0 = perfectly uniform, 1 = perfectly concentrated.
        """
        triples = self.extract_author_institution_country_triples()
        if not triples:
            return {'hhi_country': 0.0, 'hhi_institution': 0.0,
                    'gini_country': 0.0, 'gini_institution': 0.0}

        # Document-level aggregation (one country per doc-country pair) to
        # avoid over-weighting docs with many co-authors from the same country.
        doc_country = defaultdict(set)
        doc_inst = defaultdict(set)
        for t in triples:
            doc_country[t['doc_id']].add(t['country'])
            doc_inst[t['doc_id']].add(t['institution'])

        country_counts = defaultdict(int)
        for countries in doc_country.values():
            for c in countries:
                country_counts[c] += 1
        inst_counts = defaultdict(int)
        for insts in doc_inst.values():
            for i in insts:
                inst_counts[i] += 1

        return {
            'hhi_country': self._hhi(list(country_counts.values())),
            'hhi_institution': self._hhi(list(inst_counts.values())),
            'gini_country': self._gini(list(country_counts.values())),
            'gini_institution': self._gini(list(inst_counts.values())),
            'top_countries': sorted(country_counts.items(), key=lambda x: -x[1])[:10],
            'top_institutions': sorted(inst_counts.items(), key=lambda x: -x[1])[:10],
            'unique_countries': len(country_counts),
            'unique_institutions': len(inst_counts),
        }

    @staticmethod
    def _hhi(counts: List[int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        shares = np.array(counts, dtype=float) / total
        return float(np.sum(shares ** 2))

    @staticmethod
    def _gini(counts: List[int]) -> float:
        if not counts or sum(counts) == 0:
            return 0.0
        x = np.sort(np.array(counts, dtype=float))
        n = len(x)
        # Standard Gini for non-negative values.
        cum = np.cumsum(x)
        return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)

    def compute_collaboration_network(self) -> nx.Graph:
        """Country-country co-authorship graph; edge weight = number of co-authored docs."""
        triples = self.extract_author_institution_country_triples()
        doc_countries = defaultdict(set)
        for t in triples:
            doc_countries[t['doc_id']].add(t['country'])

        G = nx.Graph()
        for countries in doc_countries.values():
            countries = sorted(countries)
            for c in countries:
                G.add_node(c)
            # Co-authorship: every pair within the doc.
            for i, a in enumerate(countries):
                for b in countries[i + 1:]:
                    if G.has_edge(a, b):
                        G[a][b]['weight'] += 1
                    else:
                        G.add_edge(a, b, weight=1)
        return G

    def cocitation_geography_overlay(self, cluster_assignments: Dict[str, int]) -> Dict[int, Dict[str, int]]:
        """Per-cluster country distribution.

        ``cluster_assignments`` maps a citation-cluster label (e.g. 'ACM' or
        cluster_id) to the integer cluster_id used by VOSViewer; we report
        per-cluster country counts of the *citing* documents so the dual-pillar
        bias hypothesis can be tested directly.
        """
        triples = self.extract_author_institution_country_triples()
        # Map docs to single most-frequent country per doc for cluster overlay.
        doc_country = defaultdict(lambda: defaultdict(int))
        for t in triples:
            doc_country[t['doc_id']][t['country']] += 1
        doc_primary = {d: max(c.items(), key=lambda x: x[1])[0] for d, c in doc_country.items()}

        per_cluster = defaultdict(lambda: defaultdict(int))
        for doc_id, cluster_id in cluster_assignments.items():
            country = doc_primary.get(int(doc_id))
            if country:
                per_cluster[cluster_id][country] += 1
        return {k: dict(v) for k, v in per_cluster.items()}

    def to_json(self) -> Dict[str, Any]:
        """Serialise concentration indices + collaboration network for storage."""
        indices = self.compute_concentration_indices()
        G = self.compute_collaboration_network()
        return {
            **indices,
            'collaboration_edges': [
                {'source': u, 'target': v, 'weight': d.get('weight', 1)}
                for u, v, d in G.edges(data=True)
            ],
            'collaboration_nodes': list(G.nodes()),
        }


def _adjusted_rand_index(labels_a: List[int], labels_b: List[int]) -> float:
    """Adjusted Rand Index between two label assignments. Pure Python.

    Returns 1.0 when partitions are identical (up to relabeling), 0.0 for the
    expected value of random partitions, can be negative for worse-than-random.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Label arrays must be the same length")
    n = len(labels_a)
    if n < 2:
        return 1.0

    def comb2(k: int) -> int:
        return k * (k - 1) // 2 if k >= 2 else 0

    contingency: Dict[Tuple[int, int], int] = defaultdict(int)
    row_sums: Dict[int, int] = defaultdict(int)
    col_sums: Dict[int, int] = defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        contingency[(a, b)] += 1
        row_sums[a] += 1
        col_sums[b] += 1

    sum_ij = sum(comb2(v) for v in contingency.values())
    sum_a = sum(comb2(v) for v in row_sums.values())
    sum_b = sum(comb2(v) for v in col_sums.values())
    n_pairs = comb2(n)

    if n_pairs == 0:
        return 1.0
    expected = sum_a * sum_b / n_pairs
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


class SensitivityAnalyzer:
    """Cluster-stability and parameter-sensitivity sweeps for VOSViewer-style networks.

    Closes R1.2 (cluster stability under resolution / min-occurrence parameters).
    Reports ARI vs the published baseline and per-cluster Jaccard with bootstrap CIs.
    """

    DEFAULT_GAMMAS = (0.5, 0.75, 1.0, 1.25, 1.5)
    DEFAULT_MIN_OCCS = (1, 2, 3, 5)

    def __init__(self, base_graph: nx.Graph, baseline_seed: int = RANDOM_SEED):
        self.base_graph = base_graph
        self.baseline_seed = baseline_seed

    @classmethod
    def from_corpus_keywords(cls, df: pd.DataFrame,
                             keyword_col: str = 'Author Keywords',
                             min_occ: int = 1,
                             canonicalise: bool = True) -> nx.Graph:
        """Build a keyword co-occurrence graph directly from the Scopus dataframe.

        Useful for sweeping min_occ (which is baked into the VOSViewer export
        and otherwise un-tunable post-hoc).

        ``canonicalise`` applies the thesaurus + lemmatiser normalisation
        (``_canonicalise_keyword``) so singular/plural and synonym variants
        collapse to one node (reviewer R2 minor #3). Set ``False`` to reproduce
        the un-normalised graph for the with/without robustness comparison.

        Nodes and edges are inserted in *sorted* order so the resulting graph —
        and hence the order-sensitive Louvain partition — is byte-identical
        regardless of ``PYTHONHASHSEED``.
        """
        norm = _canonicalise_keyword if canonicalise else (lambda k: k.strip().lower())
        per_doc_keywords = []
        for raw in df[keyword_col].dropna():
            kws = {c for c in (norm(kw) for kw in str(raw).split(';') if kw.strip()) if c}
            if kws:
                per_doc_keywords.append(kws)

        # Count occurrences (docs containing the keyword).
        occ = defaultdict(int)
        for kws in per_doc_keywords:
            for kw in kws:
                occ[kw] += 1

        # Apply min_occ filter.
        keep = {kw for kw, c in occ.items() if c >= min_occ}

        # Co-occurrence edge weights.
        co = defaultdict(int)
        for kws in per_doc_keywords:
            kept = sorted(k for k in kws if k in keep)
            for i, a in enumerate(kept):
                for b in kept[i + 1:]:
                    co[(a, b)] += 1

        # Sorted insertion → deterministic node/edge order (see _set_global_seed).
        G = nx.Graph()
        for kw in sorted(keep):
            G.add_node(kw, occurrences=occ[kw], label=kw)
        for (a, b) in sorted(co):
            G.add_edge(a, b, weight=co[(a, b)])
        return G

    def parameter_sweep(self,
                        gammas: Tuple[float, ...] = DEFAULT_GAMMAS,
                        seed: int = RANDOM_SEED) -> Dict[float, List[set]]:
        """Run Louvain for each γ on the base graph; return γ → list-of-clusters."""
        partitions = {}
        for gamma in gammas:
            try:
                partitions[gamma] = community.louvain_communities(
                    self.base_graph, weight='weight',
                    resolution=gamma, seed=seed,
                )
            except Exception:
                partitions[gamma] = []
        return partitions

    def parameter_sweep_min_occ(self, df: pd.DataFrame,
                                min_occs: Tuple[int, ...] = DEFAULT_MIN_OCCS,
                                gamma: float = 1.0,
                                keyword_col: str = 'Author Keywords',
                                seed: int = RANDOM_SEED) -> Dict[int, List[set]]:
        """Sweep min-occurrence threshold by rebuilding the keyword graph each time."""
        partitions = {}
        for mo in min_occs:
            G = self.from_corpus_keywords(df, keyword_col, mo)
            if G.number_of_nodes() < 3:
                partitions[mo] = []
                continue
            partitions[mo] = community.louvain_communities(
                G, weight='weight', resolution=gamma, seed=seed,
            )
        return partitions

    @staticmethod
    def _partition_to_labels(partition: List[set], all_nodes: List) -> List[int]:
        """Convert a list-of-sets partition into a per-node integer label vector."""
        node_to_cluster = {}
        for cluster_id, members in enumerate(partition):
            for node in members:
                node_to_cluster[node] = cluster_id
        return [node_to_cluster.get(n, -1) for n in all_nodes]

    def compute_ari_matrix(self, partitions: Dict[Any, List[set]],
                           baseline_key: Any = 1.0) -> Dict[str, Any]:
        """ARI of every partition against the baseline_key partition.

        Returns dict mapping each parameter → ARI float, plus the baseline label.
        """
        if baseline_key not in partitions:
            return {'baseline': baseline_key, 'ari': {}}

        baseline = partitions[baseline_key]
        # Use the union of nodes across all partitions to handle min_occ sweeps
        # that change the node set.
        all_nodes = sorted({n for p in partitions.values() for c in p for n in c})
        baseline_labels = self._partition_to_labels(baseline, all_nodes)

        ari = {}
        for k, p in partitions.items():
            labels = self._partition_to_labels(p, all_nodes)
            ari[str(k)] = _adjusted_rand_index(baseline_labels, labels)
        return {'baseline': baseline_key, 'ari': ari}

    def compute_jaccard_stability(self, partitions: Dict[Any, List[set]],
                                  baseline_key: Any = 1.0) -> Dict[str, List[float]]:
        """For each non-baseline partition, return the maximum Jaccard each
        baseline cluster achieves against any cluster in the variant partition.

        High values (close to 1.0) indicate stable clusters; low values flag
        clusters that fragment or merge under the parameter change.
        """
        if baseline_key not in partitions:
            return {}

        baseline = [set(c) for c in partitions[baseline_key]]
        result = {}
        for k, p in partitions.items():
            if k == baseline_key:
                continue
            variant = [set(c) for c in p]
            per_cluster_jaccards = []
            for base_cluster in baseline:
                best = max((_jaccard(base_cluster, v) for v in variant), default=0.0)
                per_cluster_jaccards.append(best)
            result[str(k)] = per_cluster_jaccards
        return result

    def bootstrap_jaccard_ci(self,
                             gamma: float = 1.0,
                             n_bootstrap: int = 500,
                             frac: float = 0.9,
                             seed: int = RANDOM_SEED) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for cluster stability.

        For each iteration, drop ``1-frac`` of the edges at random, re-run
        Louvain, and compute the mean Jaccard against the original partition.
        Returns 95% CI (2.5–97.5 percentile) for the mean stability.
        """
        rng = np.random.default_rng(seed)
        original = community.louvain_communities(
            self.base_graph, weight='weight', resolution=gamma, seed=seed,
        )
        edges = list(self.base_graph.edges(data=True))
        means: List[float] = []
        for _ in range(n_bootstrap):
            keep_idx = rng.choice(len(edges), size=int(frac * len(edges)), replace=False)
            G_b = nx.Graph()
            G_b.add_nodes_from(self.base_graph.nodes(data=True))
            for i in keep_idx:
                u, v, d = edges[i]
                G_b.add_edge(u, v, **d)
            try:
                p = community.louvain_communities(
                    G_b, weight='weight', resolution=gamma,
                    seed=int(rng.integers(0, 1_000_000)),
                )
            except Exception:
                continue
            jaccards = []
            for base_c in original:
                base_set = set(base_c)
                best = max((_jaccard(base_set, set(v)) for v in p), default=0.0)
                jaccards.append(best)
            if jaccards:
                means.append(float(np.mean(jaccards)))
        if not means:
            return {'mean': (0.0, 0.0)}
        arr = np.array(means)
        return {
            'mean': (float(np.mean(arr)), float(np.std(arr))),
            'ci_95': (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
            'n_bootstrap': len(means),
        }

    # ------------------------------------------------------------------
    # Alternative community-detection algorithms (reviewer R1 #5).
    # Louvain is compared against Leiden (guarantees well-connected communities,
    # Traag et al. 2019) and Infomap (map-equation flow model, Rosvall &
    # Bergstrom 2008). All three are seeded for byte-identical output; nodes are
    # sorted so the igraph/Infomap index mapping is deterministic.
    # ------------------------------------------------------------------
    def _sorted_index(self) -> Tuple[List[Any], Dict[Any, int]]:
        nodes = sorted(self.base_graph.nodes())
        return nodes, {n: i for i, n in enumerate(nodes)}

    def partition_leiden(self, gamma: float = 1.0, seed: int = RANDOM_SEED,
                         n_iterations: int = -1) -> Optional[List[set]]:
        """Leiden partition (RBConfiguration ≈ Louvain resolution). None if unavailable."""
        try:
            import igraph as ig
            import leidenalg
        except Exception:
            warnings.warn("leidenalg/igraph unavailable; skipping Leiden comparison.")
            return None
        nodes, idx = self._sorted_index()
        edges, weights = [], []
        for u, v, d in self.base_graph.edges(data=True):
            edges.append((idx[u], idx[v]))
            weights.append(float(d.get('weight', 1.0)))
        g = ig.Graph(n=len(nodes), edges=edges)
        if weights:
            g.es['weight'] = weights
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight' if weights else None,
            resolution_parameter=gamma, seed=seed, n_iterations=n_iterations,
        )
        return [set(nodes[i] for i in comm) for comm in part]

    def partition_infomap(self, seed: int = RANDOM_SEED,
                          num_trials: int = 10) -> Optional[List[set]]:
        """Infomap (map-equation) partition. None if the package is unavailable."""
        try:
            from infomap import Infomap
        except Exception:
            warnings.warn("infomap unavailable; skipping Infomap comparison.")
            return None
        nodes, idx = self._sorted_index()
        im = Infomap(f"--two-level --silent --seed {seed} --num-trials {num_trials}")
        for u, v, d in self.base_graph.edges(data=True):
            im.add_link(idx[u], idx[v], float(d.get('weight', 1.0)))
        im.run()
        modules: Dict[Any, set] = defaultdict(set)
        assigned = im.get_modules()
        for i, n in enumerate(nodes):
            mid = assigned.get(i, ('iso', i))
            modules[mid].add(n)
        return list(modules.values())

    def algorithm_agreement(self, gamma: float = 1.0,
                            seed: int = RANDOM_SEED) -> Dict[str, Any]:
        """Compare Louvain / Leiden / Infomap on the base graph.

        Returns per-algorithm cluster count + modularity and the ARI of Leiden
        and Infomap against the Louvain baseline (reuses ``compute_ari_matrix``).
        """
        parts: Dict[str, List[set]] = {}
        louv = self.parameter_sweep((gamma,), seed=seed).get(gamma, [])
        if louv:
            parts['louvain'] = louv
        lei = self.partition_leiden(gamma=gamma, seed=seed)
        if lei is not None:
            parts['leiden'] = lei
        inf = self.partition_infomap(seed=seed)
        if inf is not None:
            parts['infomap'] = inf

        summary = {}
        for name, p in parts.items():
            try:
                mod = community.modularity(self.base_graph, p, weight='weight')
            except Exception:
                mod = float('nan')
            summary[name] = {'n_clusters': len(p), 'modularity': float(mod)}

        ari = self.compute_ari_matrix(parts, baseline_key='louvain') \
            if 'louvain' in parts else {'ari': {}}
        return {
            'gamma': gamma,
            'baseline': 'louvain',
            'summary': summary,
            'ari_vs_louvain': ari.get('ari', {}),
        }

    def to_json(self,
                gammas: Tuple[float, ...] = DEFAULT_GAMMAS,
                df: Optional[pd.DataFrame] = None,
                min_occs: Tuple[int, ...] = DEFAULT_MIN_OCCS,
                keyword_col: str = 'Author Keywords') -> Dict[str, Any]:
        """One-shot serialisation: γ-sweep, min_occ-sweep (if df given), ARI, Jaccard."""
        out: Dict[str, Any] = {}
        gamma_partitions = self.parameter_sweep(gammas)
        out['resolution_sweep'] = {
            str(g): [sorted(c) for c in p] for g, p in gamma_partitions.items()
        }
        out['cluster_counts_by_gamma'] = {str(g): len(p) for g, p in gamma_partitions.items()}
        out['ari_vs_baseline_gamma'] = self.compute_ari_matrix(gamma_partitions, baseline_key=1.0)
        out['jaccard_vs_baseline_gamma'] = self.compute_jaccard_stability(
            gamma_partitions, baseline_key=1.0,
        )
        out['bootstrap_stability_gamma_1'] = self.bootstrap_jaccard_ci(gamma=1.0)
        out['algorithm_agreement'] = self.algorithm_agreement(gamma=1.0)

        if df is not None:
            mo_partitions = self.parameter_sweep_min_occ(
                df, min_occs=min_occs, keyword_col=keyword_col,
            )
            out['min_occ_sweep'] = {
                str(mo): [sorted(c) for c in p] for mo, p in mo_partitions.items()
            }
            out['cluster_counts_by_min_occ'] = {str(mo): len(p) for mo, p in mo_partitions.items()}

        return out


def compute_normalization_effect(df: pd.DataFrame,
                                 keyword_col: str = 'Author Keywords',
                                 min_occ: int = 3,
                                 gamma: float = 1.0,
                                 seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Quantify the effect of thesaurus+lemmatiser keyword normalisation.

    Answers reviewer R2 (minor #3) with evidence, not just a claim: reports how
    many raw variants collapse, the node/edge and cluster-count change, the
    largest merges, and an Adjusted Rand Index between the un-normalised and
    normalised Louvain partitions (mapping each raw term to its canonical form)
    to show that consolidation does not destabilise the thematic structure.
    """
    raw_g = SensitivityAnalyzer.from_corpus_keywords(df, keyword_col, min_occ, canonicalise=False)
    can_g = SensitivityAnalyzer.from_corpus_keywords(df, keyword_col, min_occ, canonicalise=True)

    # Group raw variants by their canonical form to surface the largest merges.
    merges: Dict[str, set] = defaultdict(set)
    for raw in df[keyword_col].dropna():
        for kw in str(raw).split(';'):
            r = kw.strip().lower()
            c = _canonicalise_keyword(kw)
            if c:
                merges[c].add(r)
    multi = {c: sorted(v) for c, v in merges.items() if len(v) > 1}
    top_merges = sorted(multi.items(), key=lambda x: -len(x[1]))[:15]

    p_raw = community.louvain_communities(raw_g, weight='weight', resolution=gamma, seed=seed)
    p_can = community.louvain_communities(can_g, weight='weight', resolution=gamma, seed=seed)
    raw_label = {n: i for i, c in enumerate(p_raw) for n in c}
    can_label = {n: i for i, c in enumerate(p_can) for n in c}
    shared = [n for n in raw_g.nodes()
              if n in raw_label and _canonicalise_keyword(n) in can_label]
    ari = _adjusted_rand_index(
        [raw_label[n] for n in shared],
        [can_label[_canonicalise_keyword(n)] for n in shared],
    ) if len(shared) >= 2 else float('nan')

    return {
        'min_occ': min_occ,
        'raw_nodes': raw_g.number_of_nodes(),
        'raw_edges': raw_g.number_of_edges(),
        'normalised_nodes': can_g.number_of_nodes(),
        'normalised_edges': can_g.number_of_edges(),
        'nodes_merged_away': raw_g.number_of_nodes() - can_g.number_of_nodes(),
        'n_canonical_terms_with_merges': len(multi),
        'raw_clusters': len(p_raw),
        'normalised_clusters': len(p_can),
        'ari_raw_vs_normalised': ari,
        'top_merges': [{'canonical': c, 'variants': v} for c, v in top_merges],
    }


INDUSTRY_TERM_FAMILIES: Dict[str, List[str]] = {
    # Industry-salient topics the reviewer (R2 major #3) flags as peripheral/absent.
    'mobile': ['mobile', 'mobile game', 'mobile development', 'mobile learning',
               'android', 'ios', 'smartphone', 'mobile application', 'app development'],
    'generative_ai': ['generative ai', 'large language model', 'chatgpt', 'gpt',
                      'llm', 'generative artificial intelligence', 'diffusion model',
                      'prompt engineering', 'copilot', 'genai'],
    # Contrast families that ARE central, to calibrate the lag claim.
    'immersive_xr': ['virtual reality', 'augmented reality', 'mixed reality',
                     'extended reality', 'metaverse'],
    'ai_general': ['artificial intelligence', 'machine learning', 'deep learning',
                   'neural network', 'procedural content generation'],
}


def compute_industry_lag(df: pd.DataFrame,
                         keyword_col: str = 'Author Keywords',
                         min_occ: int = 3) -> Dict[str, Any]:
    """Quantify the academia–industry lag (reviewer R2 major #3) empirically.

    For each term family, contrast three signals: (a) documents whose *author
    keywords* contain a family term (network-eligible), (b) documents whose
    *abstract* mentions it (literature presence), and (c) whether any family term
    reaches ``min_occ`` as an author keyword — i.e. whether it appears as a node
    in the keyword co-occurrence network at all. A family that is common in
    abstracts but absent from the keyword network is the lag made measurable.
    """
    has_abstract = 'Abstract' in df.columns
    years = df['Year'] if 'Year' in df.columns else None
    kw_node_terms = set()
    if keyword_col in df.columns:
        kw_node_terms = {n for n in
                         SensitivityAnalyzer.from_corpus_keywords(df, keyword_col, min_occ).nodes()}

    out: Dict[str, Any] = {'min_occ': min_occ, 'families': {}}
    for fam, terms in INDUSTRY_TERM_FAMILIES.items():
        canon_terms = {_canonicalise_keyword(t) for t in terms}
        kw_docs, abs_docs, recent_abs = set(), set(), set()
        per_year: Dict[int, int] = defaultdict(int)
        for idx in df.index:
            kw_raw = str(df.at[idx, keyword_col]) if keyword_col in df.columns else ''
            kw_set = {_canonicalise_keyword(k) for k in kw_raw.split(';') if k.strip()}
            in_kw = bool(kw_set & canon_terms)
            abstract = str(df.at[idx, 'Abstract']).lower() if has_abstract else ''
            in_abs = any(t in abstract for t in terms)
            if in_kw:
                kw_docs.add(idx)
            if in_abs:
                abs_docs.add(idx)
                if years is not None and pd.notna(years.at[idx]):
                    yr = int(years.at[idx])
                    per_year[yr] += 1
                    if yr >= 2022:
                        recent_abs.add(idx)
        network_terms = sorted(kw_node_terms & canon_terms)
        out['families'][fam] = {
            'n_docs_author_keyword': len(kw_docs),
            'n_docs_abstract': len(abs_docs),
            'n_docs_abstract_2022plus': len(recent_abs),
            'in_keyword_network': bool(network_terms),
            'network_terms': network_terms,
            'abstract_by_year': dict(sorted(per_year.items())),
        }
    return out


class TheoryOperationalisation:
    """Operationalises Rogers' (2003) diffusion-of-innovations attributes and
    TAM constructs as keyword-set proxies measurable from the Scopus corpus.

    Closes R2.B by converting decorative post-hoc theory invocation into
    testable a priori predictions:
        H1: platforms with higher Rogers-score have higher degree centrality
            in cosco5;
        H2: Rogers-score correlates with avg publication year
            (later attributes reach later eras);
        H3: TAM perceived-usefulness proxy correlates with citation count.
    """

    ROGERS_ATTRIBUTES: Dict[str, set] = {
        'relative_advantage': {
            'free', 'open source', 'open-source', 'cross-platform', 'cross platform',
            'multiplatform', 'multi-platform', 'performance', 'powerful', 'efficient',
            'real-time', 'real time',
        },
        'compatibility': {
            'interoperable', 'interoperability', 'standard', 'standards', 'standardisation',
            'api', 'plugin', 'integration', 'compatible', 'pipeline', 'workflow',
        },
        'low_complexity': {  # Negated: LOW complexity is the positive Rogers attribute.
            'easy', 'tutorial', 'tutorials', 'beginner', 'intuitive', 'simple',
            'accessible', 'documentation', 'usability', 'novice',
        },
        'trialability': {
            'trial', 'demo', 'demonstration', 'sample', 'example', 'open source',
            'community', 'free', 'tutorial',
        },
        'observability': {
            'deployed', 'released', 'shipped', 'showcase', 'published', 'production',
            'industry', 'commercial', 'professional',
        },
    }

    TAM_CONSTRUCTS: Dict[str, set] = {
        'perceived_usefulness': {
            'productive', 'efficient', 'useful', 'effective', 'beneficial',
            'powerful', 'professional', 'industry',
        },
        'perceived_ease_of_use': {
            'easy', 'intuitive', 'accessible', 'simple', 'beginner', 'novice',
            'tutorial',
        },
    }

    DEFAULT_PLATFORMS = (
        'unity', 'unreal', 'unreal engine', 'godot', 'gamemaker', 'construct',
        'roblox', 'cocos2d',
    )

    # Single-word platform names that collide with common English words are matched
    # by a stricter, engine-specific pattern instead of a bare word boundary, so the
    # English verb/noun "construct" is not counted as the Construct game engine
    # (reviewer R2 major #1). Match requires a version number, "engine", or the
    # vendor name "Scirra".
    _PLATFORM_PATTERNS = {
        'construct': re.compile(r'\bconstruct\s*(?:2|3|classic|engine|game\s+engine)\b'
                                r'|\bscirra\b'),
    }

    def __init__(self, df: pd.DataFrame,
                 keyword_cols: Tuple[str, ...] = ('Author Keywords', 'Index Keywords'),
                 include_abstract: bool = True):
        self.df = df
        self.keyword_cols = keyword_cols
        # When False, proxies are drawn from keyword columns only (a robustness
        # variant that isolates author/index keywords from abstract prose).
        self.include_abstract = include_abstract

    def _doc_keyword_text(self, idx: int) -> str:
        """Concatenate configured keyword columns (and optionally the abstract)."""
        parts = []
        for col in self.keyword_cols:
            if col in self.df.columns:
                v = self.df.at[idx, col] if idx in self.df.index else None
                if isinstance(v, str):
                    parts.append(v)
        # Abstract adds coverage but dilutes keyword-level specificity; optional.
        if self.include_abstract and 'Abstract' in self.df.columns:
            v = self.df.at[idx, 'Abstract'] if idx in self.df.index else None
            if isinstance(v, str):
                parts.append(v)
        return ' '.join(parts).lower()

    def docs_mentioning(self, platform: str) -> List[int]:
        """Return doc-ids whose keyword set (and optionally abstract) mentions the platform.

        Uses word-boundary matching, except ambiguous single-word platforms use an
        engine-specific override (``_PLATFORM_PATTERNS``) so the English verb/noun
        "construct" is not counted as the Construct game engine (reviewer R2 major #1).
        Multi-word platforms (`unreal engine`) require the exact phrase.
        """
        platform_lc = platform.lower().strip()
        pattern = self._PLATFORM_PATTERNS.get(platform_lc)
        if pattern is None:
            if ' ' in platform_lc:
                pattern = re.compile(re.escape(platform_lc))
            else:
                pattern = re.compile(r'\b' + re.escape(platform_lc) + r'\b')
        out = []
        for idx in self.df.index:
            txt = self._doc_keyword_text(idx)
            if pattern.search(txt):
                out.append(int(idx))
        return out

    def attribute_score(self, doc_ids: List[int],
                        attribute_terms: set) -> float:
        """Mean fraction of attribute terms appearing across the doc subset.

        Returns a value in [0, 1]: 1.0 = every doc has every attribute term,
        0.0 = no doc has any.
        """
        if not doc_ids:
            return 0.0
        scores = []
        for d in doc_ids:
            txt = self._doc_keyword_text(d)
            hits = sum(1 for term in attribute_terms if term in txt)
            scores.append(hits / max(len(attribute_terms), 1))
        return float(np.mean(scores))

    def score_platforms(self,
                        platforms: Tuple[str, ...] = DEFAULT_PLATFORMS) -> Dict[str, Dict[str, float]]:
        """Per-platform Rogers and TAM scores."""
        out: Dict[str, Dict[str, float]] = {}
        for p in platforms:
            doc_ids = self.docs_mentioning(p)
            if not doc_ids:
                continue
            scores = {'n_docs': len(doc_ids)}
            for attr, terms in self.ROGERS_ATTRIBUTES.items():
                scores[f'rogers_{attr}'] = self.attribute_score(doc_ids, terms)
            scores['rogers_total'] = sum(
                scores[f'rogers_{a}'] for a in self.ROGERS_ATTRIBUTES
            )
            for c, terms in self.TAM_CONSTRUCTS.items():
                scores[f'tam_{c}'] = self.attribute_score(doc_ids, terms)
            # Side metrics for hypothesis tests.
            scores['avg_pub_year'] = float(self.df.loc[doc_ids, 'Year'].mean()) \
                if 'Year' in self.df.columns else float('nan')
            scores['avg_citations'] = float(self.df.loc[doc_ids, 'Cited by'].mean()) \
                if 'Cited by' in self.df.columns else float('nan')
            out[p.lower()] = scores
        return out

    @staticmethod
    def _spearman_rho(xs: List[float], ys: List[float]) -> Tuple[float, int]:
        """Spearman ρ via rank-correlation, returns (rho, n). Pure NumPy."""
        if len(xs) != len(ys) or len(xs) < 3:
            return 0.0, len(xs)
        x_rank = pd.Series(xs).rank().to_numpy()
        y_rank = pd.Series(ys).rank().to_numpy()
        if np.std(x_rank) == 0 or np.std(y_rank) == 0:
            return 0.0, len(xs)
        rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
        return rho, len(xs)

    def test_h1_rogers_vs_centrality(self,
                                     graph: nx.Graph,
                                     platforms: Tuple[str, ...] = DEFAULT_PLATFORMS,
                                     ) -> Dict[str, Any]:
        """H1: platforms with higher total Rogers score have higher degree centrality.

        Centrality is read from the platform node label in `graph` (cosco5).
        """
        scores = self.score_platforms(platforms)
        centralities = nx.degree_centrality(graph)
        # Match graph nodes by label (case-insensitive).
        label_to_node = {}
        for n, data in graph.nodes(data=True):
            label_to_node[data.get('label', '').lower()] = n

        pairs = []
        for plat, sc in scores.items():
            node = label_to_node.get(plat)
            if node is None:
                continue
            pairs.append((sc['rogers_total'], centralities.get(node, 0.0), plat))

        if len(pairs) < 3:
            return {'rho': float('nan'), 'n': len(pairs), 'pairs': pairs,
                    'note': 'Too few platforms with corpus mentions AND graph nodes for correlation.'}
        rogers, cent, _ = zip(*pairs)
        rho, n = self._spearman_rho(list(rogers), list(cent))
        return {'rho': rho, 'n': n, 'pairs': pairs,
                'note': 'H1 supported if ρ > 0.5 with positive sign.'}

    def test_h2_rogers_vs_year(self,
                               platforms: Tuple[str, ...] = DEFAULT_PLATFORMS,
                               ) -> Dict[str, Any]:
        """H2: Rogers-score correlates with avg publication year of platform-related docs.

        Direction-of-effect prediction: late-emerging features (observability,
        compatibility) should cluster with later avg-year platforms; early-stage
        features (trialability, low_complexity) with earlier years.
        """
        scores = self.score_platforms(platforms)
        if len(scores) < 3:
            return {'rho': float('nan'), 'n': len(scores)}
        rogers = [s['rogers_total'] for s in scores.values()]
        years = [s['avg_pub_year'] for s in scores.values()]
        rho, n = self._spearman_rho(rogers, years)
        return {'rho': rho, 'n': n, 'platforms': list(scores.keys())}

    def test_h3_tam_vs_citations(self,
                                 platforms: Tuple[str, ...] = DEFAULT_PLATFORMS,
                                 ) -> Dict[str, Any]:
        """H3: perceived-usefulness proxy correlates with citation count."""
        scores = self.score_platforms(platforms)
        if len(scores) < 3:
            return {'rho': float('nan'), 'n': len(scores)}
        pu = [s['tam_perceived_usefulness'] for s in scores.values()]
        cites = [s['avg_citations'] for s in scores.values()]
        rho, n = self._spearman_rho(pu, cites)
        return {'rho': rho, 'n': n, 'platforms': list(scores.keys())}

    @staticmethod
    def variant_hypothesis_tests(df: pd.DataFrame, graph: nx.Graph,
                                 platforms: Tuple[str, ...] = DEFAULT_PLATFORMS
                                 ) -> Dict[str, Any]:
        """Re-run H1-H3 under three proxy sources to test whether the nulls are an
        artefact of one keyword source (reviewer R2 major #1): author keywords only,
        index keywords only, and keywords+abstract.
        """
        configs = {
            'author_keywords_only': dict(keyword_cols=('Author Keywords',),
                                         include_abstract=False),
            'index_keywords_only': dict(keyword_cols=('Index Keywords',),
                                        include_abstract=False),
            'keywords_plus_abstract': dict(
                keyword_cols=('Author Keywords', 'Index Keywords'),
                include_abstract=True),
        }
        out: Dict[str, Any] = {}
        for name, kw in configs.items():
            t = TheoryOperationalisation(df, **kw)
            out[name] = {
                'h1': t.test_h1_rogers_vs_centrality(graph, platforms),
                'h2': t.test_h2_rogers_vs_year(platforms),
                'h3': t.test_h3_tam_vs_citations(platforms),
                'n_platforms_detected': len(t.score_platforms(platforms)),
            }
        return out

    def to_json(self, graph: nx.Graph,
                platforms: Tuple[str, ...] = DEFAULT_PLATFORMS) -> Dict[str, Any]:
        """Serialise per-platform scores + the three hypothesis tests + variants."""
        return {
            'platform_scores': self.score_platforms(platforms),
            'h1_rogers_vs_centrality': self.test_h1_rogers_vs_centrality(graph, platforms),
            'h2_rogers_vs_year': self.test_h2_rogers_vs_year(platforms),
            'h3_tam_vs_citations': self.test_h3_tam_vs_citations(platforms),
            'proxy_source_variants': self.variant_hypothesis_tests(self.df, graph, platforms),
        }


class TemporalCouplingAnalyzer:
    """Bibliographic coupling stratified across technology eras.

    Closes R1.4: pre-Unity (≤2017), VR-emergence (2018–2021), post-genAI (2022–2026).
    The Scopus 'References' field is free-text (no DOIs in this corpus); we
    normalise refs to <surname>_<year> keys, which is approximate but sufficient
    for era-level coupling structure.
    """

    DEFAULT_ERAS = (
        ('pre-2018', None, 2017),
        ('2018-2021', 2018, 2021),
        ('2022-2026', 2022, 2026),
    )

    _YEAR_RE = re.compile(r'\((\d{4})\)')
    _DOI_RE = re.compile(r'10\.\d{4,9}/[^\s,;]+')

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def parse_references_field(self, raw: Optional[str]) -> List[str]:
        """Parse a Scopus References cell into a list of normalised reference keys.

        DOIs preferred when present. Otherwise falls back to surname_year.
        """
        if not raw or not isinstance(raw, str):
            return []
        keys = []
        for ref in raw.split(';'):
            ref = ref.strip()
            if not ref:
                continue
            # DOI first.
            doi_match = self._DOI_RE.search(ref)
            if doi_match:
                keys.append(doi_match.group(0).lower().rstrip('.,'))
                continue
            # Fallback: first comma-token = surname, last (YYYY) = year.
            year_match = self._YEAR_RE.search(ref)
            if not year_match:
                continue
            year = year_match.group(1)
            first_token = ref.split(',', 1)[0].strip().lower()
            # Reject obvious non-name tokens (very short or numeric).
            if not first_token or len(first_token) < 3 or first_token.isdigit():
                continue
            # Use just the surname-like first word (handles "van Eck" → "van eck").
            surname = first_token.split()[0]
            keys.append(f"{surname}_{year}")
        return keys

    def build_doc_to_refs(self) -> Dict[int, List[str]]:
        """Return mapping doc_id → list of normalised reference keys."""
        if 'References' not in self.df.columns:
            return {}
        return {
            int(idx): self.parse_references_field(raw)
            for idx, raw in self.df['References'].items()
        }

    def filter_to_era(self, year_min: Optional[int], year_max: Optional[int]) -> List[int]:
        """Return doc_ids within [year_min, year_max] (inclusive). None = unbounded."""
        if 'Year' not in self.df.columns:
            return []
        years = self.df['Year']
        mask = pd.Series(True, index=self.df.index)
        if year_min is not None:
            mask &= years >= year_min
        if year_max is not None:
            mask &= years <= year_max
        return [int(i) for i in self.df.index[mask].tolist()]

    def build_coupling_graph(self, doc_ids: List[int],
                             min_shared: int = 1) -> nx.Graph:
        """Build a doc–doc coupling graph; edge weight = number of shared references."""
        doc_refs = self.build_doc_to_refs()
        ref_to_docs: Dict[str, List[int]] = defaultdict(list)
        for d in doc_ids:
            for k in doc_refs.get(d, []):
                ref_to_docs[k].append(d)

        edge_weights: Dict[Tuple[int, int], int] = defaultdict(int)
        for ref, docs in ref_to_docs.items():
            docs = sorted(set(docs))
            for i, a in enumerate(docs):
                for b in docs[i + 1:]:
                    edge_weights[(a, b)] += 1

        G = nx.Graph()
        for d in doc_ids:
            G.add_node(d)
        for (a, b), w in edge_weights.items():
            if w >= min_shared:
                G.add_edge(a, b, weight=w)
        return G

    def per_era_coupling(self, eras=DEFAULT_ERAS) -> Dict[str, Dict[str, Any]]:
        """Build per-era coupling networks; return summary metrics for each era."""
        out = {}
        for label, ymin, ymax in eras:
            doc_ids = self.filter_to_era(ymin, ymax)
            G = self.build_coupling_graph(doc_ids)
            # Top-coupled documents (degree-weighted).
            degree = sorted(G.degree(weight='weight'), key=lambda x: -x[1])[:10]
            out[label] = {
                'n_docs': len(doc_ids),
                'n_coupling_edges': G.number_of_edges(),
                'density': nx.density(G) if G.number_of_nodes() > 1 else 0.0,
                'avg_clustering': (nx.average_clustering(G, weight='weight')
                                   if G.number_of_edges() > 0 else 0.0),
                'top_coupled_docs': degree,
                'doc_ids': doc_ids,
            }
        return out

    def compare_eras(self, eras=DEFAULT_ERAS) -> Dict[str, Any]:
        """Kendall's τ on top-coupled-document rank between consecutive eras.

        Operates on the *intersection* of document IDs that appear in both
        eras' coupling graphs (typically empty if eras are mutually exclusive
        by year — so we instead compare cited-reference overlap by era).
        """
        # For mutually-exclusive year eras the doc-IDs don't overlap, so we
        # compare *which references* are most cited in each era and the
        # rank-correlation of their citation counts.
        per_era_refs: Dict[str, Dict[str, int]] = {}
        doc_refs = self.build_doc_to_refs()
        for label, ymin, ymax in eras:
            counts: Dict[str, int] = defaultdict(int)
            for d in self.filter_to_era(ymin, ymax):
                for k in doc_refs.get(d, []):
                    counts[k] += 1
            per_era_refs[label] = dict(counts)

        # Take union of refs; build vector per era; compute pairwise Kendall τ.
        union_refs = sorted({r for c in per_era_refs.values() for r in c})
        labels = list(per_era_refs.keys())
        comparisons = {}
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                vec_a = [per_era_refs[a].get(r, 0) for r in union_refs]
                vec_b = [per_era_refs[b].get(r, 0) for r in union_refs]
                comparisons[f'{a}_vs_{b}'] = self._kendall_tau(vec_a, vec_b)
        return {
            'per_era_top_refs': {
                label: sorted(refs.items(), key=lambda x: -x[1])[:10]
                for label, refs in per_era_refs.items()
            },
            'kendall_tau': comparisons,
        }

    @staticmethod
    def _kendall_tau(x: List[int], y: List[int]) -> float:
        """Compute Kendall's τ-b correlation (handles ties). Pure Python."""
        n = len(x)
        if n < 2:
            return 0.0
        concordant = discordant = ties_x = ties_y = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    ties_x += 1
                    continue
                if dy == 0:
                    ties_y += 1
                    continue
                if (dx > 0) == (dy > 0):
                    concordant += 1
                else:
                    discordant += 1
        total_x = concordant + discordant + ties_x
        total_y = concordant + discordant + ties_y
        denom = (total_x * total_y) ** 0.5
        if denom == 0:
            return 0.0
        return (concordant - discordant) / denom

    def to_json(self, eras=DEFAULT_ERAS) -> Dict[str, Any]:
        """Full per-era + comparison serialisation."""
        return {
            'eras': self.per_era_coupling(eras),
            'comparison': self.compare_eras(eras),
        }


class LotkaAnalyzer:
    """Author-productivity power-law fit (Lotka 1926).

    The Lotka model predicts that the number of authors with $n$ publications
    follows $f(n) = c \\cdot n^{-\\alpha}$. The classical Lotka exponent is
    $\\alpha \\approx 2$ for many disciplines; the empirical exponent and its
    fit quality are diagnostic of how concentrated the field is.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def count_publications_per_author(self) -> Dict[str, int]:
        if 'Authors' not in self.df.columns:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for raw in self.df['Authors'].dropna():
            for a in str(raw).split(';'):
                a = a.strip()
                if a:
                    counts[a] += 1
        return dict(counts)

    def productivity_distribution(self) -> Dict[int, int]:
        """Returns {n_pubs: number_of_authors_with_that_count}."""
        per_author = self.count_publications_per_author()
        dist: Dict[int, int] = defaultdict(int)
        for n in per_author.values():
            dist[n] += 1
        return dict(dist)

    def fit_lotka(self) -> Dict[str, float]:
        """Log-log linear fit on the productivity distribution.

        Returns alpha (the power-law exponent), c (the leading constant),
        and R^2 of the log-log regression.
        """
        dist = self.productivity_distribution()
        if len(dist) < 2:
            return {'alpha': float('nan'), 'c': float('nan'), 'r_squared': float('nan'),
                    'n_authors': 0}
        ns = np.array(sorted(dist.keys()), dtype=float)
        fs = np.array([dist[int(n)] for n in ns], dtype=float)
        logn = np.log(ns)
        logf = np.log(fs)
        slope, intercept = np.polyfit(logn, logf, 1)
        alpha = -float(slope)
        c = float(np.exp(intercept))
        pred = slope * logn + intercept
        ss_res = float(np.sum((logf - pred) ** 2))
        ss_tot = float(np.sum((logf - logf.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            'alpha': alpha, 'c': c, 'r_squared': float(r2),
            'n_authors': sum(dist.values()),
            'productivity_distribution': dict(sorted(dist.items())),
        }

    def top_authors(self, n: int = 10) -> List[Tuple[str, int]]:
        per_author = self.count_publications_per_author()
        return sorted(per_author.items(), key=lambda x: -x[1])[:n]

    def to_json(self) -> Dict[str, Any]:
        out = self.fit_lotka()
        out['top_authors'] = self.top_authors(15)
        return out


class BradfordAnalyzer:
    """Bradford's law of scattering (Bradford 1934).

    Sources are sorted by document count descending and partitioned into
    ``n_zones`` cumulative buckets each containing ~equal documents; the
    ratio of source counts between successive zones gives the Bradford
    multiplier $k$.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def source_counts(self) -> Dict[str, int]:
        if 'Source title' not in self.df.columns:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for s in self.df['Source title'].dropna():
            counts[str(s).strip()] += 1
        return dict(counts)

    def compute_zones(self, n_zones: int = 3) -> Dict[str, Any]:
        counts = self.source_counts()
        if not counts:
            return {'zones': [], 'multiplier_k': float('nan')}
        ranked = sorted(counts.items(), key=lambda x: -x[1])
        total_docs = sum(c for _, c in ranked)
        target = total_docs / n_zones

        zones: List[Dict[str, Any]] = []
        running = 0
        zone_sources: List[Tuple[str, int]] = []
        for src, c in ranked:
            zone_sources.append((src, c))
            running += c
            if running >= target * (len(zones) + 1) and len(zones) < n_zones - 1:
                zones.append({
                    'zone': len(zones) + 1,
                    'n_sources': len(zone_sources),
                    'n_docs': running - sum(z['n_docs'] for z in zones),
                    'sources': zone_sources[:25],  # cap stored detail
                })
                zone_sources = []
        if zone_sources:
            zones.append({
                'zone': len(zones) + 1,
                'n_sources': len(zone_sources),
                'n_docs': running - sum(z['n_docs'] for z in zones),
                'sources': zone_sources[:25],
            })

        # Bradford multiplier — geometric mean of consecutive zone-source ratios.
        if len(zones) >= 2:
            ratios = [zones[i + 1]['n_sources'] / max(zones[i]['n_sources'], 1)
                      for i in range(len(zones) - 1)]
            k = float(np.exp(np.mean(np.log(ratios)))) if all(r > 0 for r in ratios) else float('nan')
        else:
            k = float('nan')

        return {
            'zones': zones,
            'multiplier_k': k,
            'total_sources': len(ranked),
            'total_docs': total_docs,
            'top_sources': ranked[:15],
        }

    def to_json(self, n_zones: int = 3) -> Dict[str, Any]:
        return self.compute_zones(n_zones)


class ThematicMapAnalyzer:
    """Callon centrality × density quadrant map (Callon et al. 1991).

    For each cluster in the keyword network:
      centrality = sum of weights of edges *between* this cluster and others;
      density   = sum of weights of edges *inside* the cluster, divided by
                  the number of nodes (averaging internal cohesion).

    Quadrants (median split):
      motor             — high centrality, high density (well-developed central themes)
      basic/transversal — high centrality, low density  (central but not yet cohesive)
      niche             — low centrality, high density  (cohesive but isolated)
      emerging/declining — low centrality, low density   (peripheral)
    """

    QUADRANTS = ('motor', 'basic_transversal', 'niche', 'emerging_or_declining')

    def __init__(self, graph: nx.Graph, partition: Optional[List[set]] = None,
                 seed: int = RANDOM_SEED):
        self.graph = graph
        if partition is None and graph.number_of_nodes() >= 3:
            partition = community.louvain_communities(
                graph, weight='weight', resolution=1.0, seed=seed,
            )
        self.partition: List[set] = partition or []

    def compute_quadrant_metrics(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.partition:
            return out
        node_to_cluster: Dict[Any, int] = {}
        for cid, members in enumerate(self.partition):
            for n in members:
                node_to_cluster[n] = cid

        for cid, members in enumerate(self.partition):
            internal = 0.0
            external = 0.0
            for u in members:
                for v in self.graph.neighbors(u):
                    w = self.graph[u][v].get('weight', 1)
                    if v in members:
                        internal += w / 2  # avoid double-count
                    else:
                        external += w
            density = internal / max(len(members), 1)
            # Secondary key on the label breaks occurrence ties deterministically
            # (member iteration is set-ordered → PYTHONHASHSEED-dependent otherwise).
            label_keywords = sorted(
                ((n, self.graph.nodes[n].get('occurrences', 0)) for n in members),
                key=lambda x: (-x[1], x[0]),
            )[:5]
            out.append({
                'cluster_id': cid,
                'n_nodes': len(members),
                'centrality': external,
                'density': density,
                'top_keywords': label_keywords,
            })
        return out

    def assign_quadrants(self) -> List[Dict[str, Any]]:
        metrics = self.compute_quadrant_metrics()
        if not metrics:
            return metrics
        cent_med = float(np.median([m['centrality'] for m in metrics]))
        dens_med = float(np.median([m['density'] for m in metrics]))
        for m in metrics:
            high_cent = m['centrality'] >= cent_med
            high_dens = m['density'] >= dens_med
            if high_cent and high_dens:
                m['quadrant'] = 'motor'
            elif high_cent and not high_dens:
                m['quadrant'] = 'basic_transversal'
            elif (not high_cent) and high_dens:
                m['quadrant'] = 'niche'
            else:
                m['quadrant'] = 'emerging_or_declining'
        return metrics

    def to_json(self) -> Dict[str, Any]:
        clusters = self.assign_quadrants()
        return {
            'clusters': clusters,
            'centrality_median': float(np.median([c['centrality'] for c in clusters])) if clusters else float('nan'),
            'density_median': float(np.median([c['density'] for c in clusters])) if clusters else float('nan'),
        }


class TrendTopicsAnalyzer:
    """Keyword frequency over time + emergence index.

    Closes the bibliometrix-standard ``trendTopics`` analysis: for each top
    keyword, report counts per year; emergence index compares the most recent
    ``window`` years to the preceding ``window``.
    """

    def __init__(self, df: pd.DataFrame, keyword_col: str = 'Author Keywords'):
        self.df = df
        self.keyword_col = keyword_col

    def frequency_by_year(self, top_n: int = 25) -> Dict[str, Any]:
        if self.keyword_col not in self.df.columns or 'Year' not in self.df.columns:
            return {'years': [], 'keywords': [], 'matrix': []}

        # Pre-pass: keyword totals.
        totals: Dict[str, int] = defaultdict(int)
        per_year: Dict[Tuple[str, int], int] = defaultdict(int)
        for _, row in self.df.iterrows():
            year = row.get('Year')
            if pd.isna(year):
                continue
            year = int(year)
            raw = row.get(self.keyword_col)
            if not isinstance(raw, str):
                continue
            for kw in raw.split(';'):
                kw = kw.strip().lower()
                if not kw:
                    continue
                totals[kw] += 1
                per_year[(kw, year)] += 1

        top_kw = [kw for kw, _ in sorted(totals.items(), key=lambda x: -x[1])[:top_n]]
        years = sorted({y for (_, y) in per_year.keys()})
        matrix = [[per_year.get((kw, y), 0) for y in years] for kw in top_kw]
        return {'years': years, 'keywords': top_kw, 'matrix': matrix}

    def emergence_index(self, top_n: int = 25, window: int = 3) -> List[Dict[str, Any]]:
        freq = self.frequency_by_year(top_n=top_n)
        if not freq['years']:
            return []
        years = freq['years']
        recent = set(years[-window:])
        prior = set(years[-2 * window:-window]) if len(years) >= 2 * window else set()
        out = []
        for kw, row in zip(freq['keywords'], freq['matrix']):
            r = sum(c for y, c in zip(years, row) if y in recent)
            p = sum(c for y, c in zip(years, row) if y in prior)
            ei = (r - p) / max(p, 1)
            out.append({'keyword': kw, 'recent': r, 'prior': p, 'emergence_index': ei})
        out.sort(key=lambda x: -x['emergence_index'])
        return out

    def to_json(self, top_n: int = 25, window: int = 3) -> Dict[str, Any]:
        return {
            'frequency_by_year': self.frequency_by_year(top_n),
            'emergence': self.emergence_index(top_n, window),
        }


class ThreeFieldPlotAnalyzer:
    """Sankey-style three-field plot of (Source × Author × Keyword) flows.

    For each (source, keyword) pair we count the number of corpus documents
    where both occur; similarly (author, keyword). The top-``top_each``
    entities per axis are kept; flows weight is document-count.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _split_field(self, field: Any) -> List[str]:
        if not isinstance(field, str):
            return []
        return [s.strip() for s in field.split(';') if s.strip()]

    def compute_flows(self, top_each: int = 10) -> Dict[str, Any]:
        if not all(c in self.df.columns for c in ['Source title', 'Authors']):
            return {}

        # Top entities per axis.
        src_counts = self.df['Source title'].value_counts().head(top_each)
        top_sources = list(src_counts.index)
        author_counts: Dict[str, int] = defaultdict(int)
        for raw in self.df['Authors'].dropna():
            for a in self._split_field(raw):
                author_counts[a] += 1
        top_authors = [a for a, _ in sorted(author_counts.items(), key=lambda x: -x[1])[:top_each]]
        kw_counts: Dict[str, int] = defaultdict(int)
        for raw in self.df['Author Keywords'].dropna() if 'Author Keywords' in self.df.columns else []:
            for k in self._split_field(raw):
                kw_counts[k.lower()] += 1
        top_keywords = [k for k, _ in sorted(kw_counts.items(), key=lambda x: -x[1])[:top_each]]

        # Bipartite edges.
        src_kw: Dict[Tuple[str, str], int] = defaultdict(int)
        au_kw: Dict[Tuple[str, str], int] = defaultdict(int)
        for _, row in self.df.iterrows():
            src = row.get('Source title')
            authors = self._split_field(row.get('Authors'))
            kws = [k.lower() for k in self._split_field(row.get('Author Keywords'))]
            if isinstance(src, str) and src in top_sources:
                for k in kws:
                    if k in top_keywords:
                        src_kw[(src, k)] += 1
            for a in authors:
                if a in top_authors:
                    for k in kws:
                        if k in top_keywords:
                            au_kw[(a, k)] += 1

        return {
            'top_sources': [(s, int(src_counts[s])) for s in top_sources],
            'top_authors': [(a, author_counts[a]) for a in top_authors],
            'top_keywords': [(k, kw_counts[k]) for k in top_keywords],
            'source_keyword_flows': sorted(
                ({'source': s, 'keyword': k, 'weight': w} for (s, k), w in src_kw.items()),
                key=lambda x: -x['weight'],
            )[:60],
            'author_keyword_flows': sorted(
                ({'author': a, 'keyword': k, 'weight': w} for (a, k), w in au_kw.items()),
                key=lambda x: -x['weight'],
            )[:60],
        }

    def to_json(self, top_each: int = 10) -> Dict[str, Any]:
        return self.compute_flows(top_each)


class Visualizer:
    """Visualization generator for bibliometric analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_network(self, network: VOSViewerNetwork, filename: str,
                     figsize: Tuple[int, int] = (12, 10)) -> str:
        """Plot network visualization."""
        if network.graph is None or network.graph.number_of_nodes() == 0:
            return ""

        G = network.graph

        fig, ax = plt.subplots(figsize=figsize)

        # Get positions from VOSViewer coordinates
        pos = {node_id: (data['x'], data['y'])
               for node_id, data in G.nodes(data=True)}

        # Get cluster colors
        clusters = [data.get('cluster', 0) for node_id, data in G.nodes(data=True)]
        unique_clusters = sorted(set(clusters))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        cluster_colors = {c: colors[i] for i, c in enumerate(unique_clusters)}
        node_colors = [cluster_colors[c] for c in clusters]

        # Node sizes based on occurrences
        sizes = [data.get('occurrences', 1) * 50 + 100
                 for node_id, data in G.nodes(data=True)]

        # Edge weights for width
        edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]

        # Draw network
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes,
                               alpha=0.8, ax=ax)

        # Add labels for larger nodes
        labels = {node_id: data['label']
                  for node_id, data in G.nodes(data=True)
                  if data.get('occurrences', 0) >= 3}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        ax.set_title(f"{network.analysis_type}: {network.unit_of_analysis}\n"
                     f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        ax.axis('off')

        # Add legend for clusters
        for cluster_id, color in cluster_colors.items():
            ax.scatter([], [], c=[color], s=100, label=f"Cluster {cluster_id}")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_publication_timeline(self, df_years: pd.DataFrame, filename: str,
                                  figsize: Tuple[int, int] = (12, 5),
                                  completeness: Optional[Dict[str, float]] = None) -> str:
        """Plot publication timeline (annual + cumulative with Gompertz fit).

        ``completeness`` is the dict returned by ``ScopusAnalyzer.compute_field_completeness_score``.
        When provided, the right panel overlays linear / exponential / Gompertz
        fits with R² annotations to support the R1.3 corpus-completeness claim.
        """
        if df_years.empty:
            return ""

        df_sorted = df_years.sort_values('Year').reset_index(drop=True)
        years = df_sorted['Year'].to_numpy(dtype=float)
        counts = df_sorted['count'].to_numpy(dtype=float)
        cum = counts.cumsum()

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax_left, ax_right = axes

        # Left: annual bars + linear trend.
        ax_left.bar(years, counts, color='steelblue', edgecolor='white')
        ax_left.set_xlabel('Year', fontsize=12)
        ax_left.set_ylabel('Number of Publications', fontsize=12)
        ax_left.set_title('Annual Publication Volume', fontsize=13)
        slope, intercept = np.polyfit(years, counts, 1)
        ax_left.plot(years, slope * years + intercept, "r--", alpha=0.8,
                     label=f"Linear fit (slope={slope:.2f}/yr)")
        ax_left.legend(loc='upper left')

        # Right: cumulative curve + Gompertz overlay (R1.3 saturation diagnostic).
        ax_right.scatter(years, cum, color='steelblue', s=40, label='Observed cumulative')
        ax_right.set_xlabel('Year', fontsize=12)
        ax_right.set_ylabel('Cumulative Publications', fontsize=12)
        ax_right.set_title('Cumulative Volume + Saturation Fits', fontsize=13)

        if completeness and not np.isnan(completeness.get('gompertz_r2', np.nan)):
            t = years - years.min()
            t_smooth = np.linspace(0, t.max() + 5, 200)
            yr_smooth = years.min() + t_smooth

            from scipy.optimize import curve_fit

            def gompertz(x, K, b, c):
                return K * np.exp(-b * np.exp(-c * x))

            try:
                K_guess = max(cum[-1] * 1.5, 100.0)
                g_p, _ = curve_fit(gompertz, t, cum,
                                   p0=(K_guess, 5.0, 0.2), maxfev=20000,
                                   bounds=([cum[-1], 0.1, 0.01],
                                           [cum[-1] * 100, 50.0, 5.0]))
                ax_right.plot(yr_smooth, gompertz(t_smooth, *g_p), 'g-', alpha=0.85,
                              label=f"Gompertz (R²={completeness['gompertz_r2']:.3f}, "
                                    f"asymptote≈{g_p[0]:.0f})")
            except Exception:
                pass

        # Overlay the linear cumulative fit too.
        cum_lin = np.polyfit(years, cum, 1)
        ax_right.plot(years, np.polyval(cum_lin, years), 'r--', alpha=0.6,
                      label=(f"Linear cumulative (R²="
                             f"{completeness['linear_r2']:.3f})"
                             if completeness else "Linear cumulative"))
        ax_right.legend(loc='upper left')

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_cluster_heatmap(self, networks: Dict[int, VOSViewerNetwork],
                             filename: str, figsize: Tuple[int, int] = (14, 8)) -> str:
        """Plot heatmap of cluster sizes across networks."""
        data = []

        for num, network in networks.items():
            if network.graph:
                cluster_counts = defaultdict(int)
                for _, node_data in network.graph.nodes(data=True):
                    cluster_counts[node_data.get('cluster', 0)] += 1

                for cluster, count in cluster_counts.items():
                    data.append({
                        'Network': f"cosco{num}",
                        'Cluster': f"C{cluster}",
                        'Count': count
                    })

        if not data:
            return ""

        df = pd.DataFrame(data)
        pivot = df.pivot_table(values='Count', index='Network', columns='Cluster', fill_value=0)

        fig, ax = plt.subplots(figsize=figsize)
        # Use 'g' format for general numbers to handle both int and float
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        ax.set_title('Cluster Distribution Across Networks', fontsize=14)

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def plot_keyword_wordcloud_alt(self, keywords: Dict[str, int], filename: str,
                                   figsize: Tuple[int, int] = (12, 8)) -> str:
        """Plot keyword frequency as bar chart (wordcloud alternative)."""
        if not keywords:
            return ""

        # Get top 20 keywords
        top_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20])

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(top_keywords))
        ax.barh(y_pos, list(top_keywords.values()), color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(top_keywords.keys()))
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title('Top 20 Author Keywords', fontsize=14)

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.pdf"
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)


class ResultsExporter:
    """Export analysis results for LaTeX integration."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_network_summary(self, networks: Dict[int, VOSViewerNetwork],
                               filename: str = "network_summary.json") -> str:
        """Export network summary to JSON."""
        summary = {}

        for num, network in networks.items():
            analyzer = NetworkAnalyzer(network)
            metrics = analyzer.calculate_network_metrics()
            cluster_summary = analyzer.get_cluster_summary()

            summary[f"cosco{num}"] = {
                'analysis_type': network.analysis_type,
                'unit_of_analysis': network.unit_of_analysis,
                'network_metrics': asdict(metrics),
                'cluster_summary': {
                    str(k): {
                        'num_nodes': v['num_nodes'],
                        'keywords': v['keywords'],
                        'total_occurrences': v['total_occurrences'],
                        'avg_year': v['avg_year'],
                        'top_keywords': [(kw, occ) for kw, occ in v['top_keywords']]
                    } for k, v in cluster_summary.items()
                }
            }

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_node_metrics(self, networks: Dict[int, VOSViewerNetwork],
                            filename: str = "node_metrics.csv") -> str:
        """Export node-level metrics to CSV."""
        all_metrics = []

        for num, network in networks.items():
            analyzer = NetworkAnalyzer(network)
            metrics = analyzer.calculate_node_metrics()

            for m in metrics:
                all_metrics.append({
                    'network': f"cosco{num}",
                    'analysis_type': network.analysis_type,
                    **asdict(m)
                })

        df = pd.DataFrame(all_metrics)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        return str(output_path)

    def export_latex_table(self, df: pd.DataFrame, filename: str,
                           caption: str = "", label: str = "") -> str:
        """Export DataFrame as LaTeX table."""
        latex = df.to_latex(index=False, escape=True, float_format="%.2f")

        # Add caption and label
        if caption or label:
            latex = latex.replace("\\begin{tabular}",
                                  f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
            latex = f"\\begin{{table}}[htbp]\n\\centering\n{latex}\\end{{table}}"

        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)

        return str(output_path)


def run_full_analysis(data_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run complete bibliometric analysis pipeline."""
    _set_global_seed(RANDOM_SEED)

    data_path = Path(data_dir)

    results = {
        'networks': {},
        'scopus': {},
        'figures': [],
        'tables': []
    }

    print("=" * 60)
    print("BIBLIOMETRIC ANALYSIS PIPELINE")
    print(f"  Random seed: {RANDOM_SEED}")
    print("=" * 60)

    # 1. Parse VOSViewer networks
    print("\n[1/4] Parsing VOSViewer networks...")
    parser = VOSViewerParser(
        map_dir=str(data_path / "map"),
        net_dir=str(data_path / "net")
    )
    networks = parser.parse_all_networks()
    results['networks'] = networks

    # 2. Analyze Scopus data
    print("\n[2/4] Analyzing Scopus data...")
    scopus_files = list(data_path.glob("scopus_export*.csv"))
    scopus = None
    if scopus_files:
        # Prefer the de-biased export when available; the original biased query
        # is retained on disk only as a sensitivity-test reference (R2.A case study).
        primary = next((p for p in scopus_files if 'debiased' in p.name.lower()),
                       scopus_files[0])
        scopus = ScopusAnalyzer(str(primary))
        results['scopus'] = {
            'stats': scopus.get_basic_stats(),
            'publications_by_year': scopus.get_publications_by_year(),
            'top_authors': scopus.get_top_authors(),
            'top_sources': scopus.get_top_sources(),
            'keywords': scopus.get_keyword_frequency(),
            'completeness': scopus.compute_field_completeness_score(),
        }
        print(f"  Loaded {results['scopus']['stats']['total_documents']} documents from {primary.name}")
        print(f"  Field completeness: linear R²={results['scopus']['completeness']['linear_r2']:.3f}, "
              f"Gompertz R²={results['scopus']['completeness']['gompertz_r2']:.3f}, "
              f"saturation={results['scopus']['completeness']['saturation_ratio']:.3f}")

    # 3. Generate visualizations
    print("\n[3/4] Generating visualizations...")
    viz = Visualizer(output_dir)

    # Network visualizations for key networks
    for num in [5, 6, 7, 17]:  # Keyword, Citation, Co-citation
        if num in networks:
            path = viz.plot_network(networks[num], f"network_cosco{num}")
            if path:
                results['figures'].append(path)
                print(f"  Created: network_cosco{num}.pdf")

    # Publication timeline (with Gompertz saturation diagnostic).
    if 'publications_by_year' in results['scopus']:
        path = viz.plot_publication_timeline(
            results['scopus']['publications_by_year'],
            "publication_timeline",
            completeness=results['scopus'].get('completeness'),
        )
        if path:
            results['figures'].append(path)
            print(f"  Created: publication_timeline.pdf")

    # Cluster heatmap
    path = viz.plot_cluster_heatmap(networks, "cluster_heatmap")
    if path:
        results['figures'].append(path)
        print(f"  Created: cluster_heatmap.pdf")

    # Keyword frequency
    if 'keywords' in results['scopus']:
        path = viz.plot_keyword_wordcloud_alt(
            results['scopus']['keywords'],
            "keyword_frequency"
        )
        if path:
            results['figures'].append(path)
            print(f"  Created: keyword_frequency.pdf")

    # 3b. New robustness analyses (R1.2, R1.4, R1.5).
    if scopus is not None:
        print("\n[3b/4] Running robustness analyses (sensitivity / temporal / affiliation)...")
        processed_dir = Path(data_dir) / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Affiliation bias (G5).
        try:
            ab = AffiliationBiasAnalyzer(scopus.df)
            ab_payload = ab.to_json()
            (processed_dir / 'affiliation_matrix.json').write_text(
                json.dumps(ab_payload, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8',
            )
            results['affiliation'] = {
                'hhi_country': ab_payload['hhi_country'],
                'hhi_institution': ab_payload['hhi_institution'],
                'gini_country': ab_payload['gini_country'],
                'gini_institution': ab_payload['gini_institution'],
                'unique_countries': ab_payload['unique_countries'],
                'unique_institutions': ab_payload['unique_institutions'],
            }
            print(f"  Created: affiliation_matrix.json (HHI country={ab_payload['hhi_country']:.3f})")
        except Exception as exc:
            warnings.warn(f"AffiliationBiasAnalyzer failed: {exc}")

        # Temporal coupling (G4).
        try:
            tc = TemporalCouplingAnalyzer(scopus.df)
            tc_payload = tc.to_json()
            (processed_dir / 'coupling_eras.json').write_text(
                json.dumps(tc_payload, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8',
            )
            results['temporal_coupling'] = {
                era: {
                    'n_docs': info['n_docs'],
                    'n_coupling_edges': info['n_coupling_edges'],
                    'density': info['density'],
                }
                for era, info in tc_payload['eras'].items()
            }
            print(f"  Created: coupling_eras.json")
        except Exception as exc:
            warnings.warn(f"TemporalCouplingAnalyzer failed: {exc}")

        # 2026 partial-year robustness (reviewers R1.3 / R2.2): re-fit growth and
        # temporal coupling on the corpus capped at 2025 and report the deltas.
        try:
            scopus_2025 = ScopusAnalyzer(str(primary), year_max=2025)
            comp_full = results['scopus'].get('completeness', {})
            comp_2025 = scopus_2025.compute_field_completeness_score()
            tc_2025 = TemporalCouplingAnalyzer(scopus_2025.df).to_json(
                eras=(('pre-2018', None, 2017), ('2018-2021', 2018, 2021),
                      ('2022-2025', 2022, 2025)))
            growth_keys = ('best_model', 'gompertz_r2', 'bilogistic_r2', 'saturation_ratio',
                           'gompertz_asymptote', 'bilogistic_asymptote',
                           'delta_aic_bilogistic_minus_gompertz')
            n_2026 = int((scopus.df['Year'] == 2026).sum()) if 'Year' in scopus.df.columns else 0
            robustness = {
                'n_docs_full': int(len(scopus.df)),
                'n_docs_no2026': int(len(scopus_2025.df)),
                'n_2026_dropped': n_2026,
                'growth_full': {k: comp_full.get(k) for k in growth_keys},
                'growth_no2026': {k: comp_2025.get(k) for k in growth_keys},
                'temporal_coupling_no2026': {
                    era: {'n_docs': info['n_docs'],
                          'n_coupling_edges': info['n_coupling_edges'],
                          'density': info['density']}
                    for era, info in tc_2025['eras'].items()},
                'kendall_tau_no2026': tc_2025['comparison']['kendall_tau'],
            }
            (processed_dir / 'robustness_no2026.json').write_text(
                json.dumps(robustness, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8')
            results['robustness_no2026'] = robustness
            print(f"  Created: robustness_no2026.json (dropped {n_2026} 2026 docs; "
                  f"best_model full={comp_full.get('best_model')} / "
                  f"no2026={comp_2025.get('best_model')})")
        except Exception as exc:
            warnings.warn(f"2026 robustness variant failed: {exc}")

        # Build the keyword co-occurrence network directly from the corpus
        # (the VOSViewer cosco5 export reflects only the 88-doc corpus).
        kw_graph = SensitivityAnalyzer.from_corpus_keywords(scopus.df, min_occ=3)
        # Persist the rebuilt graph as JSON for figure generators / reviewers.
        kw_payload = {
            'min_occ': 3,
            'nodes': [
                {'label': n, 'occurrences': int(d.get('occurrences', 0))}
                for n, d in kw_graph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, 'weight': int(d.get('weight', 1))}
                for u, v, d in kw_graph.edges(data=True)
            ],
        }
        (processed_dir / 'keyword_network.json').write_text(
            json.dumps(kw_payload, indent=2, ensure_ascii=False, default=str),
            encoding='utf-8',
        )
        print(f"  Created: keyword_network.json ({kw_graph.number_of_nodes()} nodes, {kw_graph.number_of_edges()} edges)")

        # Keyword-normalisation robustness (R2 minor #3): with/without thesaurus.
        try:
            norm_effect = compute_normalization_effect(scopus.df, min_occ=3)
            (processed_dir / 'normalization_effect.json').write_text(
                json.dumps(norm_effect, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8',
            )
            results['normalization_effect'] = norm_effect
            print(f"  Created: normalization_effect.json "
                  f"({norm_effect['raw_nodes']}->{norm_effect['normalised_nodes']} nodes, "
                  f"{norm_effect['raw_clusters']}->{norm_effect['normalised_clusters']} clusters, "
                  f"ARI={norm_effect['ari_raw_vs_normalised']:.3f})")
        except Exception as exc:
            warnings.warn(f"compute_normalization_effect failed: {exc}")

        # Cluster-stability sensitivity (G2) — on corpus-built keyword graph.
        try:
            sens = SensitivityAnalyzer(kw_graph)
            sens_payload = sens.to_json(df=scopus.df)
            (processed_dir / 'sensitivity_matrix.json').write_text(
                json.dumps(sens_payload, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8',
            )
            results['sensitivity'] = {
                'cluster_counts_by_gamma': sens_payload['cluster_counts_by_gamma'],
                'ari_vs_baseline_gamma': sens_payload['ari_vs_baseline_gamma']['ari'],
            }
            print(f"  Created: sensitivity_matrix.json")
        except Exception as exc:
            warnings.warn(f"SensitivityAnalyzer failed: {exc}")

        # Theory operationalisation (R2.B) — Rogers / TAM proxy hypothesis tests.
        try:
            theory = TheoryOperationalisation(scopus.df)
            theory_payload = theory.to_json(kw_graph)
            (processed_dir / 'theory_tests.json').write_text(
                json.dumps(theory_payload, indent=2, ensure_ascii=False, default=str),
                encoding='utf-8',
            )
            results['theory'] = {
                'h1_rho': theory_payload['h1_rogers_vs_centrality'].get('rho'),
                'h2_rho': theory_payload['h2_rogers_vs_year'].get('rho'),
                'h3_rho': theory_payload['h3_tam_vs_citations'].get('rho'),
            }
            print(f"  Created: theory_tests.json")
        except Exception as exc:
            warnings.warn(f"TheoryOperationalisation failed: {exc}")

        # Industry–academia lag (R2 major #3): mobile / generative-AI term families.
        try:
            lag = compute_industry_lag(scopus.df, min_occ=3)
            (processed_dir / 'industry_lag.json').write_text(
                json.dumps(lag, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['industry_lag'] = lag
            fams = lag['families']
            print("  Created: industry_lag.json (" + ", ".join(
                f"{k}: kw={v['n_docs_author_keyword']}/abs={v['n_docs_abstract']}"
                f"{'*innet' if v['in_keyword_network'] else '/absent'}"
                for k, v in fams.items()) + ")")
        except Exception as exc:
            warnings.warn(f"compute_industry_lag failed: {exc}")

        # bibliometrix-standard analytical menu (Lotka, Bradford, Thematic, Trend, ThreeField).
        try:
            lotka = LotkaAnalyzer(scopus.df).to_json()
            (processed_dir / 'lotka.json').write_text(
                json.dumps(lotka, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['lotka'] = {'alpha': lotka.get('alpha'), 'r_squared': lotka.get('r_squared'),
                                'n_authors': lotka.get('n_authors')}
            print(f"  Created: lotka.json (alpha={lotka.get('alpha'):.3f}, R^2={lotka.get('r_squared'):.3f})")
        except Exception as exc:
            warnings.warn(f"LotkaAnalyzer failed: {exc}")

        try:
            bradford = BradfordAnalyzer(scopus.df).to_json()
            (processed_dir / 'bradford.json').write_text(
                json.dumps(bradford, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['bradford'] = {'multiplier_k': bradford.get('multiplier_k'),
                                    'total_sources': bradford.get('total_sources')}
            print(f"  Created: bradford.json (k={bradford.get('multiplier_k'):.3f}, "
                  f"sources={bradford.get('total_sources')})")
        except Exception as exc:
            warnings.warn(f"BradfordAnalyzer failed: {exc}")

        try:
            thematic = ThematicMapAnalyzer(kw_graph).to_json()
            (processed_dir / 'thematic_map.json').write_text(
                json.dumps(thematic, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['thematic_map'] = {'n_clusters': len(thematic.get('clusters', [])),
                                        'cent_med': thematic.get('centrality_median'),
                                        'dens_med': thematic.get('density_median')}
            print(f"  Created: thematic_map.json ({len(thematic.get('clusters', []))} clusters)")
        except Exception as exc:
            warnings.warn(f"ThematicMapAnalyzer failed: {exc}")

        try:
            trend = TrendTopicsAnalyzer(scopus.df).to_json()
            (processed_dir / 'trend_topics.json').write_text(
                json.dumps(trend, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['trend_topics'] = {'n_keywords': len(trend.get('frequency_by_year', {}).get('keywords', [])),
                                        'n_years': len(trend.get('frequency_by_year', {}).get('years', []))}
            print(f"  Created: trend_topics.json")
        except Exception as exc:
            warnings.warn(f"TrendTopicsAnalyzer failed: {exc}")

        try:
            three = ThreeFieldPlotAnalyzer(scopus.df).to_json()
            (processed_dir / 'three_field.json').write_text(
                json.dumps(three, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            results['three_field'] = {'n_sources': len(three.get('top_sources', [])),
                                       'n_authors': len(three.get('top_authors', [])),
                                       'n_keywords': len(three.get('top_keywords', []))}
            print(f"  Created: three_field.json")
        except Exception as exc:
            warnings.warn(f"ThreeFieldPlotAnalyzer failed: {exc}")

    # 4. Export results
    print("\n[4/4] Exporting results...")
    exporter = ResultsExporter(output_dir)

    # Network summary JSON
    path = exporter.export_network_summary(networks)
    print(f"  Created: {Path(path).name}")

    # Node metrics CSV
    path = exporter.export_node_metrics(networks)
    print(f"  Created: {Path(path).name}")

    # LaTeX tables
    if 'top_authors' in results['scopus']:
        path = exporter.export_latex_table(
            results['scopus']['top_authors'],
            "table_authors.tex",
            caption="Top Authors by Publication Count",
            label="tab:authors"
        )
        results['tables'].append(path)
        print(f"  Created: table_authors.tex")

    if 'top_sources' in results['scopus']:
        path = exporter.export_latex_table(
            results['scopus']['top_sources'],
            "table_sources.tex",
            caption="Top Publication Sources",
            label="tab:sources"
        )
        results['tables'].append(path)
        print(f"  Created: table_sources.tex")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import sys

    # Default paths
    data_dir = "/home/cc/claude_code/capstone_project/data"
    output_dir = "/home/cc/claude_code/capstone_project/thesis/figures"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    results = run_full_analysis(data_dir, output_dir)

    # Print summary
    print(f"\nGenerated {len(results['figures'])} figures")
    print(f"Generated {len(results['tables'])} LaTeX tables")
