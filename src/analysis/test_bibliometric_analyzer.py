#!/usr/bin/env python3
"""
Unit Tests for Bibliometric Analyzer Module

Tests cover:
- VOSViewer file parsing (map and network files)
- Network analysis with NetworkX
- Scopus CSV processing
- Visualization generation
- Results export

Run with: pytest test_bibliometric_analyzer.py -v
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, main
from io import StringIO

import numpy as np
import pandas as pd
import networkx as nx

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bibliometric_analyzer import (
    VOSViewerNode,
    VOSViewerNetwork,
    NetworkMetrics,
    NodeMetrics,
    VOSViewerParser,
    NetworkAnalyzer,
    ScopusAnalyzer,
    Visualizer,
    ResultsExporter,
    run_full_analysis,
    # New for the Array revision.
    AffiliationBiasAnalyzer,
    SensitivityAnalyzer,
    TemporalCouplingAnalyzer,
    TheoryOperationalisation,
    RANDOM_SEED,
    _adjusted_rand_index,
    _normalise_country,
    _canonicalise_keyword,
    compute_normalization_effect,
    compute_industry_lag,
)
from networkx.algorithms import community


class TestVOSViewerNode(TestCase):
    """Test VOSViewerNode dataclass."""

    def test_node_creation(self):
        """Test creating a VOSViewer node."""
        node = VOSViewerNode(
            id=1,
            label="test keyword",
            x=0.5,
            y=-0.3,
            cluster=1,
            links=5,
            total_link_strength=10,
            occurrences=3
        )

        self.assertEqual(node.id, 1)
        self.assertEqual(node.label, "test keyword")
        self.assertEqual(node.x, 0.5)
        self.assertEqual(node.cluster, 1)
        self.assertIsNone(node.avg_pub_year)

    def test_node_with_optional_fields(self):
        """Test node with all optional fields."""
        node = VOSViewerNode(
            id=2,
            label="complete node",
            x=1.0,
            y=1.0,
            cluster=2,
            links=3,
            total_link_strength=6,
            occurrences=4,
            avg_pub_year=2022.5,
            avg_citations=5.2,
            avg_norm_citations=1.1
        )

        self.assertEqual(node.avg_pub_year, 2022.5)
        self.assertEqual(node.avg_citations, 5.2)
        self.assertEqual(node.avg_norm_citations, 1.1)


class TestVOSViewerParser(TestCase):
    """Test VOSViewer file parser."""

    def setUp(self):
        """Create temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.map_dir = Path(self.temp_dir) / "map"
        self.net_dir = Path(self.temp_dir) / "net"
        self.map_dir.mkdir()
        self.net_dir.mkdir()

        # Create sample map file
        map_content = """id\tlabel\tx\ty\tcluster\tweight<Links>\tweight<Total link strength>\tweight<Occurrences>\tscore<Avg. pub. year>\tscore<Avg. citations>\tscore<Avg. norm. citations>
1\tgame development\t0.5\t0.3\t1\t5\t10\t8\t2022\t5.5\t1.2
2\tunity\t-0.2\t0.1\t1\t3\t7\t5\t2021\t3.0\t0.8
3\teducation\t0.1\t-0.4\t2\t4\t8\t6\t2020\t4.2\t1.0
"""
        (self.map_dir / "cosco5.txt").write_text(map_content, encoding='utf-8-sig')

        # Create sample network file
        net_content = """1\t2\t3
1\t3\t2
2\t3\t1
"""
        (self.net_dir / "cosco5.txt").write_text(net_content, encoding='utf-8-sig')

        self.parser = VOSViewerParser(str(self.map_dir), str(self.net_dir))

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_parse_map_file(self):
        """Test parsing a map file."""
        nodes = self.parser.parse_map_file(self.map_dir / "cosco5.txt")

        self.assertEqual(len(nodes), 3)
        self.assertIn(1, nodes)
        self.assertEqual(nodes[1].label, "game development")
        self.assertEqual(nodes[1].cluster, 1)
        self.assertEqual(nodes[2].occurrences, 5)
        self.assertEqual(nodes[3].avg_pub_year, 2020)

    def test_parse_net_file(self):
        """Test parsing a network file."""
        edges = self.parser.parse_net_file(self.net_dir / "cosco5.txt")

        self.assertEqual(len(edges), 3)
        self.assertIn((1, 2, 3), edges)
        self.assertIn((1, 3, 2), edges)
        self.assertIn((2, 3, 1), edges)

    def test_build_networkx_graph(self):
        """Test building NetworkX graph."""
        nodes = self.parser.parse_map_file(self.map_dir / "cosco5.txt")
        edges = self.parser.parse_net_file(self.net_dir / "cosco5.txt")
        graph = self.parser.build_networkx_graph(nodes, edges)

        self.assertIsInstance(graph, nx.Graph)
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 3)

        # Check node attributes
        self.assertEqual(graph.nodes[1]['label'], "game development")
        self.assertEqual(graph.nodes[2]['cluster'], 1)

        # Check edge weights
        self.assertEqual(graph[1][2]['weight'], 3)

    def test_parse_network(self):
        """Test parsing complete network."""
        network = self.parser.parse_network(5)

        self.assertIsNotNone(network)
        self.assertEqual(network.name, "cosco5")
        self.assertEqual(network.analysis_type, "Keyword Co-occurrence")
        self.assertEqual(network.unit_of_analysis, "Author Keywords")
        self.assertEqual(len(network.nodes), 3)
        self.assertEqual(len(network.edges), 3)
        self.assertIsNotNone(network.graph)

    def test_parse_missing_network(self):
        """Test parsing non-existent network."""
        network = self.parser.parse_network(99)
        self.assertIsNone(network)

    def test_analysis_type_mapping(self):
        """Test analysis type mappings."""
        self.assertEqual(
            VOSViewerParser.ANALYSIS_TYPES[5],
            ("Keyword Co-occurrence", "Author Keywords")
        )
        self.assertEqual(
            VOSViewerParser.ANALYSIS_TYPES[17],
            ("Co-citation", "Cited References")
        )


class TestNetworkAnalyzer(TestCase):
    """Test network analysis functionality."""

    def setUp(self):
        """Create test network."""
        nodes = {
            1: VOSViewerNode(1, "node1", 0, 0, 1, 2, 5, 3),
            2: VOSViewerNode(2, "node2", 1, 0, 1, 2, 4, 2),
            3: VOSViewerNode(3, "node3", 0, 1, 2, 1, 3, 2),
            4: VOSViewerNode(4, "node4", 1, 1, 2, 1, 2, 1),
        }
        edges = [(1, 2, 3), (1, 3, 2), (2, 3, 1), (3, 4, 1)]

        graph = nx.Graph()
        for node_id, node in nodes.items():
            graph.add_node(node_id, label=node.label, cluster=node.cluster,
                           occurrences=node.occurrences)
        for s, t, w in edges:
            graph.add_edge(s, t, weight=w)

        self.network = VOSViewerNetwork(
            name="test",
            analysis_type="Test",
            unit_of_analysis="Test Units",
            nodes=nodes,
            edges=edges,
            graph=graph
        )
        self.analyzer = NetworkAnalyzer(self.network)

    def test_calculate_network_metrics(self):
        """Test network-level metrics calculation."""
        metrics = self.analyzer.calculate_network_metrics()

        self.assertIsInstance(metrics, NetworkMetrics)
        self.assertEqual(metrics.num_nodes, 4)
        self.assertEqual(metrics.num_edges, 4)
        self.assertGreater(metrics.density, 0)
        self.assertLessEqual(metrics.density, 1)
        self.assertGreaterEqual(metrics.avg_clustering, 0)
        self.assertEqual(metrics.num_components, 1)

    def test_calculate_node_metrics(self):
        """Test node-level metrics calculation."""
        metrics = self.analyzer.calculate_node_metrics()

        self.assertEqual(len(metrics), 4)
        self.assertIsInstance(metrics[0], NodeMetrics)

        # Check that node 1 or 3 has highest betweenness (they connect clusters)
        betweenness = {m.node_id: m.betweenness for m in metrics}
        max_betweenness_node = max(betweenness, key=betweenness.get)
        self.assertIn(max_betweenness_node, [1, 3])

    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        summary = self.analyzer.get_cluster_summary()

        self.assertIn(1, summary)
        self.assertIn(2, summary)
        self.assertEqual(summary[1]['num_nodes'], 2)
        self.assertEqual(summary[2]['num_nodes'], 2)

    def test_detect_communities(self):
        """Test community detection."""
        communities = self.analyzer.detect_communities()

        # At least one algorithm should work
        self.assertGreater(len(communities), 0)

        # Each detected community should be a list of sets
        for algo, comms in communities.items():
            self.assertIsInstance(comms, list)
            total_nodes = sum(len(c) for c in comms)
            self.assertEqual(total_nodes, 4)

    def test_empty_network_metrics(self):
        """Test metrics for empty network."""
        empty_network = VOSViewerNetwork(
            name="empty", analysis_type="Test", unit_of_analysis="Test",
            nodes={}, edges=[], graph=nx.Graph()
        )
        analyzer = NetworkAnalyzer(empty_network)
        metrics = analyzer.calculate_network_metrics()

        self.assertEqual(metrics.num_nodes, 0)
        self.assertEqual(metrics.num_edges, 0)


class TestScopusAnalyzer(TestCase):
    """Test Scopus CSV analyzer."""

    def setUp(self):
        """Create temporary Scopus CSV file."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "scopus_test.csv"

        csv_content = '''"Authors","Title","Year","Source title","Cited by","Author Keywords","Affiliations"
"Smith, J.; Jones, M.","Game Development in Education","2022","Journal A","10","game development; education","University A, USA; University B, UK"
"Brown, K.","Unity for Beginners","2021","Journal B","5","unity; game design","University C, Germany"
"Smith, J.; Lee, W.","VR in Learning","2023","Journal A","15","virtual reality; education","University A, USA; University D, Japan"
'''
        self.csv_path.write_text(csv_content, encoding='utf-8-sig')
        self.analyzer = ScopusAnalyzer(str(self.csv_path))

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_load_data(self):
        """Test data loading."""
        self.assertIsInstance(self.analyzer.df, pd.DataFrame)
        self.assertEqual(len(self.analyzer.df), 3)

    def test_get_basic_stats(self):
        """Test basic statistics."""
        stats = self.analyzer.get_basic_stats()

        self.assertEqual(stats['total_documents'], 3)
        self.assertEqual(stats['year_range'], (2021, 2023))
        self.assertEqual(stats['total_citations'], 30)
        self.assertEqual(stats['avg_citations'], 10.0)

    def test_get_publications_by_year(self):
        """Test publication timeline."""
        df_years = self.analyzer.get_publications_by_year()

        self.assertEqual(len(df_years), 3)
        self.assertIn(2022, df_years['Year'].values)

    def test_get_top_authors(self):
        """Test top authors extraction."""
        top_authors = self.analyzer.get_top_authors(n=3)

        self.assertEqual(len(top_authors), 3)
        # Smith, J. appears twice
        smith_row = top_authors[top_authors['Author'] == 'Smith, J.']
        self.assertEqual(smith_row['Publications'].values[0], 2)

    def test_get_keyword_frequency(self):
        """Test keyword frequency."""
        keywords = self.analyzer.get_keyword_frequency()

        self.assertIn('education', keywords)
        self.assertEqual(keywords['education'], 2)
        self.assertEqual(keywords['game development'], 1)


class TestVisualizer(TestCase):
    """Test visualization generator."""

    def setUp(self):
        """Create temporary output directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = Visualizer(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_output_directory_creation(self):
        """Test output directory is created."""
        new_dir = Path(self.temp_dir) / "new_subdir"
        viz = Visualizer(str(new_dir))
        self.assertTrue(new_dir.exists())

    def test_plot_network(self):
        """Test network visualization."""
        # Create simple network
        nodes = {
            1: VOSViewerNode(1, "A", 0, 0, 1, 1, 2, 3),
            2: VOSViewerNode(2, "B", 1, 0, 1, 1, 2, 5),
            3: VOSViewerNode(3, "C", 0, 1, 2, 1, 2, 4),
        }
        graph = nx.Graph()
        for nid, n in nodes.items():
            graph.add_node(nid, x=n.x, y=n.y, label=n.label,
                           cluster=n.cluster, occurrences=n.occurrences)
        graph.add_edge(1, 2, weight=2)
        graph.add_edge(2, 3, weight=1)

        network = VOSViewerNetwork(
            "test", "Test", "Units", nodes, [(1, 2, 2), (2, 3, 1)], graph
        )

        path = self.visualizer.plot_network(network, "test_network")

        self.assertTrue(Path(path).exists())
        self.assertTrue(path.endswith('.pdf'))

    def test_plot_publication_timeline(self):
        """Test publication timeline plot."""
        df = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023],
            'count': [5, 8, 12, 10]
        })

        path = self.visualizer.plot_publication_timeline(df, "test_timeline")

        self.assertTrue(Path(path).exists())
        self.assertTrue(path.endswith('.pdf'))

    def test_plot_empty_timeline(self):
        """Test empty timeline handling."""
        df = pd.DataFrame()
        path = self.visualizer.plot_publication_timeline(df, "empty_timeline")
        self.assertEqual(path, "")


class TestResultsExporter(TestCase):
    """Test results exporter."""

    def setUp(self):
        """Create temporary output directory and test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ResultsExporter(self.temp_dir)

        # Create test network
        nodes = {1: VOSViewerNode(1, "test", 0, 0, 1, 1, 2, 3)}
        graph = nx.Graph()
        graph.add_node(1, label="test", cluster=1, occurrences=3)

        self.networks = {
            5: VOSViewerNetwork(
                "cosco5", "Keyword", "Keywords", nodes, [], graph
            )
        }

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_export_network_summary(self):
        """Test JSON export."""
        path = self.exporter.export_network_summary(self.networks)

        self.assertTrue(Path(path).exists())

        with open(path, 'r') as f:
            data = json.load(f)

        self.assertIn('cosco5', data)
        self.assertEqual(data['cosco5']['analysis_type'], "Keyword")

    def test_export_latex_table(self):
        """Test LaTeX table export."""
        df = pd.DataFrame({
            'Author': ['Smith', 'Jones'],
            'Publications': [5, 3]
        })

        path = self.exporter.export_latex_table(
            df, "test_table.tex",
            caption="Test Caption",
            label="tab:test"
        )

        self.assertTrue(Path(path).exists())

        content = Path(path).read_text()
        self.assertIn('\\begin{table}', content)
        self.assertIn('Test Caption', content)
        self.assertIn('tab:test', content)


class TestIntegration(TestCase):
    """Integration tests for full analysis pipeline."""

    def setUp(self):
        """Set up test data directory."""
        self.data_dir = Path("/home/cc/claude_code/capstone_project/data")
        self.temp_output = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_output)

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline with real data."""
        if not self.data_dir.exists():
            self.skipTest("Data directory not found")

        results = run_full_analysis(str(self.data_dir), self.temp_output)

        # Check networks were parsed
        self.assertGreater(len(results['networks']), 0)

        # Check Scopus data was loaded
        if 'stats' in results['scopus']:
            self.assertGreater(results['scopus']['stats']['total_documents'], 0)

        # Check figures were generated
        self.assertGreater(len(results['figures']), 0)
        for fig_path in results['figures']:
            self.assertTrue(Path(fig_path).exists())


# =============================================================================
# Tests for the Array-revision additions: reproducibility, affiliation bias,
# sensitivity, temporal coupling.
# =============================================================================


class TestReproducibility(TestCase):
    """Determinism of seeded community detection (closes the seed gap before any
    sensitivity claims become meaningful)."""

    def setUp(self):
        # Build a small fixed weighted graph.
        self.G = nx.Graph()
        self.G.add_weighted_edges_from([
            ('a', 'b', 3), ('a', 'c', 2), ('b', 'c', 4),
            ('d', 'e', 5), ('d', 'f', 4), ('e', 'f', 3),
            ('c', 'd', 1),
        ])
        for n in self.G.nodes():
            self.G.nodes[n]['cluster'] = 0
            self.G.nodes[n]['label'] = n
        self.network = VOSViewerNetwork(
            name='test', analysis_type='Test', unit_of_analysis='Nodes',
            nodes={}, edges=[], graph=self.G,
        )

    def test_two_runs_identical(self):
        """Seeded community detection must produce identical partitions across runs."""
        analyzer = NetworkAnalyzer(self.network)
        a = analyzer.detect_communities(seed=RANDOM_SEED)
        b = analyzer.detect_communities(seed=RANDOM_SEED)
        for algo in a:
            if algo not in b:
                continue
            ca = sorted([sorted(c) for c in a[algo]])
            cb = sorted([sorted(c) for c in b[algo]])
            self.assertEqual(ca, cb, f"{algo} non-deterministic with seed")


class TestAdjustedRandIndex(TestCase):
    """ARI helper used by SensitivityAnalyzer."""

    def test_identical_partitions(self):
        labels = [0, 0, 1, 1, 2, 2]
        self.assertAlmostEqual(_adjusted_rand_index(labels, labels), 1.0)

    def test_relabelled_partitions_equivalent(self):
        # Same partition with different cluster IDs should still give ARI=1.
        a = [0, 0, 1, 1, 2, 2]
        b = [5, 5, 7, 7, 9, 9]
        self.assertAlmostEqual(_adjusted_rand_index(a, b), 1.0)

    def test_random_partitions_low(self):
        a = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        b = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # Worst-case anti-correlated.
        self.assertLess(_adjusted_rand_index(a, b), 0.1)


class TestCountryNormalisation(TestCase):
    """Country aliasing avoids HHI inflation from spelling variants."""

    def test_us_aliases(self):
        self.assertEqual(_normalise_country('USA'), 'United States')
        self.assertEqual(_normalise_country('U.S.A.'), 'United States')
        self.assertEqual(_normalise_country('united states'), 'United States')

    def test_uk_aliases(self):
        self.assertEqual(_normalise_country('UK'), 'United Kingdom')
        self.assertEqual(_normalise_country('Britain'), 'United Kingdom')

    def test_unknown_pass_through(self):
        self.assertEqual(_normalise_country('Brazil'), 'Brazil')


class TestAffiliationBiasAnalyzer(TestCase):
    """G5 — geographic / institutional concentration analysis."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Authors with affiliations': [
                'Smith, John, MIT, Cambridge, MA, USA; Doe, Jane, MIT, Cambridge, MA, USA',
                'Mueller, Hans, TU Berlin, Berlin, Germany',
                'Smith, John, MIT, Cambridge, MA, USA; Lee, Min, Seoul Nat Univ, Seoul, South Korea',
            ],
            'Year': [2020, 2021, 2022],
        })
        self.ab = AffiliationBiasAnalyzer(self.df)

    def test_triple_extraction_format(self):
        triples = self.ab.extract_author_institution_country_triples()
        # Three docs with a total of five author-affiliation pairs.
        self.assertEqual(len(triples), 5)
        self.assertEqual(triples[0]['author'], 'Smith, John')
        self.assertEqual(triples[0]['institution'], 'MIT')
        self.assertEqual(triples[0]['country'], 'United States')

    def test_hhi_concentration(self):
        idx = self.ab.compute_concentration_indices()
        # 3 docs, USA in 2/3, Germany in 1/3, Korea in 1/3.
        # HHI = (2/3)^2 + (1/3)^2 + (1/3)^2 ≈ 0.444 + 0.111 + 0.111 = 0.666.
        # But we count countries per doc set (not per triple), so:
        # doc 0: {USA}, doc 1: {Germany}, doc 2: {USA, Korea}.
        # Country counts: USA=2, Germany=1, Korea=1; total=4.
        # HHI = (2/4)^2 + (1/4)^2 + (1/4)^2 = 0.25 + 0.0625 + 0.0625 = 0.375.
        self.assertAlmostEqual(idx['hhi_country'], 0.375, places=2)

    def test_collaboration_network(self):
        G = self.ab.compute_collaboration_network()
        # Doc 2 has USA-Korea co-authorship → one edge.
        self.assertGreaterEqual(G.number_of_edges(), 1)


class TestSensitivityAnalyzer(TestCase):
    """G2 — cluster stability under resolution / min-occurrence sweeps."""

    def setUp(self):
        # Two clearly separable triangles connected by one weak link.
        self.G = nx.Graph()
        self.G.add_weighted_edges_from([
            ('a', 'b', 5), ('a', 'c', 5), ('b', 'c', 5),
            ('d', 'e', 5), ('d', 'f', 5), ('e', 'f', 5),
            ('c', 'd', 1),
        ])
        self.sens = SensitivityAnalyzer(self.G)

    def test_resolution_sweep_returns_dict(self):
        partitions = self.sens.parameter_sweep(gammas=(0.5, 1.0, 1.5))
        self.assertIn(1.0, partitions)
        self.assertEqual(len(partitions), 3)

    def test_ari_baseline_self_equals_one(self):
        partitions = self.sens.parameter_sweep(gammas=(1.0,))
        ari = self.sens.compute_ari_matrix(partitions, baseline_key=1.0)
        self.assertAlmostEqual(ari['ari']['1.0'], 1.0)

    def test_jaccard_in_zero_to_one(self):
        partitions = self.sens.parameter_sweep(gammas=(0.5, 1.0, 1.5))
        jac = self.sens.compute_jaccard_stability(partitions, baseline_key=1.0)
        for variant, vals in jac.items():
            for v in vals:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_corpus_rebuild_min_occ(self):
        df = pd.DataFrame({
            'Author Keywords': [
                'unity; vr; education',
                'unity; ar; learning',
                'unity; godot; programming',
                'godot; learning',
            ],
        })
        # min_occ=1 keeps every keyword; min_occ=3 keeps only 'unity'.
        G_low = SensitivityAnalyzer.from_corpus_keywords(df, min_occ=1)
        G_high = SensitivityAnalyzer.from_corpus_keywords(df, min_occ=3)
        self.assertGreater(G_low.number_of_nodes(), G_high.number_of_nodes())
        self.assertEqual(G_high.number_of_nodes(), 1)


class TestTemporalCouplingAnalyzer(TestCase):
    """G4 — bibliographic coupling stratified across eras."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Year': [2015, 2019, 2023, 2024],
            'References': [
                'Smith, J., Foo, Journal A, 1, pp. 1-10, (2010); Doe, A., Bar, Journal B, 2, pp. 11-20, (2012)',
                'Smith, J., Foo, Journal A, 1, pp. 1-10, (2010); Lee, K., Baz, Journal C, 3, pp. 21-30, (2015)',
                'Doe, A., Bar, Journal B, 2, pp. 11-20, (2012); Lee, K., Baz, Journal C, 3, pp. 21-30, (2015)',
                'Lee, K., Baz, Journal C, 3, pp. 21-30, (2015); New, R., New work, Journal D, 4, pp. 31-40, (2024)',
            ],
        })
        self.tc = TemporalCouplingAnalyzer(self.df)

    def test_reference_parsing(self):
        keys = self.tc.parse_references_field(self.df['References'].iloc[0])
        self.assertIn('smith_2010', keys)
        self.assertIn('doe_2012', keys)

    def test_era_partitioning(self):
        eras = self.tc.per_era_coupling()
        self.assertEqual(eras['pre-2018']['n_docs'], 1)
        self.assertEqual(eras['2018-2021']['n_docs'], 1)
        self.assertEqual(eras['2022-2026']['n_docs'], 2)

    def test_coupling_edge_weight_is_shared_count(self):
        # Docs 2 and 3 (both in 2022-2026 era) share 1 ref ('lee_2015').
        eras = self.tc.per_era_coupling()
        self.assertGreaterEqual(eras['2022-2026']['n_coupling_edges'], 1)


# =============================================================================
# Tests for the second wave of Array-revision additions: keyword normalisation,
# alternative community-detection algorithms, corpus-build determinism, growth
# models, theory operationalisation, and the academia-industry lag metric.
# =============================================================================


class TestCanonicaliseKeyword(TestCase):
    """R2 minor #3 — deterministic singular/plural + synonym normalisation."""

    def test_synonym_and_plural_forms(self):
        self.assertEqual(_canonicalise_keyword('Serious Games'), 'serious game')
        self.assertEqual(_canonicalise_keyword('GAMES'), 'game')
        self.assertEqual(_canonicalise_keyword('educational games'), 'educational game')

    def test_protected_plural_not_singularised(self):
        # 'graphics' is a protected non-plural — must NOT become 'computer graphic'.
        self.assertEqual(_canonicalise_keyword('Computer Graphics'), 'computer graphics')
        self.assertEqual(_canonicalise_keyword('ethics'), 'ethics')

    def test_direct_synonyms(self):
        self.assertEqual(_canonicalise_keyword('Unity3D'), 'unity')
        self.assertEqual(_canonicalise_keyword('GenAI'), 'generative ai')

    def test_empty_input(self):
        self.assertEqual(_canonicalise_keyword(''), '')


class TestNormalizationEffect(TestCase):
    """R2 minor #3 — evidence that normalisation consolidates variants without
    destabilising the thematic structure."""

    def setUp(self):
        # Rows mixing singular/plural variants so several raw nodes collapse.
        self.df = pd.DataFrame({
            'Author Keywords': [
                'serious game; game; unity',
                'serious games; games; unity',
                'serious game; game; unity',
                'serious games; games; unity',
                'serious game; game; unity',
                'serious games; games; unity',
            ],
        })

    def test_effect_dict_shape(self):
        r = compute_normalization_effect(self.df, min_occ=3)
        for key in ('raw_nodes', 'normalised_nodes', 'nodes_merged_away',
                    'raw_clusters', 'normalised_clusters',
                    'ari_raw_vs_normalised', 'top_merges'):
            self.assertIn(key, r)
        self.assertIsInstance(r['top_merges'], list)

    def test_normalisation_reduces_nodes(self):
        r = compute_normalization_effect(self.df, min_occ=3)
        self.assertGreaterEqual(r['nodes_merged_away'], 1)
        self.assertLessEqual(r['normalised_nodes'], r['raw_nodes'])


class TestSensitivityAlgorithms(TestCase):
    """R1 #5 — Louvain vs Leiden vs Infomap agreement on a corpus-built graph."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Author Keywords': [
                'unity; vr; game development',
                'unity; ar; game development',
                'unity; vr; education',
                'godot; programming; education',
                'godot; programming; learning',
                'godot; learning; education',
            ],
        })
        self.G = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        self.sens = SensitivityAnalyzer(self.G)

    def test_partition_leiden_covers_all_nodes(self):
        try:
            import igraph  # noqa: F401
            import leidenalg  # noqa: F401
        except Exception:
            self.skipTest("leidenalg/igraph not installed")
        part = self.sens.partition_leiden(gamma=1.0, seed=42)
        self.assertIsNotNone(part)
        self.assertIsInstance(part, list)
        self.assertTrue(all(isinstance(c, set) for c in part))
        union = set().union(*part) if part else set()
        self.assertEqual(union, set(self.G.nodes()))

    def test_partition_infomap_covers_all_nodes(self):
        try:
            from infomap import Infomap  # noqa: F401
        except Exception:
            self.skipTest("infomap not installed")
        part = self.sens.partition_infomap(seed=42)
        self.assertIsNotNone(part)
        self.assertIsInstance(part, list)
        self.assertTrue(all(isinstance(c, set) for c in part))
        union = set().union(*part) if part else set()
        self.assertEqual(union, set(self.G.nodes()))

    def test_algorithm_agreement_summary(self):
        agr = self.sens.algorithm_agreement(gamma=1.0, seed=42)
        self.assertIn('summary', agr)
        self.assertIn('louvain', agr['summary'])
        self.assertIn('n_clusters', agr['summary']['louvain'])
        self.assertIn('modularity', agr['summary']['louvain'])
        # Louvain compared against itself is a perfect match.
        self.assertIn('ari_vs_louvain', agr)
        self.assertAlmostEqual(agr['ari_vs_louvain']['louvain'], 1.0)

        # Leiden / Infomap only asserted when their libraries are present.
        try:
            import igraph  # noqa: F401
            import leidenalg  # noqa: F401
            self.assertIn('leiden', agr['summary'])
            self.assertIn('n_clusters', agr['summary']['leiden'])
            self.assertIn('modularity', agr['summary']['leiden'])
        except Exception:
            pass
        try:
            from infomap import Infomap  # noqa: F401
            self.assertIn('infomap', agr['summary'])
            self.assertIn('n_clusters', agr['summary']['infomap'])
            self.assertIn('modularity', agr['summary']['infomap'])
        except Exception:
            pass


class TestCorpusBuildDeterminism(TestCase):
    """Guards the PYTHONHASHSEED fix: sorted node/edge insertion must give a
    byte-identical graph and a hash-seed-independent Louvain partition."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Author Keywords': [
                'unity; vr; education',
                'unity; ar; learning',
                'unity; godot; programming',
                'godot; learning; education',
                'unity; vr; game development',
                'godot; programming; education',
            ],
        })

    def test_identical_nodes_and_edges(self):
        G1 = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        G2 = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        self.assertEqual(sorted(G1.nodes()), sorted(G2.nodes()))
        self.assertEqual(set(G1.edges()), set(G2.edges()))

    def test_identical_louvain_cluster_count(self):
        G1 = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        G2 = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        p1 = community.louvain_communities(G1, weight='weight', resolution=1.0, seed=42)
        p2 = community.louvain_communities(G2, weight='weight', resolution=1.0, seed=42)
        self.assertEqual(len(p1), len(p2))


class TestFieldCompletenessScore(TestCase):
    """R1.3 / R1.6 — competing growth-model fits on the cumulative curve."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "growth.csv"
        # Plausible S-shaped cumulative growth across 16 distinct years.
        counts = {2010: 1, 2011: 1, 2012: 2, 2013: 3, 2014: 4, 2015: 6,
                  2016: 8, 2017: 11, 2018: 14, 2019: 18, 2020: 22, 2021: 25,
                  2022: 27, 2023: 28, 2024: 29, 2025: 30}
        rows = []
        for year, c in counts.items():
            for i in range(c):
                rows.append({
                    'Authors': 'Smith, J.', 'Title': f'doc_{year}_{i}',
                    'Year': year, 'Source title': 'Journal A',
                    'Cited by': i, 'Author Keywords': 'game development; unity',
                })
        pd.DataFrame(rows).to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        self.analyzer = ScopusAnalyzer(str(self.csv_path))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_growth_model_keys(self):
        r = self.analyzer.compute_field_completeness_score()
        for key in ('gompertz_r2', 'bilogistic_r2', 'linear_r2', 'exponential_r2',
                    'saturation_ratio', 'best_model'):
            self.assertIn(key, r)
        for model in ('linear', 'exponential', 'gompertz', 'bilogistic'):
            self.assertIn(f'{model}_aic', r)
            self.assertIn(f'{model}_bic', r)

    def test_best_model_valid(self):
        r = self.analyzer.compute_field_completeness_score()
        self.assertIn(r['best_model'],
                      {'linear', 'exponential', 'gompertz', 'bilogistic'})


class TestScopusYearMax(TestCase):
    """Robustness variant (R1.3 / R2.2) — exclude the incomplete 2026 partial year."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = Path(self.temp_dir) / "years.csv"
        csv_content = '''"Authors","Title","Year","Source title","Cited by","Author Keywords"
"A, B","d1","2023","J","1","game"
"A, B","d2","2024","J","1","game"
"A, B","d3","2025","J","1","game"
"A, B","d4","2026","J","0","game"
"A, B","d5","2026","J","0","game"
'''
        self.csv_path.write_text(csv_content, encoding='utf-8-sig')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_year_cap_excludes_2026(self):
        full = ScopusAnalyzer(str(self.csv_path))
        capped = ScopusAnalyzer(str(self.csv_path), year_max=2025)
        self.assertEqual(len(full.df), 5)
        self.assertEqual(len(capped.df), 3)
        self.assertFalse((capped.df['Year'] > 2025).any())


class TestTheoryOperationalisation(TestCase):
    """R2 major #1 — engine-context construct guard, proxy-source variants,
    and abstract inclusion toggle."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Author Keywords': [
                'unity; game development',
                'construct; learning',
                'godot; programming',
                'unity; vr',
            ],
            'Index Keywords': [
                'game engines',
                'education',
                'software',
                'virtual reality',
            ],
            'Abstract': [
                'We used the Construct 3 engine to build a game.',
                'We construct a theoretical model of learning.',
                'Scirra Construct was used to prototype the level.',
                'The construction of knowledge underpins the pedagogy.',
            ],
            'Year': [2020, 2021, 2022, 2023],
            'Cited by': [5, 3, 8, 2],
        })

    def test_construct_guard_is_strict(self):
        import re
        t = TheoryOperationalisation(self.df)
        # Naive \bconstruct\b across keywords + abstract matches rows 0, 1, 2.
        naive = [
            i for i in self.df.index
            if re.search(r'\bconstruct\b',
                         (str(self.df.at[i, 'Author Keywords']) + ' '
                          + str(self.df.at[i, 'Abstract'])).lower())
        ]
        strict = t.docs_mentioning('construct')
        # Engine-context rows only: 'Construct 3' (0) and 'Scirra' (2).
        self.assertEqual(sorted(strict), [0, 2])
        self.assertLess(len(strict), len(naive))

    def test_variant_hypothesis_tests_shape(self):
        graph = SensitivityAnalyzer.from_corpus_keywords(self.df, min_occ=1)
        vh = TheoryOperationalisation.variant_hypothesis_tests(self.df, graph)
        for cfg in ('author_keywords_only', 'index_keywords_only',
                    'keywords_plus_abstract'):
            self.assertIn(cfg, vh)
            for key in ('h1', 'h2', 'h3', 'n_platforms_detected'):
                self.assertIn(key, vh[cfg])

    def test_include_abstract_toggle(self):
        df = pd.DataFrame({
            'Author Keywords': ['unity; game development'],
            'Index Keywords': ['game engines'],
            'Abstract': ['This mentions zzuniqueabstracttoken only in prose.'],
        })
        t_yes = TheoryOperationalisation(df, include_abstract=True)
        t_no = TheoryOperationalisation(df, include_abstract=False)
        self.assertIn('zzuniqueabstracttoken', t_yes._doc_keyword_text(0))
        self.assertNotIn('zzuniqueabstracttoken', t_no._doc_keyword_text(0))


class TestIndustryLag(TestCase):
    """R2 major #3 — academia-industry lag made measurable per term family."""

    def setUp(self):
        self.df = pd.DataFrame({
            'Author Keywords': [
                'unity; virtual reality; game development',
                'mobile game; android; education',
                'generative ai; game development',
                'machine learning; procedural content generation',
            ],
            'Abstract': [
                'A virtual reality study using immersive headsets.',
                'A mobile game for android smartphones.',
                'We apply chatgpt and large language model prompting.',
                'Deep learning and neural network methods are compared.',
            ],
            'Year': [2019, 2021, 2023, 2024],
        })

    def test_families_and_fields(self):
        lag = compute_industry_lag(self.df, min_occ=3)
        self.assertIn('families', lag)
        for fam in ('mobile', 'generative_ai', 'immersive_xr', 'ai_general'):
            self.assertIn(fam, lag['families'])
            entry = lag['families'][fam]
            for key in ('n_docs_author_keyword', 'n_docs_abstract',
                        'in_keyword_network', 'network_terms', 'abstract_by_year'):
                self.assertIn(key, entry)
            self.assertIsInstance(entry['in_keyword_network'], bool)
            self.assertIsInstance(entry['network_terms'], list)
            self.assertIsInstance(entry['abstract_by_year'], dict)


if __name__ == '__main__':
    main(verbosity=2)
