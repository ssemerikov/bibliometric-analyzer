#!/usr/bin/env python3
"""
Unit Tests for Chat Analyzer Module

Tests cover:
- Chat transcript parsing
- Insight extraction
- Discussion point generation
- Export functionality

Run with: pytest test_chat_analyzer.py -v
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, main

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat_analyzer import (
    ChatMessage,
    ChatTranscript,
    ClusterInterpretation,
    ResearchInsight,
    DiscussionPoint,
    ChatParser,
    InsightExtractor,
    DiscussionGenerator,
    run_chat_analysis
)


class TestChatMessage(TestCase):
    """Test ChatMessage dataclass."""

    def test_message_creation(self):
        """Test creating a chat message."""
        msg = ChatMessage(
            speaker="Sergey Semerikov",
            timestamp="19:30:00",
            content="Let's discuss the clusters.",
            is_advisor=True
        )

        self.assertEqual(msg.speaker, "Sergey Semerikov")
        self.assertEqual(msg.timestamp, "19:30:00")
        self.assertTrue(msg.is_advisor)

    def test_message_default_advisor(self):
        """Test default is_advisor value."""
        msg = ChatMessage(
            speaker="Student",
            timestamp="19:31:00",
            content="Question about VR."
        )

        self.assertFalse(msg.is_advisor)


class TestChatTranscript(TestCase):
    """Test ChatTranscript dataclass."""

    def test_transcript_creation(self):
        """Test creating a chat transcript."""
        transcript = ChatTranscript(
            date="2025-12-04",
            filepath="/path/to/file.txt",
            messages=[
                ChatMessage("Speaker1", "10:00:00", "Hello"),
                ChatMessage("Speaker2", "10:01:00", "Hi"),
            ],
            duration_minutes=60.0,
            advisor_messages=5,
            student_messages=10
        )

        self.assertEqual(transcript.date, "2025-12-04")
        self.assertEqual(len(transcript.messages), 2)
        self.assertEqual(transcript.duration_minutes, 60.0)


class TestChatParser(TestCase):
    """Test ChatParser functionality."""

    def setUp(self):
        """Create temporary directory with test chat files."""
        self.temp_dir = tempfile.mkdtemp()
        self.chats_dir = Path(self.temp_dir) / "chats"
        self.chats_dir.mkdir()

        # Create sample chat file
        chat_content = """[Sergey Semerikov] 19:00:00
Hello, let's start the meeting.

[Bohdan Ostashchenko] 19:00:10
Yes, I'm ready.

[Sergey Semerikov] 19:00:30
Let's discuss the VOSViewer clusters and Unity.

[Bohdan Ostashchenko] 19:01:00
I have questions about the methodology.
"""
        (self.chats_dir / "2025-12-04.txt").write_text(chat_content, encoding='utf-8')

        self.parser = ChatParser(str(self.chats_dir))

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_parse_file(self):
        """Test parsing a single chat file."""
        transcript = self.parser.parse_file(self.chats_dir / "2025-12-04.txt")

        self.assertEqual(transcript.date, "2025-12-04")
        self.assertEqual(len(transcript.messages), 4)

        # Check first message
        self.assertEqual(transcript.messages[0].speaker, "Sergey Semerikov")
        self.assertTrue(transcript.messages[0].is_advisor)

        # Check student message
        self.assertEqual(transcript.messages[1].speaker, "Bohdan Ostashchenko")
        self.assertFalse(transcript.messages[1].is_advisor)

    def test_advisor_detection(self):
        """Test correct identification of advisor messages."""
        transcript = self.parser.parse_file(self.chats_dir / "2025-12-04.txt")

        advisor_messages = [m for m in transcript.messages if m.is_advisor]
        student_messages = [m for m in transcript.messages if not m.is_advisor]

        self.assertEqual(len(advisor_messages), 2)
        self.assertEqual(len(student_messages), 2)

    def test_duration_calculation(self):
        """Test meeting duration calculation."""
        transcript = self.parser.parse_file(self.chats_dir / "2025-12-04.txt")

        # Duration should be ~1 minute (from 19:00:00 to 19:01:00)
        self.assertGreater(transcript.duration_minutes, 0)
        self.assertLess(transcript.duration_minutes, 5)

    def test_parse_all(self):
        """Test parsing all chat files."""
        # Add another chat file
        chat2 = """[Sergey Semerikov] 10:00:00
Morning meeting.

[Bohdan Ostashchenko] 10:00:30
Hello.
"""
        (self.chats_dir / "2025-12-12.txt").write_text(chat2, encoding='utf-8')

        transcripts = self.parser.parse_all()

        self.assertEqual(len(transcripts), 2)
        self.assertIn("2025-12-04", transcripts)
        self.assertIn("2025-12-12", transcripts)


class TestInsightExtractor(TestCase):
    """Test InsightExtractor functionality."""

    def setUp(self):
        """Create test transcripts."""
        # Create mock transcripts with relevant content
        self.transcripts = {
            "2025-12-04": ChatTranscript(
                date="2025-12-04",
                filepath="test.txt",
                messages=[
                    ChatMessage("Sergey Semerikov", "19:00:00",
                               "The first cluster contains Unity 3D and virtual reality.",
                               is_advisor=True),
                    ChatMessage("Bohdan Ostashchenko", "19:00:30",
                               "What about game development?"),
                    ChatMessage("Sergey Semerikov", "19:01:00",
                               "Game development is central. Use the Scopus methodology.",
                               is_advisor=True),
                ]
            ),
            "2025-12-16": ChatTranscript(
                date="2025-12-16",
                filepath="test2.txt",
                messages=[
                    ChatMessage("Sergey Semerikov", "19:00:00",
                               "The clustering shows education and serious games together.",
                               is_advisor=True),
                ]
            )
        }

        self.extractor = InsightExtractor(self.transcripts)

    def test_extract_cluster_discussions(self):
        """Test cluster discussion extraction."""
        clusters = self.extractor.extract_cluster_discussions()

        self.assertGreater(len(clusters), 0)
        self.assertIsInstance(clusters[0], ClusterInterpretation)

        # Check that clusters have required fields
        for cluster in clusters:
            self.assertIsNotNone(cluster.cluster_id)
            self.assertIsNotNone(cluster.theme)
            self.assertGreater(len(cluster.keywords), 0)

    def test_extract_methodology_guidance(self):
        """Test methodology insight extraction."""
        insights = self.extractor.extract_methodology_guidance()

        self.assertGreater(len(insights), 0)
        self.assertIsInstance(insights[0], ResearchInsight)

        # Should find the Scopus methodology mention
        categories = [i.category for i in insights]
        self.assertIn("Database Usage", categories)

    def test_extract_key_themes(self):
        """Test theme extraction."""
        themes = self.extractor.extract_key_themes()

        self.assertIsInstance(themes, dict)
        # Should find common themes mentioned in test data
        # Note: themes are lowercase
        theme_keys = [k.lower() for k in themes.keys()]
        self.assertTrue(any('unity' in k for k in theme_keys) or
                       any('game' in k for k in theme_keys))


class TestDiscussionGenerator(TestCase):
    """Test DiscussionGenerator functionality."""

    def setUp(self):
        """Create test data for discussion generation."""
        self.clusters = [
            ClusterInterpretation(
                cluster_id=1,
                color="Red",
                keywords=["unity 3d", "virtual reality", "game development"],
                theme="Immersive Game Development",
                description="VR/AR game development cluster.",
                advisor_insights=["Unity enables VR development."]
            ),
            ClusterInterpretation(
                cluster_id=2,
                color="Green",
                keywords=["education", "serious games"],
                theme="Educational Gaming",
                description="Educational game design cluster.",
                advisor_insights=["Serious games for learning."]
            ),
        ]

        self.insights = [
            ResearchInsight(
                category="Methodology",
                content="Use Scopus for reliable sources.",
                source_date="2025-12-04",
                speaker="Advisor",
                context="Meeting discussion"
            ),
            ResearchInsight(
                category="Reproducibility",
                content="Document all steps like a recipe.",
                source_date="2025-12-04",
                speaker="Advisor",
                context="Meeting discussion"
            ),
        ]

        self.themes = {
            "unity": 15,
            "game": 12,
            "education": 8,
            "vr": 5,
        }

        self.generator = DiscussionGenerator(
            self.clusters, self.insights, self.themes
        )

    def test_generate_discussion_points(self):
        """Test discussion point generation."""
        points = self.generator.generate_discussion_points()

        self.assertGreater(len(points), 0)
        self.assertIsInstance(points[0], DiscussionPoint)

        # Check sections are present
        sections = [p.section for p in points]
        self.assertTrue(any("6.1" in s for s in sections))  # Cluster interpretations
        self.assertTrue(any("6.2" in s for s in sections))  # Methodology
        self.assertTrue(any("6.3" in s for s in sections))  # Research implications

    def test_export_to_json(self):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            result_path = self.generator.export_to_json(output_path)

            self.assertTrue(Path(result_path).exists())

            with open(result_path, 'r') as f:
                data = json.load(f)

            self.assertIn('clusters', data)
            self.assertIn('discussion_points', data)
            self.assertIn('themes', data)
            self.assertEqual(len(data['clusters']), 2)
        finally:
            Path(output_path).unlink()

    def test_export_to_markdown(self):
        """Test Markdown export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name

        try:
            result_path = self.generator.export_to_markdown(output_path)

            self.assertTrue(Path(result_path).exists())

            content = Path(result_path).read_text()

            # Check Markdown structure
            self.assertIn("# Discussion Points", content)
            self.assertIn("## 6.1 Thematic Cluster Interpretations", content)
            self.assertIn("| Theme | Frequency |", content)  # Theme table
            self.assertIn("Cluster 1", content)
        finally:
            Path(output_path).unlink()


class TestIntegration(TestCase):
    """Integration tests for full chat analysis pipeline."""

    def setUp(self):
        """Set up test directories."""
        self.chats_dir = Path("/home/cc/claude_code/capstone_project/data/chats")
        self.temp_output = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_output)

    def test_full_analysis_pipeline(self):
        """Test complete chat analysis pipeline with real data."""
        if not self.chats_dir.exists():
            self.skipTest("Chats directory not found")

        results = run_chat_analysis(str(self.chats_dir), self.temp_output)

        # Check transcripts were parsed
        self.assertGreater(len(results['transcripts']), 0)

        # Check clusters were identified
        self.assertGreater(len(results['clusters']), 0)

        # Check output files were created
        self.assertGreater(len(results['output_files']), 0)
        for file_path in results['output_files']:
            self.assertTrue(Path(file_path).exists())

        # Check specific output files
        expected_files = ['chat_insights.json', 'discussion_points.md', 'analysis_summary.json']
        for filename in expected_files:
            self.assertTrue((Path(self.temp_output) / filename).exists(),
                           f"Expected file not found: {filename}")


class TestClusterInterpretation(TestCase):
    """Test ClusterInterpretation dataclass."""

    def test_cluster_creation(self):
        """Test creating a cluster interpretation."""
        cluster = ClusterInterpretation(
            cluster_id=1,
            color="Red",
            keywords=["unity", "vr", "game"],
            theme="VR Game Development",
            description="Development of VR games.",
            advisor_insights=["VR is important.", "Use Unity."]
        )

        self.assertEqual(cluster.cluster_id, 1)
        self.assertEqual(len(cluster.keywords), 3)
        self.assertEqual(len(cluster.advisor_insights), 2)


class TestDiscussionPoint(TestCase):
    """Test DiscussionPoint dataclass."""

    def test_point_creation(self):
        """Test creating a discussion point."""
        point = DiscussionPoint(
            section="6.1 Thematic Analysis",
            title="Unity in Education",
            content="Unity is widely used for educational games.",
            supporting_quotes=["Quote 1", "Quote 2"],
            keywords_mentioned=["unity", "education"]
        )

        self.assertEqual(point.section, "6.1 Thematic Analysis")
        self.assertEqual(len(point.supporting_quotes), 2)
        self.assertEqual(len(point.keywords_mentioned), 2)


if __name__ == '__main__':
    main(verbosity=2)
