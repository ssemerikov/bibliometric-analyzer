#!/usr/bin/env python3
"""
Chat Transcript Analyzer for Advisor-Student Meeting Transcripts

This module processes chat transcripts from advisor-student meetings to extract:
- Key research insights and interpretations
- Cluster analysis discussions
- Methodological guidance
- Discussion points for thesis

The transcripts are in mixed Russian/Ukrainian language and require
careful processing to extract meaningful academic content.

Author: Generated with Claude Opus 4.5 assistance
Date: 2026-01-09
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime


@dataclass
class ChatMessage:
    """Represents a single message in a chat transcript."""
    speaker: str
    timestamp: str
    content: str
    is_advisor: bool = False


@dataclass
class ChatTranscript:
    """Represents a complete chat transcript."""
    date: str
    filepath: str
    messages: List[ChatMessage] = field(default_factory=list)
    duration_minutes: float = 0.0
    advisor_messages: int = 0
    student_messages: int = 0


@dataclass
class ClusterInterpretation:
    """Represents an interpretation of a VOSViewer cluster."""
    cluster_id: int
    color: str
    keywords: List[str]
    theme: str
    description: str
    advisor_insights: List[str]


@dataclass
class ResearchInsight:
    """Represents a key research insight extracted from discussions."""
    category: str  # methodology, interpretation, recommendation, etc.
    content: str
    source_date: str
    speaker: str
    context: str


@dataclass
class DiscussionPoint:
    """Represents a structured discussion point for the thesis."""
    section: str  # e.g., "6.1 Thematic Interpretation"
    title: str
    content: str
    supporting_quotes: List[str]
    keywords_mentioned: List[str]


class ChatParser:
    """Parser for advisor-student chat transcripts."""

    # Pattern to match chat message format: [Speaker] HH:MM:SS
    MESSAGE_PATTERN = re.compile(r'\[([^\]]+)\]\s+(\d{2}:\d{2}:\d{2})\s*\n(.+?)(?=\[|$)', re.DOTALL)

    # Known advisors (based on transcript analysis)
    ADVISORS = ['Sergey Semerikov', 'Semerikov']

    def __init__(self, chats_dir: str):
        self.chats_dir = Path(chats_dir)

    def parse_file(self, filepath: Path) -> ChatTranscript:
        """Parse a single chat transcript file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract date from filename
        date = filepath.stem  # e.g., "2025-12-04"

        messages = []
        for match in self.MESSAGE_PATTERN.finditer(content):
            speaker = match.group(1).strip()
            timestamp = match.group(2).strip()
            text = match.group(3).strip()

            # Determine if speaker is advisor
            is_advisor = any(adv in speaker for adv in self.ADVISORS)

            messages.append(ChatMessage(
                speaker=speaker,
                timestamp=timestamp,
                content=text,
                is_advisor=is_advisor
            ))

        # Calculate duration if messages exist
        duration = 0.0
        if len(messages) >= 2:
            try:
                start = datetime.strptime(messages[0].timestamp, "%H:%M:%S")
                end = datetime.strptime(messages[-1].timestamp, "%H:%M:%S")
                duration = (end - start).total_seconds() / 60
            except:
                pass

        # Count messages by role
        advisor_count = sum(1 for m in messages if m.is_advisor)
        student_count = len(messages) - advisor_count

        return ChatTranscript(
            date=date,
            filepath=str(filepath),
            messages=messages,
            duration_minutes=duration,
            advisor_messages=advisor_count,
            student_messages=student_count
        )

    def parse_all(self) -> Dict[str, ChatTranscript]:
        """Parse all chat transcripts in directory."""
        transcripts = {}

        for filepath in sorted(self.chats_dir.glob("*.txt")):
            transcript = self.parse_file(filepath)
            transcripts[transcript.date] = transcript
            print(f"Parsed {filepath.name}: {len(transcript.messages)} messages, "
                  f"{transcript.duration_minutes:.1f} min")

        return transcripts


class InsightExtractor:
    """Extracts research insights from chat transcripts."""

    # Keywords related to VOSViewer and bibliometric analysis
    VOSVIEWER_KEYWORDS = [
        'vosviewer', 'вос виюрер', 'воз виюрер', 'возвьюер', 'вос вюер',
        'cluster', 'кластер', 'карта', 'map', 'сеть', 'network',
        'coauthorship', 'co-authorship', 'citation', 'coupling', 'co-citation',
        'keyword', 'ключевые слова', 'слова'
    ]

    # Keywords for cluster/theme identification
    CLUSTER_KEYWORDS = [
        'unity', 'юники', 'юнити',
        'game', 'игра', 'игры', 'гейм',
        'virtual reality', 'виртуальная реальность', 'vr', 'вр',
        'augmented reality', 'дополненная реальность', 'ar', 'ар',
        'education', 'образование', 'обучение',
        'serious game', 'серьезные игры',
        'development', 'разработка', 'девелопмент',
        'gamification', 'геймификация',
        'motivation', 'мотивация',
        '3d', 'три д', 'трехмерный'
    ]

    def __init__(self, transcripts: Dict[str, ChatTranscript]):
        self.transcripts = transcripts

    def extract_cluster_discussions(self) -> List[ClusterInterpretation]:
        """Extract discussions about VOSViewer clusters."""
        clusters = []

        # Pre-defined cluster interpretations based on cosco5 (Author Keywords)
        # These are extracted from the actual chat discussions
        cluster_data = [
            {
                'id': 1,
                'color': 'Red',
                'keywords': ['augmented reality', 'education', 'game development',
                            'serious game', 'unity 3d', 'virtual reality'],
                'theme': '3D Immersive Technologies for Educational Game Development',
                'description': 'This cluster represents the intersection of immersive technologies '
                             '(VR/AR) with educational game development using Unity3D as the primary tool.'
            },
            {
                'id': 2,
                'color': 'Green',
                'keywords': ['game', 'gamification', 'motivation', 'unity',
                            'video game development'],
                'theme': 'General Game Development with Motivational Design',
                'description': 'This cluster focuses on general game development practices '
                             'emphasizing gamification and player motivation.'
            },
            {
                'id': 3,
                'color': 'Blue',
                'keywords': ['educational game', 'unity game engine'],
                'theme': 'Educational Game Development with Unity',
                'description': 'A focused cluster on creating educational games '
                             'using the Unity game engine.'
            },
            {
                'id': 4,
                'color': 'Yellow',
                'keywords': ['game design', 'game-based learning'],
                'theme': 'Game Design for Learning',
                'description': 'This cluster emphasizes the design aspects '
                             'of game-based learning approaches.'
            },
            {
                'id': 5,
                'color': 'Purple',
                'keywords': ['game ai', 'unity engine'],
                'theme': 'AI Integration in Game Engines',
                'description': 'This cluster represents the technical aspects '
                             'of AI implementation in Unity-based games.'
            }
        ]

        for data in cluster_data:
            # Find advisor insights about this cluster from transcripts
            insights = self._find_cluster_insights(data['keywords'])

            clusters.append(ClusterInterpretation(
                cluster_id=data['id'],
                color=data['color'],
                keywords=data['keywords'],
                theme=data['theme'],
                description=data['description'],
                advisor_insights=insights
            ))

        return clusters

    def _find_cluster_insights(self, keywords: List[str]) -> List[str]:
        """Find advisor comments mentioning specific keywords."""
        insights = []
        keyword_patterns = [re.compile(kw, re.IGNORECASE) for kw in keywords]

        for date, transcript in self.transcripts.items():
            for msg in transcript.messages:
                if not msg.is_advisor:
                    continue

                # Check if message mentions any cluster keywords
                if any(pattern.search(msg.content) for pattern in keyword_patterns):
                    # Clean and truncate the insight
                    insight = msg.content.strip()
                    if len(insight) > 200:
                        insight = insight[:200] + "..."
                    if insight:
                        insights.append(f"[{date}] {insight}")

        return insights[:5]  # Limit to 5 insights per cluster

    def extract_methodology_guidance(self) -> List[ResearchInsight]:
        """Extract methodological guidance from advisor comments."""
        insights = []

        # Methodology-related patterns
        method_patterns = [
            (r'методик|метод|procedure|процедур', 'Methodology'),
            (r'рецепт|recipe|воспроизводим', 'Reproducibility'),
            (r'skopus|scopus|скопус|база', 'Database Usage'),
            (r'поиск|search|искать', 'Search Strategy'),
            (r'карта|map|сеть|network', 'Network Analysis'),
            (r'кластер|cluster|группа', 'Clustering'),
            (r'файл|file|сохран', 'Documentation'),
        ]

        for date, transcript in self.transcripts.items():
            for msg in transcript.messages:
                if not msg.is_advisor:
                    continue

                for pattern, category in method_patterns:
                    if re.search(pattern, msg.content, re.IGNORECASE):
                        # Get context (surrounding text)
                        insight = ResearchInsight(
                            category=category,
                            content=msg.content.strip()[:300],
                            source_date=date,
                            speaker=msg.speaker,
                            context=f"From meeting on {date}"
                        )
                        insights.append(insight)
                        break  # Only categorize once per message

        return insights

    def extract_key_themes(self) -> Dict[str, int]:
        """Extract frequency of key themes across all transcripts."""
        theme_counts = defaultdict(int)

        for date, transcript in self.transcripts.items():
            for msg in transcript.messages:
                content_lower = msg.content.lower()

                for keyword in self.CLUSTER_KEYWORDS:
                    if keyword.lower() in content_lower:
                        theme_counts[keyword] += 1

        return dict(sorted(theme_counts.items(), key=lambda x: x[1], reverse=True))


class DiscussionGenerator:
    """Generates structured discussion points for the thesis."""

    def __init__(self, clusters: List[ClusterInterpretation],
                 insights: List[ResearchInsight],
                 themes: Dict[str, int]):
        self.clusters = clusters
        self.insights = insights
        self.themes = themes

    def generate_discussion_points(self) -> List[DiscussionPoint]:
        """Generate discussion points for Chapter 6."""
        points = []

        # 6.1 Thematic Cluster Interpretations
        for cluster in self.clusters:
            points.append(DiscussionPoint(
                section="6.1 Thematic Cluster Interpretations",
                title=f"Cluster {cluster.cluster_id} ({cluster.color}): {cluster.theme}",
                content=cluster.description,
                supporting_quotes=cluster.advisor_insights[:3],
                keywords_mentioned=cluster.keywords
            ))

        # 6.2 Methodological Reflections
        method_insights = [i for i in self.insights
                          if i.category in ['Methodology', 'Reproducibility', 'Database Usage']]

        if method_insights:
            points.append(DiscussionPoint(
                section="6.2 Methodological Reflections",
                title="Bibliometric Methodology",
                content="The research methodology was developed through iterative discussions "
                       "emphasizing reproducibility and systematic approaches to literature analysis.",
                supporting_quotes=[i.content[:200] for i in method_insights[:3]],
                keywords_mentioned=['methodology', 'scopus', 'vosviewer', 'reproducibility']
            ))

        # 6.3 Research Implications
        top_themes = list(self.themes.keys())[:10]
        points.append(DiscussionPoint(
            section="6.3 Research Implications",
            title="Key Themes in Game Development Education",
            content="The bibliometric analysis reveals that game development education "
                   "research centers around Unity-based development, immersive technologies "
                   "(VR/AR), and gamification as key pedagogical approaches.",
            supporting_quotes=[],
            keywords_mentioned=top_themes
        ))

        return points

    def export_to_json(self, output_path: str) -> str:
        """Export discussion points to JSON."""
        points = self.generate_discussion_points()

        output = {
            'clusters': [asdict(c) for c in self.clusters],
            'discussion_points': [asdict(p) for p in points],
            'themes': self.themes,
            'insights_summary': {
                'total': len(self.insights),
                'by_category': defaultdict(int)
            }
        }

        for insight in self.insights:
            output['insights_summary']['by_category'][insight.category] += 1

        output['insights_summary']['by_category'] = dict(output['insights_summary']['by_category'])

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output_path

    def export_to_markdown(self, output_path: str) -> str:
        """Export discussion points to Markdown for thesis integration."""
        points = self.generate_discussion_points()

        lines = [
            "# Discussion Points Extracted from Advisor Meetings",
            "",
            "This document contains structured discussion points extracted from",
            "advisor-student meeting transcripts for integration into Chapter 6 (Discussion).",
            "",
            "---",
            ""
        ]

        current_section = ""
        for point in points:
            if point.section != current_section:
                lines.append(f"## {point.section}")
                lines.append("")
                current_section = point.section

            lines.append(f"### {point.title}")
            lines.append("")
            lines.append(point.content)
            lines.append("")

            if point.keywords_mentioned:
                lines.append(f"**Keywords:** {', '.join(point.keywords_mentioned)}")
                lines.append("")

            if point.supporting_quotes:
                lines.append("**Supporting Evidence from Meetings:**")
                for quote in point.supporting_quotes:
                    # Clean up the quote
                    clean_quote = quote.replace('\n', ' ').strip()
                    if clean_quote:
                        lines.append(f"> {clean_quote[:200]}...")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Add theme frequency section
        lines.append("## Theme Frequency Analysis")
        lines.append("")
        lines.append("| Theme | Frequency |")
        lines.append("|-------|-----------|")
        for theme, count in list(self.themes.items())[:15]:
            lines.append(f"| {theme} | {count} |")
        lines.append("")

        # Add cluster summary
        lines.append("## Cluster Summary")
        lines.append("")
        lines.append("| Cluster | Color | Theme | Keywords |")
        lines.append("|---------|-------|-------|----------|")
        for cluster in self.clusters:
            keywords = ', '.join(cluster.keywords[:3])
            lines.append(f"| {cluster.cluster_id} | {cluster.color} | {cluster.theme} | {keywords}... |")
        lines.append("")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return output_path


def run_chat_analysis(chats_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run complete chat analysis pipeline."""
    print("=" * 60)
    print("CHAT TRANSCRIPT ANALYSIS PIPELINE")
    print("=" * 60)

    results = {
        'transcripts': {},
        'clusters': [],
        'insights': [],
        'themes': {},
        'output_files': []
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Parse chat transcripts
    print("\n[1/4] Parsing chat transcripts...")
    parser = ChatParser(chats_dir)
    transcripts = parser.parse_all()
    results['transcripts'] = {k: asdict(v) for k, v in transcripts.items()}

    total_messages = sum(len(t.messages) for t in transcripts.values())
    print(f"  Total: {len(transcripts)} transcripts, {total_messages} messages")

    # 2. Extract insights
    print("\n[2/4] Extracting research insights...")
    extractor = InsightExtractor(transcripts)

    clusters = extractor.extract_cluster_discussions()
    results['clusters'] = [asdict(c) for c in clusters]
    print(f"  Identified {len(clusters)} cluster interpretations")

    insights = extractor.extract_methodology_guidance()
    results['insights'] = [asdict(i) for i in insights]
    print(f"  Extracted {len(insights)} methodology insights")

    themes = extractor.extract_key_themes()
    results['themes'] = themes
    print(f"  Found {len(themes)} unique themes")

    # 3. Generate discussion points
    print("\n[3/4] Generating discussion points...")
    generator = DiscussionGenerator(clusters, insights, themes)

    # Export to JSON
    json_path = generator.export_to_json(str(output_path / "chat_insights.json"))
    results['output_files'].append(json_path)
    print(f"  Created: chat_insights.json")

    # Export to Markdown
    md_path = generator.export_to_markdown(str(output_path / "discussion_points.md"))
    results['output_files'].append(md_path)
    print(f"  Created: discussion_points.md")

    # 4. Generate summary statistics
    print("\n[4/4] Generating summary...")
    summary = {
        'total_transcripts': len(transcripts),
        'total_messages': total_messages,
        'total_duration_minutes': sum(t.duration_minutes for t in transcripts.values()),
        'advisor_message_ratio': sum(t.advisor_messages for t in transcripts.values()) / total_messages if total_messages > 0 else 0,
        'clusters_identified': len(clusters),
        'insights_extracted': len(insights),
        'unique_themes': len(themes),
        'top_themes': dict(list(themes.items())[:10])
    }

    summary_path = output_path / "analysis_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    results['output_files'].append(str(summary_path))
    print(f"  Created: analysis_summary.json")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Processed {summary['total_transcripts']} transcripts")
    print(f"  - {summary['total_messages']} total messages")
    print(f"  - {summary['total_duration_minutes']:.1f} minutes of discussion")
    print(f"  - {summary['insights_extracted']} methodology insights")
    print(f"  - Top themes: {', '.join(list(summary['top_themes'].keys())[:5])}")

    return results


if __name__ == "__main__":
    import sys

    # Default paths
    chats_dir = "/home/cc/claude_code/capstone_project/data/chats"
    output_dir = "/home/cc/claude_code/capstone_project/data/processed"

    if len(sys.argv) > 1:
        chats_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    results = run_chat_analysis(chats_dir, output_dir)
