"""
Storyteller Simulator - Automate multi-visit storytelling conversations

Simulates a user having repeated conversations with Roampal about a story.
Tracks outcomes and measures learning over time.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StorytellerVisit:
    """Single visit/interaction in a storytelling conversation"""
    id: str
    visit_number: int
    story_title: str
    domain: str
    sensei_id: str
    situation: str  # User's question/request
    requires_memory: bool
    ideal_response_elements: List[str] = None
    sensei_preferences: Dict[str, List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorytellerVisit':
        """Load visit from JSON data"""
        return cls(
            id=data['id'],
            visit_number=data['visit_number'],
            story_title=data['story_title'],
            domain=data['domain'],
            sensei_id=data['sensei_id'],
            situation=data['situation'],
            requires_memory=data.get('requires_memory', False),
            ideal_response_elements=data.get('ideal_response_elements', []),
            sensei_preferences=data.get('sensei_preferences', {})
        )


class StorytellerSimulator:
    """
    Simulates multi-visit storytelling conversations.

    Loads storyteller dataset and provides methods to:
    - Get all visits for a story (chronological order)
    - Simulate user queries
    - Evaluate if retrieved memories were helpful
    """

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to conversation_dataset_storytellers_150.json
        """
        self.dataset_path = Path(dataset_path)
        self.visits = []
        self.stories = {}  # story_title -> list of visits
        self._load_dataset()

    def _load_dataset(self):
        """Load storyteller dataset from JSON"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse visits
        for item in data:
            visit = StorytellerVisit.from_dict(item)
            self.visits.append(visit)

            # Group by story
            if visit.story_title not in self.stories:
                self.stories[visit.story_title] = []
            self.stories[visit.story_title].append(visit)

        # Sort each story's visits by visit_number
        for story_title in self.stories:
            self.stories[story_title].sort(key=lambda v: v.visit_number)

    def get_story_titles(self) -> List[str]:
        """Get list of all story titles"""
        return list(self.stories.keys())

    def get_visits_for_story(self, story_title: str) -> List[StorytellerVisit]:
        """
        Get all visits for a specific story, in chronological order.

        Args:
            story_title: Title of story

        Returns:
            List of visits (sorted by visit_number)
        """
        return self.stories.get(story_title, [])

    def get_story_metadata(self, story_title: str) -> Dict[str, Any]:
        """
        Get metadata about a story.

        Returns:
            Dict with domain, sensei_id, num_visits, etc.
        """
        visits = self.get_visits_for_story(story_title)
        if not visits:
            return {}

        first_visit = visits[0]
        return {
            'story_title': story_title,
            'domain': first_visit.domain,
            'sensei_id': first_visit.sensei_id,
            'num_visits': len(visits),
            'requires_memory_count': sum(1 for v in visits if v.requires_memory)
        }

    def evaluate_memory_helpfulness(
        self,
        retrieved_memories: List[Dict[str, Any]],
        visit: StorytellerVisit,
        threshold: float = 0.3
    ) -> bool:
        """
        Evaluate if retrieved memories were helpful for this visit.

        Args:
            retrieved_memories: Memories retrieved by system
            visit: Current visit being processed
            threshold: Minimum match ratio to be considered helpful

        Returns:
            True if memories were helpful
        """
        if not retrieved_memories:
            return False

        # Check if any memory contains ideal response elements
        if visit.ideal_response_elements:
            matches = 0
            all_content = ' '.join(
                m.get('content', '') + ' ' + m.get('metadata', {}).get('text', '')
                for m in retrieved_memories
            ).lower()

            for element in visit.ideal_response_elements:
                if element.lower() in all_content:
                    matches += 1

            match_ratio = matches / len(visit.ideal_response_elements)
            if match_ratio >= threshold:
                return True

        # Check if memories align with sensei preferences
        if visit.sensei_preferences:
            loves = visit.sensei_preferences.get('loves', [])
            hates = visit.sensei_preferences.get('hates', [])

            all_content = ' '.join(
                m.get('content', '') + ' ' + m.get('metadata', {}).get('text', '')
                for m in retrieved_memories
            ).lower()

            # Positive signals (loves)
            love_matches = sum(1 for love in loves if love.lower() in all_content)

            # Negative signals (hates)
            hate_matches = sum(1 for hate in hates if hate.lower() in all_content)

            # Helpful if has preferences and minimal hates
            if love_matches > 0 and hate_matches == 0:
                return True

        # If requires memory, check if any previous visit content is retrieved
        if visit.requires_memory and visit.visit_number > 1:
            # At minimum, should retrieve SOMETHING if memory is required
            return len(retrieved_memories) > 0

        return False

    def create_test_questions(
        self,
        story_title: str,
        visits_completed: List[StorytellerVisit]
    ) -> List[Dict[str, Any]]:
        """
        Generate test questions to evaluate system knowledge.

        Args:
            story_title: Story being tested
            visits_completed: Visits that have been processed

        Returns:
            List of test questions with expected answers
        """
        if not visits_completed:
            return []

        first_visit = visits_completed[0]
        questions = []

        # Q1: Genre/domain recall
        questions.append({
            'question': f"What genre is '{story_title}'?",
            'expected_answers': [first_visit.domain],
            'type': 'domain_recall'
        })

        # Q2: Sensei preference recall
        if first_visit.sensei_preferences:
            loves = first_visit.sensei_preferences.get('loves', [])
            if loves:
                questions.append({
                    'question': f"What does {first_visit.sensei_id} prefer in stories?",
                    'expected_answers': loves[:2],  # Top 2 preferences
                    'type': 'preference_recall'
                })

            hates = first_visit.sensei_preferences.get('hates', [])
            if hates:
                questions.append({
                    'question': f"What should I avoid in '{story_title}'?",
                    'expected_answers': hates[:2],
                    'type': 'constraint_recall'
                })

        # Q3: Situation recall (if multiple visits)
        if len(visits_completed) >= 3:
            mid_visit = visits_completed[len(visits_completed) // 2]
            questions.append({
                'question': f"What did I ask about in visit {mid_visit.visit_number}?",
                'expected_answers': [mid_visit.situation[:50]],  # First 50 chars
                'type': 'conversation_history'
            })

        # Q4: Story title recall
        questions.append({
            'question': "What story am I working on?",
            'expected_answers': [story_title],
            'type': 'story_recall'
        })

        return questions

    def get_stories_by_domain(self, domain: str) -> List[str]:
        """Get story titles in a specific domain"""
        return [
            title for title, visits in self.stories.items()
            if visits and visits[0].domain == domain
        ]

    def get_all_domains(self) -> List[str]:
        """Get unique list of all domains"""
        domains = set()
        for visits in self.stories.values():
            if visits:
                domains.add(visits[0].domain)
        return sorted(list(domains))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the dataset"""
        return {
            'total_visits': len(self.visits),
            'total_stories': len(self.stories),
            'domains': self.get_all_domains(),
            'visits_per_story': {
                title: len(visits)
                for title, visits in self.stories.items()
            },
            'memory_required_visits': sum(1 for v in self.visits if v.requires_memory)
        }


def load_storyteller_dataset(dataset_path: str = None) -> StorytellerSimulator:
    """
    Convenience function to load storyteller dataset.

    Args:
        dataset_path: Path to dataset JSON. If None, looks in standard location.

    Returns:
        StorytellerSimulator instance
    """
    if dataset_path is None:
        # Default to standard location
        dataset_path = Path(__file__).parent.parent.parent / "conversation_dataset_storytellers_150.json"

    return StorytellerSimulator(str(dataset_path))
