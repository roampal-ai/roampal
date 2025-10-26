"""
Outcome Tracking System
Tracks what actually worked and updates memory scores accordingly
"""

import logging
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)

OutcomeType = Literal["worked", "failed", "partial", "unknown"]


class OutcomeTracker:
    """
    Tracks outcomes of memory usage and solution attempts
    Enables learning from what actually worked
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.db_path = self.data_dir / "outcomes.db"
        self._init_database()

    def _init_database(self):
        """Initialize outcome tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Outcomes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                query TEXT,
                response TEXT,
                outcome TEXT CHECK(outcome IN ('worked', 'failed', 'partial', 'unknown')),
                confidence REAL DEFAULT 0.5,
                user_feedback TEXT,
                implicit_signal TEXT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(doc_id, collection, timestamp)
            )
        ''')

        # Pattern success table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_success (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                problem_signature TEXT,
                solution_signature TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                partial_count INTEGER DEFAULT 0,
                contexts TEXT,  -- JSON array of contexts where it worked
                last_outcome TEXT,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success_rate REAL GENERATED ALWAYS AS
                    (CAST(success_count AS REAL) / NULLIF(success_count + failure_count + partial_count, 0)) STORED
            )
        ''')

        # Implicit signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS implicit_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                signal_type TEXT,  -- 'silence', 'new_topic', 'copy_code', 'run_command'
                previous_response TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    async def record_outcome(
        self,
        doc_id: str,
        collection: str,
        outcome: OutcomeType,
        query: Optional[str] = None,
        response: Optional[str] = None,
        confidence: float = 0.5,
        user_feedback: Optional[str] = None,
        implicit_signal: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Record the outcome of using a memory or solution

        Args:
            doc_id: Document/memory ID that was used
            collection: Which collection it came from
            outcome: Whether it worked, failed, partially worked, or unknown
            query: Original query
            response: Response generated
            confidence: Confidence in the outcome assessment
            user_feedback: Explicit user feedback if available
            implicit_signal: Implicit signal detected (silence, new topic, etc.)
            session_id: Session identifier

        Returns:
            Outcome record ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO outcomes
                (doc_id, collection, query, response, outcome, confidence,
                 user_feedback, implicit_signal, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (doc_id, collection, query, response, outcome, confidence,
                  user_feedback, implicit_signal, session_id))

            outcome_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Recorded outcome {outcome_id}: {doc_id} -> {outcome} (confidence: {confidence})")
            return outcome_id

        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate outcome record attempted for {doc_id}")
            return -1
        finally:
            conn.close()

    async def record_pattern_outcome(
        self,
        problem: str,
        solution: str,
        outcome: OutcomeType,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record outcome for a problem->solution pattern

        Args:
            problem: Problem description/signature
            solution: Solution description/signature
            outcome: Whether the solution worked
            context: Additional context about when/why it worked
        """
        # Create hash for pattern
        pattern_hash = hash(f"{problem}::{solution}") % (10 ** 8)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if pattern exists
        cursor.execute('''
            SELECT id, success_count, failure_count, partial_count, contexts
            FROM pattern_success
            WHERE pattern_hash = ?
        ''', (str(pattern_hash),))

        result = cursor.fetchone()

        if result:
            # Update existing pattern
            pattern_id, success_count, failure_count, partial_count, contexts_json = result

            # Update counts
            if outcome == "worked":
                success_count += 1
            elif outcome == "failed":
                failure_count += 1
            elif outcome == "partial":
                partial_count += 1

            # Update contexts
            contexts = json.loads(contexts_json) if contexts_json else []
            if context:
                contexts.append(context)
                # Keep only last 10 contexts
                contexts = contexts[-10:]

            cursor.execute('''
                UPDATE pattern_success
                SET success_count = ?, failure_count = ?, partial_count = ?,
                    contexts = ?, last_outcome = ?, last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (success_count, failure_count, partial_count,
                  json.dumps(contexts), outcome, pattern_id))

        else:
            # Create new pattern record
            initial_counts = {"worked": 1, "failed": 0, "partial": 0}
            if outcome == "failed":
                initial_counts = {"worked": 0, "failed": 1, "partial": 0}
            elif outcome == "partial":
                initial_counts = {"worked": 0, "failed": 0, "partial": 1}

            contexts = [context] if context else []

            cursor.execute('''
                INSERT INTO pattern_success
                (pattern_hash, problem_signature, solution_signature,
                 success_count, failure_count, partial_count,
                 contexts, last_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (str(pattern_hash), problem, solution,
                  initial_counts["worked"], initial_counts["failed"],
                  initial_counts["partial"], json.dumps(contexts), outcome))

        conn.commit()
        conn.close()

        logger.info(f"Recorded pattern outcome: {problem[:50]} -> {solution[:50]} = {outcome}")

    async def record_implicit_signal(
        self,
        signal_type: str,
        session_id: str,
        previous_response: Optional[str] = None,
        confidence: float = 0.5
    ):
        """
        Record implicit signals like user silence, topic change, etc.

        Args:
            signal_type: Type of implicit signal
            session_id: Session identifier
            previous_response: Previous assistant response
            confidence: Confidence in signal interpretation
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO implicit_signals
            (session_id, signal_type, previous_response, confidence)
            VALUES (?, ?, ?, ?)
        ''', (session_id, signal_type, previous_response, confidence))

        conn.commit()
        conn.close()

        logger.info(f"Recorded implicit signal: {signal_type} (confidence: {confidence})")

    async def infer_outcome_from_signals(
        self,
        session_id: str,
        lookback_messages: int = 5
    ) -> Dict[str, Any]:
        """
        Infer outcome from implicit signals

        Args:
            session_id: Session to analyze
            lookback_messages: How many messages to look back

        Returns:
            Inferred outcome with confidence
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent signals
        cursor.execute('''
            SELECT signal_type, confidence, timestamp
            FROM implicit_signals
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, lookback_messages))

        signals = cursor.fetchall()
        conn.close()

        if not signals:
            return {"outcome": "unknown", "confidence": 0.0, "reasoning": "No signals"}

        # Analyze signals
        latest_signal = signals[0][0] if signals else None

        # Signal interpretation rules
        if latest_signal == "silence":
            # User went silent - probably worked
            return {
                "outcome": "worked",
                "confidence": 0.6,
                "reasoning": "User went silent after solution"
            }
        elif latest_signal == "new_topic":
            # User moved to new topic - previous probably worked
            return {
                "outcome": "worked",
                "confidence": 0.7,
                "reasoning": "User moved to new topic"
            }
        elif latest_signal == "copy_code":
            # User copied code - definitely using it
            return {
                "outcome": "worked",
                "confidence": 0.85,
                "reasoning": "User copied the code"
            }
        elif latest_signal == "error_repeat":
            # Same error again - solution failed
            return {
                "outcome": "failed",
                "confidence": 0.8,
                "reasoning": "Same error reported again"
            }

        return {"outcome": "unknown", "confidence": 0.3, "reasoning": "Unclear signals"}

    async def get_pattern_success_rate(
        self,
        problem: str,
        solution: str
    ) -> Dict[str, Any]:
        """
        Get success rate for a problem->solution pattern

        Args:
            problem: Problem signature
            solution: Solution signature

        Returns:
            Success statistics
        """
        pattern_hash = hash(f"{problem}::{solution}") % (10 ** 8)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT success_count, failure_count, partial_count, success_rate, contexts
            FROM pattern_success
            WHERE pattern_hash = ?
        ''', (str(pattern_hash),))

        result = cursor.fetchone()
        conn.close()

        if result:
            success, failure, partial, rate, contexts_json = result
            total = success + failure + partial

            return {
                "pattern": f"{problem} -> {solution}",
                "success_count": success,
                "failure_count": failure,
                "partial_count": partial,
                "total_attempts": total,
                "success_rate": rate,
                "contexts": json.loads(contexts_json) if contexts_json else []
            }

        return {
            "pattern": f"{problem} -> {solution}",
            "success_count": 0,
            "failure_count": 0,
            "partial_count": 0,
            "total_attempts": 0,
            "success_rate": 0.0,
            "contexts": []
        }

    async def get_best_patterns(
        self,
        min_attempts: int = 3,
        min_success_rate: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get the most successful patterns

        Args:
            min_attempts: Minimum number of attempts to consider
            min_success_rate: Minimum success rate to include

        Returns:
            List of successful patterns
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT problem_signature, solution_signature, success_count,
                   failure_count, partial_count, success_rate, contexts
            FROM pattern_success
            WHERE (success_count + failure_count + partial_count) >= ?
              AND success_rate >= ?
            ORDER BY success_rate DESC, success_count DESC
            LIMIT 20
        ''', (min_attempts, min_success_rate))

        results = cursor.fetchall()
        conn.close()

        patterns = []
        for row in results:
            problem, solution, success, failure, partial, rate, contexts_json = row
            patterns.append({
                "problem": problem,
                "solution": solution,
                "success_count": success,
                "failure_count": failure,
                "partial_count": partial,
                "success_rate": rate,
                "total_attempts": success + failure + partial,
                "contexts": json.loads(contexts_json) if contexts_json else []
            })

        return patterns