"""
Simplified Metrics Service
Tracks key performance indicators without excessive complexity
"""

import logging
import time
from typing import Dict, Any, Optional
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime

from config.feature_flags import is_enabled

logger = logging.getLogger(__name__)


class SimpleMetrics:
    """Lightweight metrics tracking"""

    def __init__(self):
        self.counters = defaultdict(int)
        self.latencies = defaultdict(list)
        self.success_rates = defaultdict(lambda: {"success": 0, "total": 0})
        self.last_reset = datetime.now()

        # Testing-specific metrics
        self.outcome_agreements = {"agreed": 0, "total": 0}
        self.top1_accepts = {"accepted": 0, "total": 0}
        self.dont_repeat_failures = {"avoided": 0, "total": 0}
        self.task_outcomes = []

    def increment(self, name: str, value: int = 1):
        """Increment a counter"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.counters[name] += value

    def record_latency(self, operation: str, duration_ms: float):
        """Record operation latency"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.latencies[operation].append(duration_ms)
        # Keep bounded
        if len(self.latencies[operation]) > 100:
            self.latencies[operation] = self.latencies[operation][-100:]

    def record_success(self, operation: str, success: bool):
        """Record operation success/failure"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.success_rates[operation]["total"] += 1
        if success:
            self.success_rates[operation]["success"] += 1

    def record_outcome_agreement(self, expected: str, actual: str):
        """Record whether predicted outcome matched actual"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.outcome_agreements["total"] += 1
        if expected == actual:
            self.outcome_agreements["agreed"] += 1

    def record_top1_accept(self, accepted: bool):
        """Record whether first suggestion was accepted"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.top1_accepts["total"] += 1
        if accepted:
            self.top1_accepts["accepted"] += 1

    def record_dont_repeat_failure(self, avoided: bool):
        """Record whether we avoided repeating a failed solution"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.dont_repeat_failures["total"] += 1
        if avoided:
            self.dont_repeat_failures["avoided"] += 1

    def record_task_outcome(self, task_id: str, expected: str, actual: str, confidence: float, latency_ms: float):
        """Record complete task outcome for analysis"""
        if not is_enabled("ENABLE_METRICS"):
            return
        self.task_outcomes.append({
            'task_id': task_id,
            'expected': expected,
            'actual': actual,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'agreement': expected == actual,
            'timestamp': datetime.now().isoformat()
        })
        # Keep bounded
        if len(self.task_outcomes) > 1000:
            self.task_outcomes = self.task_outcomes[-1000:]

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {
            "counters": dict(self.counters),
            "success_rates": {},
            "latencies": {},
            "since": self.last_reset.isoformat()
        }

        # Calculate success rates
        for op, stats in self.success_rates.items():
            if stats["total"] > 0:
                summary["success_rates"][op] = {
                    "rate": stats["success"] / stats["total"],
                    "total": stats["total"]
                }

        # Calculate latency stats
        for op, values in self.latencies.items():
            if values:
                sorted_vals = sorted(values)
                summary["latencies"][op] = {
                    "avg": sum(values) / len(values),
                    "p50": sorted_vals[len(sorted_vals) // 2],
                    "p95": sorted_vals[int(len(sorted_vals) * 0.95)] if len(sorted_vals) > 1 else sorted_vals[0],
                    "max": max(values)
                }

        # Add testing-specific metrics
        if self.outcome_agreements["total"] > 0:
            summary["outcome_agreement_rate"] = self.outcome_agreements["agreed"] / self.outcome_agreements["total"]
        else:
            summary["outcome_agreement_rate"] = 0.0

        if self.top1_accepts["total"] > 0:
            summary["top1_accept_rate"] = self.top1_accepts["accepted"] / self.top1_accepts["total"]
        else:
            summary["top1_accept_rate"] = 0.0

        if self.dont_repeat_failures["total"] > 0:
            summary["dont_repeat_failed_fix_rate"] = self.dont_repeat_failures["avoided"] / self.dont_repeat_failures["total"]
        else:
            summary["dont_repeat_failed_fix_rate"] = 1.0  # Default to good behavior

        # Calculate p95 latency across all task outcomes
        if self.task_outcomes:
            all_latencies = [t["latency_ms"] for t in self.task_outcomes]
            sorted_latencies = sorted(all_latencies)
            p95_index = int(len(sorted_latencies) * 0.95)
            summary["p95_latency_ms"] = sorted_latencies[min(p95_index, len(sorted_latencies)-1)]
        else:
            summary["p95_latency_ms"] = 0.0

        # Include raw task outcomes for detailed analysis
        summary["task_outcomes"] = self.task_outcomes

        return summary

    def reset(self):
        """Reset all metrics"""
        self.counters.clear()
        self.latencies.clear()
        self.success_rates.clear()
        self.outcome_agreements = {"agreed": 0, "total": 0}
        self.top1_accepts = {"accepted": 0, "total": 0}
        self.dont_repeat_failures = {"avoided": 0, "total": 0}
        self.task_outcomes = []
        self.last_reset = datetime.now()

    def export_json(self, filepath: str):
        """Export metrics to JSON file"""
        import json
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def export_csv(self, filepath: str):
        """Export task outcomes to CSV"""
        import csv
        if not self.task_outcomes:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.task_outcomes[0].keys())
            writer.writeheader()
            writer.writerows(self.task_outcomes)


# Global instance
_metrics = SimpleMetrics()


def get_metrics() -> SimpleMetrics:
    """Get global metrics instance"""
    return _metrics


def track_performance(operation: str):
    """Decorator to track function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_enabled("ENABLE_METRICS"):
                return await func(*args, **kwargs)

            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                _metrics.record_latency(operation, duration_ms)
                _metrics.record_success(operation, True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _metrics.record_latency(operation, duration_ms)
                _metrics.record_success(operation, False)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_enabled("ENABLE_METRICS"):
                return func(*args, **kwargs)

            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                _metrics.record_latency(operation, duration_ms)
                _metrics.record_success(operation, True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                _metrics.record_latency(operation, duration_ms)
                _metrics.record_success(operation, False)
                raise

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator