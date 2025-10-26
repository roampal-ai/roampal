#!/usr/bin/env python3
"""
Canary Toggle Script - Gradual feature rollout with monitoring
Purpose: Enable features for subset of users/sessions to measure impact
Owner: @LoopSmith/production
Risk Level: Medium (affects production traffic)
TODO: Add A/B test statistical significance (expires: 2025-10-18)
"""

import argparse
import json
import os
import hashlib
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CanaryManager:
    """Manages canary deployments for features"""

    def __init__(self, config_path: str = "config/canary.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load canary configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "canaries": {},
            "history": [],
            "created": datetime.now().isoformat()
        }

    def save_config(self):
        """Save canary configuration"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def set_canary(self, feature: str, rate: float, description: str = "") -> bool:
        """Set canary rate for a feature"""
        if rate < 0 or rate > 1:
            print(f"[ERROR] Rate must be between 0 and 1, got {rate}")
            return False

        # Record history
        self.config["history"].append({
            "timestamp": datetime.now().isoformat(),
            "feature": feature,
            "old_rate": self.config["canaries"].get(feature, {}).get("rate", 0),
            "new_rate": rate,
            "description": description
        })

        # Update canary
        if rate == 0:
            # Remove canary if rate is 0
            if feature in self.config["canaries"]:
                del self.config["canaries"][feature]
                print(f"[REMOVED] Canary for {feature}")
        else:
            self.config["canaries"][feature] = {
                "rate": rate,
                "enabled_at": datetime.now().isoformat(),
                "description": description,
                "sessions_in_canary": 0,
                "sessions_in_control": 0
            }
            print(f"[SET] {feature} canary to {rate*100:.1f}%")

        # Also set environment variable
        env_key = f"ROAMPAL_{feature}_CANARY_RATE"
        os.environ[env_key] = str(rate)

        self.save_config()
        return True

    def is_in_canary(self, feature: str, session_id: str) -> bool:
        """Check if a session should get canary treatment"""
        if feature not in self.config["canaries"]:
            return False

        canary = self.config["canaries"][feature]
        rate = canary["rate"]

        if rate == 0:
            return False
        if rate == 1:
            return True

        # Deterministic assignment based on session ID
        hash_val = int(hashlib.md5(f"{feature}:{session_id}".encode()).hexdigest()[:8], 16)
        in_canary = (hash_val % 1000) < (rate * 1000)

        # Track assignment (in real system, this would be in database)
        if in_canary:
            canary["sessions_in_canary"] += 1
        else:
            canary["sessions_in_control"] += 1

        return in_canary

    def get_status(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """Get canary status"""
        if feature:
            if feature not in self.config["canaries"]:
                return {"error": f"No canary for {feature}"}
            return self.config["canaries"][feature]
        return self.config["canaries"]

    def graduate(self, feature: str) -> bool:
        """Graduate canary to 100% (full rollout)"""
        if feature not in self.config["canaries"]:
            print(f"[ERROR] No canary found for {feature}")
            return False

        return self.set_canary(feature, 1.0, "Graduated to full rollout")

    def rollback(self, feature: str) -> bool:
        """Rollback canary to 0% (disable)"""
        return self.set_canary(feature, 0.0, "Rolled back due to issues")

def main():
    parser = argparse.ArgumentParser(description='Manage canary feature rollouts')
    parser.add_argument('--feature', required=True, help='Feature flag name (e.g., ENABLE_KG)')
    parser.add_argument('--rate', type=float, help='Canary rate (0.0-1.0)')
    parser.add_argument('--status', action='store_true', help='Show canary status')
    parser.add_argument('--graduate', action='store_true', help='Graduate to 100%')
    parser.add_argument('--rollback', action='store_true', help='Rollback to 0%')
    parser.add_argument('--check', help='Check if session is in canary')
    parser.add_argument('--description', default='', help='Description of change')

    args = parser.parse_args()
    manager = CanaryManager()

    if args.status:
        status = manager.get_status(args.feature)
        print(json.dumps(status, indent=2))

    elif args.graduate:
        if manager.graduate(args.feature):
            print(f"✅ Graduated {args.feature} to full rollout")

    elif args.rollback:
        if manager.rollback(args.feature):
            print(f"⏪ Rolled back {args.feature}")

    elif args.check:
        in_canary = manager.is_in_canary(args.feature, args.check)
        print(f"Session {args.check}: {'IN CANARY' if in_canary else 'IN CONTROL'}")

    elif args.rate is not None:
        if manager.set_canary(args.feature, args.rate, args.description):
            print(f"✅ Updated {args.feature} canary rate to {args.rate*100:.1f}%")

            # Show example sessions
            print("\nExample session assignments:")
            for i in range(10):
                session = f"session_{i:03d}"
                in_canary = manager.is_in_canary(args.feature, session)
                print(f"  {session}: {'canary' if in_canary else 'control'}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()