#!/bin/bash
# Auto-Rollback Check Script
# Purpose: Automatically check metrics and rollback if degradation detected
# Owner: @LoopSmith/production
# Risk Level: High (can trigger automatic rollback)
# TODO: Add Slack/email notifications (expires: 2025-10-18)

set -e

# Configuration
METRICS_DIR="metrics"
BASELINE_FILE="metrics/production_baseline.json"
ROLLBACK_THRESHOLD=0.05  # 5% degradation triggers rollback
LOG_FILE="logs/auto_rollback.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting auto-rollback check..."

# Find latest metrics file
LATEST_METRICS=$(ls -t $METRICS_DIR/run_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_METRICS" ]; then
    log "ERROR: No metrics files found"
    exit 1
fi

if [ ! -f "$BASELINE_FILE" ]; then
    log "WARNING: No baseline file, using oldest metrics as baseline"
    BASELINE_FILE=$(ls -t $METRICS_DIR/run_*.json | tail -1)
fi

log "Comparing: $BASELINE_FILE vs $LATEST_METRICS"

# Run comparison
python testing/compare_metrics.py \
    --baseline "$BASELINE_FILE" \
    --current "$LATEST_METRICS" \
    --format json > /tmp/metrics_comparison.json

# Check for degradation
VERDICT=$(jq -r '.verdict' /tmp/metrics_comparison.json)
DEGRADATIONS=$(jq -r '.degradations[]' /tmp/metrics_comparison.json 2>/dev/null)

if [ "$VERDICT" == "FAIL" ]; then
    log "ERROR: Metrics degradation detected!"
    log "Degraded metrics: $DEGRADATIONS"

    # Check if auto-rollback is enabled
    if [ "$LOOPSMITH_AUTO_ROLLBACK" == "true" ]; then
        log "EXECUTING AUTOMATIC ROLLBACK"

        # Save current state for investigation
        cp .env .env.before_rollback
        cp config/feature_flags.json config/feature_flags.json.before_rollback

        # Execute rollback
        ./testing/rollback.sh

        # Log incident
        cat >> incidents.log <<EOF
================================================================================
Incident: Automatic Rollback
Time: $(date)
Trigger: $DEGRADATIONS
Baseline: $BASELINE_FILE
Current: $LATEST_METRICS
Comparison: $(cat /tmp/metrics_comparison.json)
================================================================================
EOF

        log "Rollback completed. Check incidents.log for details."

        # Send alert (if configured)
        if [ -n "$ALERT_WEBHOOK" ]; then
            curl -X POST "$ALERT_WEBHOOK" \
                -H "Content-Type: application/json" \
                -d "{\"text\": \"🚨 LoopSmith Auto-Rollback: $DEGRADATIONS degraded\"}"
        fi

        exit 2  # Special exit code for rollback
    else
        log "Auto-rollback disabled. Manual intervention required."
        exit 1
    fi
else
    log "✅ Metrics check passed - no degradation detected"

    # Update baseline if current is significantly better
    IMPROVEMENTS=$(jq -r '.improvements[]' /tmp/metrics_comparison.json 2>/dev/null | wc -l)
    if [ "$IMPROVEMENTS" -gt 2 ]; then
        log "Multiple improvements detected. Consider updating baseline."
        # Uncomment to auto-update baseline:
        # cp "$LATEST_METRICS" "$BASELINE_FILE"
        # log "Baseline updated to $LATEST_METRICS"
    fi
fi

# Cleanup
rm -f /tmp/metrics_comparison.json

log "Auto-rollback check completed"