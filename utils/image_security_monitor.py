"""
Image Security Monitor
Provides comprehensive logging and monitoring for image upload and processing operations.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class ImageSecurityEvent:
    timestamp: datetime
    event_type: str  # upload, analysis, error, security_violation
    shard_id: str
    user_id: str
    filename: Optional[str]
    file_size: Optional[int]
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any]
    processing_time_ms: Optional[float] = None

class ImageSecurityMonitor:
    """
    Monitor and log all image-related security events
    """
    
    def __init__(self):
        self.events_log = deque(maxlen=10000)  # Keep last 10k events in memory
        self.stats = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.log_file = Path("logs/image_security.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup dedicated logger for image security
        self.security_logger = logging.getLogger('image_security')
        self.security_logger.setLevel(logging.INFO)
        
        # File handler for persistent logging
        if not self.security_logger.handlers:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.security_logger.addHandler(handler)
    
    def log_event(self, event: ImageSecurityEvent):
        """Log an image security event"""
        try:
            # Add to memory
            self.events_log.append(event)
            
            # Update stats
            self.stats[f"{event.event_type}_total"] += 1
            if event.success:
                self.stats[f"{event.event_type}_success"] += 1
            else:
                self.stats[f"{event.event_type}_failure"] += 1
                self.error_counts[f"{event.shard_id}_{event.event_type}"] += 1
            
            # Log to file
            log_data = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "shard_id": event.shard_id,
                "user_id": event.user_id,
                "filename": event.filename,
                "file_size": event.file_size,
                "ip_address": event.ip_address,
                "user_agent": event.user_agent,
                "success": event.success,
                "processing_time_ms": event.processing_time_ms,
                "details": event.details
            }
            
            self.security_logger.info(json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Failed to log image security event: {e}")
    
    def log_upload_start(self, shard_id: str, user_id: str, filename: str, 
                        file_size: int, ip_address: str, user_agent: str):
        """Log image upload start"""
        event = ImageSecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="upload_start",
            shard_id=shard_id,
            user_id=user_id,
            filename=filename,
            file_size=file_size,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            details={"status": "started"}
        )
        self.log_event(event)
    
    def log_upload_success(self, shard_id: str, user_id: str, filename: str,
                          processing_time_ms: float, analysis_type: str,
                          ip_address: str, user_agent: str):
        """Log successful image upload"""
        event = ImageSecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="upload_success",
            shard_id=shard_id,
            user_id=user_id,
            filename=filename,
            file_size=None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=True,
            processing_time_ms=processing_time_ms,
            details={"analysis_type": analysis_type}
        )
        self.log_event(event)
    
    def log_security_violation(self, shard_id: str, user_id: str, 
                              violation_type: str, details: Dict[str, Any],
                              ip_address: str, user_agent: str):
        """Log security violation"""
        event = ImageSecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="security_violation",
            shard_id=shard_id,
            user_id=user_id,
            filename=details.get("filename"),
            file_size=details.get("file_size"),
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            details={"violation_type": violation_type, **details}
        )
        self.log_event(event)
        
        # Also log as WARNING
        logger.warning(f"SECURITY_VIOLATION: {violation_type} from {user_id}@{ip_address} on shard {shard_id}: {details}")
    
    def log_analysis_failure(self, shard_id: str, user_id: str, filename: str,
                           error_message: str, ip_address: str, user_agent: str):
        """Log image analysis failure"""
        event = ImageSecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="analysis_failure",
            shard_id=shard_id,
            user_id=user_id,
            filename=filename,
            file_size=None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            details={"error": error_message}
        )
        self.log_event(event)
    
    def log_chat_integration(self, shard_id: str, user_id: str, session_id: str,
                           has_image_context: bool, success: bool,
                           processing_time_ms: float, ip_address: str, user_agent: str):
        """Log chat integration with images"""
        event = ImageSecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="chat_integration",
            shard_id=shard_id,
            user_id=user_id,
            filename=None,
            file_size=None,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            processing_time_ms=processing_time_ms,
            details={
                "session_id": session_id,
                "has_image_context": has_image_context
            }
        )
        self.log_event(event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return {
            "total_events": len(self.events_log),
            "event_stats": dict(self.stats),
            "error_counts": dict(self.error_counts),
            "recent_events": len([e for e in self.events_log 
                                if (datetime.utcnow() - e.timestamp).seconds < 3600]),
            "top_errors": sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_recent_events(self, limit: int = 100) -> list:
        """Get recent events"""
        recent = list(self.events_log)[-limit:]
        return [asdict(event) for event in recent]
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Cleanup old log files"""
        try:
            import glob
            import os
            
            log_pattern = str(self.log_file.parent / "image_security.log.*")
            old_files = glob.glob(log_pattern)
            
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            for log_file in old_files:
                if os.path.getmtime(log_file) < cutoff_time:
                    os.remove(log_file)
                    logger.info(f"Cleaned up old log file: {log_file}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")

# Global monitor instance
image_security_monitor = ImageSecurityMonitor()