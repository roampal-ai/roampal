"""
Secure Session Manager
Handles cryptographically secure session token generation and validation
Ensures session isolation between users and shards
"""

import secrets
import hashlib
import hmac
import json
import time
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class SecureSessionManager:
    """
    Manages secure session tokens with cryptographic validation
    Ensures strict isolation between users and shards
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_length: int = 32,
        session_timeout_hours: int = 24,
        max_sessions_per_user: int = 10
    ):
        """
        Initialize the session manager
        
        Args:
            secret_key: Secret key for HMAC validation (generated if not provided)
            token_length: Length of random token in bytes
            session_timeout_hours: How long sessions remain valid
            max_sessions_per_user: Maximum concurrent sessions per user
        """
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_length = token_length
        self.session_timeout_hours = session_timeout_hours
        self.max_sessions_per_user = max_sessions_per_user
        
        # Session storage: {session_id: session_data}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        
        # User session tracking: {user_id: [session_ids]}
        self._user_sessions: Dict[str, list] = {}
        
        # Statistics
        self._stats = {
            'sessions_created': 0,
            'sessions_validated': 0,
            'sessions_invalidated': 0,
            'validation_failures': 0
        }
    
    def _generate_secure_token(self) -> str:
        """Generate a cryptographically secure random token"""
        return secrets.token_urlsafe(self.token_length)
    
    def _compute_session_hmac(self, user_id: str, shard_id: str, token: str) -> str:
        """
        Compute HMAC for session validation
        
        Args:
            user_id: User identifier
            shard_id: Shard identifier
            token: Random session token
            
        Returns:
            HMAC hex digest
        """
        message = f"{user_id}:{shard_id}:{token}".encode('utf-8')
        return hmac.new(
            self.secret_key.encode('utf-8'),
            message,
            hashlib.sha256
        ).hexdigest()
    
    def create_secure_session(
        self,
        user_id: str,
        shard_id: str,
        metadata: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Create a new secure session
        
        Args:
            user_id: User identifier
            shard_id: Shard identifier
            metadata: Optional session metadata
            
        Returns:
            Tuple of (session_id, session_token)
        """
        # Generate secure random token
        token = self._generate_secure_token()
        
        # Compute HMAC for validation
        session_hmac = self._compute_session_hmac(user_id, shard_id, token)
        
        # Create session ID with embedded metadata for quick validation
        # Format: {user_id}_{shard_id}_{token_prefix}_{hmac_prefix}
        session_id = f"{user_id}_{shard_id}_{token[:8]}_{session_hmac[:8]}"
        
        # Full session token for client
        session_token = f"{session_id}:{token}"
        
        # Store session data
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'shard_id': shard_id,
            'token': token,
            'hmac': session_hmac,
            'created_at': time.time(),
            'last_activity': time.time(),
            'metadata': metadata or {},
            'valid': True
        }
        
        self._sessions[session_id] = session_data
        
        # Track user sessions
        if user_id not in self._user_sessions:
            self._user_sessions[user_id] = []
        
        self._user_sessions[user_id].append(session_id)
        
        # Enforce max sessions per user
        if len(self._user_sessions[user_id]) > self.max_sessions_per_user:
            # Invalidate oldest session
            oldest_session_id = self._user_sessions[user_id][0]
            self.invalidate_session(oldest_session_id)
        
        self._stats['sessions_created'] += 1
        
        logger.info(
            f"Secure session created: User={user_id}, Shard={shard_id}, "
            f"SessionID={session_id}"
        )
        
        return session_id, session_token
    
    def validate_session(
        self,
        session_token: str,
        user_id: str,
        shard_id: str
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate a session token
        
        Args:
            session_token: Full session token from client
            user_id: Expected user ID
            shard_id: Expected shard ID
            
        Returns:
            Tuple of (is_valid, error_message, session_data)
        """
        self._stats['sessions_validated'] += 1
        
        try:
            # Parse session token
            if ':' not in session_token:
                self._stats['validation_failures'] += 1
                return False, "Invalid session token format", None
            
            session_id, token = session_token.rsplit(':', 1)
            
            # Check if session exists
            if session_id not in self._sessions:
                self._stats['validation_failures'] += 1
                return False, "Session not found", None
            
            session_data = self._sessions[session_id]
            
            # Check if session is valid
            if not session_data.get('valid', False):
                self._stats['validation_failures'] += 1
                return False, "Session has been invalidated", None
            
            # Check timeout
            current_time = time.time()
            timeout_seconds = self.session_timeout_hours * 3600
            if current_time - session_data['created_at'] > timeout_seconds:
                self._stats['validation_failures'] += 1
                self.invalidate_session(session_id)
                return False, "Session has expired", None
            
            # Validate user and shard match
            if session_data['user_id'] != user_id:
                self._stats['validation_failures'] += 1
                logger.warning(
                    f"Session user mismatch: Expected {user_id}, "
                    f"got {session_data['user_id']}"
                )
                return False, "Session does not belong to this user", None
            
            if session_data['shard_id'] != shard_id:
                self._stats['validation_failures'] += 1
                logger.warning(
                    f"Session shard mismatch: Expected {shard_id}, "
                    f"got {session_data['shard_id']}"
                )
                return False, "Session does not belong to this shard", None
            
            # Validate token matches
            if session_data['token'] != token:
                self._stats['validation_failures'] += 1
                return False, "Invalid session token", None
            
            # Validate HMAC
            expected_hmac = self._compute_session_hmac(user_id, shard_id, token)
            if not hmac.compare_digest(session_data['hmac'], expected_hmac):
                self._stats['validation_failures'] += 1
                logger.error(f"HMAC validation failed for session {session_id}")
                return False, "Session validation failed", None
            
            # Update last activity
            session_data['last_activity'] = current_time
            
            return True, None, session_data
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            self._stats['validation_failures'] += 1
            return False, f"Validation error: {str(e)}", None
    
    def validate_session_ownership(
        self,
        session_id: str,
        user_id: str,
        shard_id: str
    ) -> bool:
        """
        Validate that a session ID belongs to the specified user and shard
        
        Args:
            session_id: Session identifier
            user_id: Expected user ID
            shard_id: Expected shard ID
            
        Returns:
            True if session belongs to user/shard, False otherwise
        """
        # Quick validation from session ID format
        expected_prefix = f"{user_id}_{shard_id}_"
        if not session_id.startswith(expected_prefix):
            return False
        
        # Full validation from stored data
        if session_id in self._sessions:
            session_data = self._sessions[session_id]
            return (
                session_data['user_id'] == user_id and
                session_data['shard_id'] == shard_id and
                session_data.get('valid', False)
            )
        
        return False
    
    def invalidate_session(self, session_id: str):
        """Invalidate a session"""
        if session_id in self._sessions:
            self._sessions[session_id]['valid'] = False
            self._stats['sessions_invalidated'] += 1
            
            # Remove from user sessions list
            user_id = self._sessions[session_id]['user_id']
            if user_id in self._user_sessions:
                self._user_sessions[user_id] = [
                    sid for sid in self._user_sessions[user_id]
                    if sid != session_id
                ]
            
            logger.info(f"Session invalidated: {session_id}")
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user"""
        count = 0
        if user_id in self._user_sessions:
            for session_id in self._user_sessions[user_id]:
                if session_id in self._sessions:
                    self._sessions[session_id]['valid'] = False
                    count += 1
            
            del self._user_sessions[user_id]
            self._stats['sessions_invalidated'] += count
            logger.info(f"Invalidated {count} sessions for user {user_id}")
        
        return count
    
    def invalidate_shard_sessions(self, shard_id: str) -> int:
        """Invalidate all sessions for a shard"""
        count = 0
        sessions_to_invalidate = []
        
        for session_id, session_data in self._sessions.items():
            if session_data['shard_id'] == shard_id:
                sessions_to_invalidate.append(session_id)
        
        for session_id in sessions_to_invalidate:
            self.invalidate_session(session_id)
            count += 1
        
        logger.info(f"Invalidated {count} sessions for shard {shard_id}")
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        current_time = time.time()
        timeout_seconds = self.session_timeout_hours * 3600
        expired_sessions = []
        
        for session_id, session_data in self._sessions.items():
            if current_time - session_data['created_at'] > timeout_seconds:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
            del self._sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_user_sessions(self, user_id: str) -> list:
        """Get all valid sessions for a user"""
        if user_id not in self._user_sessions:
            return []
        
        valid_sessions = []
        for session_id in self._user_sessions[user_id]:
            if session_id in self._sessions and self._sessions[session_id].get('valid', False):
                valid_sessions.append(self._sessions[session_id])
        
        return valid_sessions
    
    def get_stats(self) -> Dict:
        """Get session manager statistics"""
        return {
            **self._stats,
            'active_sessions': len([s for s in self._sessions.values() if s.get('valid', False)]),
            'total_sessions': len(self._sessions),
            'users_with_sessions': len(self._user_sessions)
        }


# Global instance
secure_session_manager = SecureSessionManager()


# Convenience functions
def create_session(user_id: str, shard_id: str, metadata: Optional[Dict] = None) -> Tuple[str, str]:
    """Create a new secure session"""
    return secure_session_manager.create_secure_session(user_id, shard_id, metadata)


def validate_session(session_token: str, user_id: str, shard_id: str) -> Tuple[bool, Optional[str]]:
    """Validate a session token"""
    is_valid, error_msg, _ = secure_session_manager.validate_session(session_token, user_id, shard_id)
    return is_valid, error_msg


def validate_ownership(session_id: str, user_id: str, shard_id: str) -> bool:
    """Check if session belongs to user/shard"""
    return secure_session_manager.validate_session_ownership(session_id, user_id, shard_id)