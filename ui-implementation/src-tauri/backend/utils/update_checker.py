"""
Update Checker - v0.2.10
Checks for available updates from roampal.ai and notifies users.

Architecture:
- Fetches https://roampal.ai/updates/latest.json on app startup
- Compares version using semantic versioning
- Returns update info if newer version available
- Directs users to Gumroad for download (maintains sales funnel)
"""
import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

UPDATE_CHECK_URL = "https://roampal.ai/updates/latest.json"
CURRENT_VERSION = "0.2.12"


def parse_version(version_str: str) -> tuple:
    """Parse version string into tuple for comparison (e.g., '0.2.8' -> (0, 2, 8))"""
    try:
        parts = version_str.strip().split('.')
        return tuple(int(p) for p in parts[:3])
    except:
        return (0, 0, 0)


async def check_for_updates() -> Optional[Dict[str, Any]]:
    """
    Check for updates on startup.

    Returns:
        Dict with update info if newer version available, None otherwise.
        {
            "version": "0.2.9",
            "notes": "Bug fixes...",
            "download_url": "https://...",
            "is_critical": False
        }
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(UPDATE_CHECK_URL)
            if response.status_code != 200:
                logger.debug(f"[UPDATE] Check returned status {response.status_code}")
                return None

            data = response.json()
            latest = data.get("version", "0.0.0")
            min_version = data.get("min_version", "0.0.0")

            latest_tuple = parse_version(latest)
            current_tuple = parse_version(CURRENT_VERSION)
            min_tuple = parse_version(min_version)

            if latest_tuple > current_tuple:
                logger.info(f"[UPDATE] New version available: {latest} (current: {CURRENT_VERSION})")
                return {
                    "version": latest,
                    "notes": data.get("notes", ""),
                    "download_url": data.get("download_url", "https://roampal.gumroad.com/l/roampal"),
                    "is_critical": current_tuple < min_tuple,
                    "pub_date": data.get("pub_date", "")
                }

            logger.debug(f"[UPDATE] Already on latest version ({CURRENT_VERSION})")
            return None

    except httpx.TimeoutException:
        logger.debug("[UPDATE] Check timed out (5s)")
        return None
    except Exception as e:
        # Fail silently - update check is non-critical
        logger.debug(f"[UPDATE] Check failed: {e}")
        return None


def get_current_version() -> str:
    """Return the current app version"""
    return CURRENT_VERSION
