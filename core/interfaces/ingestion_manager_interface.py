# backend/core/interfaces/ingestion_manager_interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# To avoid circular dependency with concrete models, we can use 'Any' here
# or forward reference if models.py doesn't import from core.interfaces.
# For now, 'Any' is simplest for the interface definition.

class IngestionManagerInterface(ABC):
    @abstractmethod
    async def initialize(self): # Add settings, llm_client if directly needed by interface methods
        """Initializes the ingestion manager."""
        pass

    @abstractmethod
    async def submit_job_request(self, job_request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts an ingestion job request (as dict or Pydantic model), 
        stores it, and potentially triggers processing.
        Returns the created job metadata (as dict or Pydantic model).
        """
        pass

    @abstractmethod
    async def process_single_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single ingestion job.
        Takes job data, returns updated job data.
        """
        pass

    @abstractmethod
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the status of a specific ingestion job.
        """
        pass
