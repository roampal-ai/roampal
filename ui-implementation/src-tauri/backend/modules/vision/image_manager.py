import httpx
import json
from typing import Dict, Any
import base64
from pathlib import Path
import os
import aiofiles

class ImageManager:
    """Wrapper for image processing that calls tools_service API"""
    
    def __init__(self):
        self.tools_service_url = os.getenv('TOOLS_SERVICE_URL', 'http://localhost:8002')
    
    async def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process image by calling tools_service API"""
        try:
            # Read image file asynchronously
            async with aiofiles.open(image_path, 'rb') as f:
                image_data = await f.read()
            
            # Prepare form data
            files = {'image_file': (Path(image_path).name, image_data, 'image/jpeg')}
            
            # Call tools_service image analysis endpoint asynchronously
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.tools_service_url}/tools/speech-vision/vision/analyze",
                    files=files
                )
            
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "analysis": result.get("data", {}),
                        "image_hash": result.get("image_hash", ""),
                        "metadata": result.get("metadata", {})
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Tools service returned {response.status_code}",
                        "analysis": {},
                        "image_hash": "",
                        "metadata": {}
                    }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {},
                "image_hash": "",
                "metadata": {}
            } 