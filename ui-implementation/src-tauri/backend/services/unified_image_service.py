"""
Unified Image Service for Roampal
Consolidates all image processing capabilities into a single, efficient service
"""

import base64
import io
import logging
from typing import Optional, Dict, Any, List
from PIL import Image
import pytesseract
from pathlib import Path

logger = logging.getLogger(__name__)

class UnifiedImageService:
    """Single image service handling all image-related operations"""

    def __init__(self, embedding_service=None, memory_adapter=None):
        self.embedding_service = embedding_service
        self.memory_adapter = memory_adapter
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    async def process_image(self,
                           image_data: str,
                           context: str = "",
                           extract_text: bool = True,
                           analyze_content: bool = True) -> Dict[str, Any]:
        """
        Process an image with multiple capabilities

        Args:
            image_data: Base64 encoded image or file path
            context: Additional context for the image
            extract_text: Whether to perform OCR
            analyze_content: Whether to analyze image content

        Returns:
            Dict containing extracted text, analysis, and metadata
        """
        try:
            # Decode image
            image = self._decode_image(image_data)

            result = {
                "success": True,
                "metadata": self._get_image_metadata(image),
                "context": context
            }

            # OCR text extraction
            if extract_text:
                result["extracted_text"] = self._extract_text(image)

            # Content analysis (placeholder for vision model integration)
            if analyze_content:
                result["content_analysis"] = await self._analyze_content(image, context)

            # Store in memory if available
            if self.memory_adapter and (result.get("extracted_text") or result.get("content_analysis")):
                await self._store_in_memory(result)

            return result

        except Exception as e:
            logger.error(f"Image processing error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image or load from file"""
        if image_data.startswith('data:image'):
            # Handle data URL
            base64_str = image_data.split(',')[1]
            image_bytes = base64.b64decode(base64_str)
            return Image.open(io.BytesIO(image_bytes))
        elif Path(image_data).exists():
            # Load from file
            return Image.open(image_data)
        else:
            # Assume base64
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))

    def _get_image_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract image metadata"""
        return {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        }

    def _extract_text(self, image: Image.Image) -> str:
        """Extract text using OCR"""
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    async def _analyze_content(self, image: Image.Image, context: str) -> Dict[str, Any]:
        """
        Analyze image content
        This is a placeholder for vision model integration (LLAVA, CLIP, etc)
        """
        # For now, return basic analysis
        return {
            "description": "Image analysis pending vision model integration",
            "context_provided": context,
            "colors": self._get_dominant_colors(image),
            "is_text_heavy": bool(self._extract_text(image))
        }

    def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        """Get dominant colors from image"""
        try:
            # Resize for faster processing
            small_image = image.copy()
            small_image.thumbnail((150, 150))

            # Convert to RGB if needed
            if small_image.mode != 'RGB':
                small_image = small_image.convert('RGB')

            # Get colors
            pixels = small_image.getdata()
            color_counts = {}

            for pixel in pixels:
                color_counts[pixel] = color_counts.get(pixel, 0) + 1

            # Get top colors
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            top_colors = [f"rgb{color}" for color, _ in sorted_colors[:num_colors]]

            return top_colors

        except Exception as e:
            logger.warning(f"Color extraction failed: {e}")
            return []

    async def _store_in_memory(self, result: Dict[str, Any]):
        """Store image analysis in memory system"""
        if not self.memory_adapter:
            return

        try:
            # Create fragment from image data
            fragment_text = f"Image Analysis:\n"

            if result.get("extracted_text"):
                fragment_text += f"Text: {result['extracted_text']}\n"

            if result.get("content_analysis"):
                fragment_text += f"Content: {result['content_analysis'].get('description', '')}\n"

            if result.get("context"):
                fragment_text += f"Context: {result['context']}\n"

            # Store with image-specific metadata
            metadata = {
                "type": "image",
                "has_text": bool(result.get("extracted_text")),
                "dimensions": f"{result['metadata']['width']}x{result['metadata']['height']}",
                "usefulness": 0.7,  # Images are generally useful
                "sentiment": 0.5    # Neutral sentiment by default
            }

            await self.memory_adapter.add_memory(
                text=fragment_text,
                metadata=metadata
            )

            logger.info("Image analysis stored in memory")

        except Exception as e:
            logger.error(f"Failed to store image in memory: {e}")

    async def search_image_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for image-related memories"""
        if not self.memory_adapter:
            return []

        try:
            # Search for image fragments
            results = await self.memory_adapter.search_memories(
                query_text=query,
                filter_dict={"type": "image"},
                top_k=limit
            )
            return results

        except Exception as e:
            logger.error(f"Image memory search failed: {e}")
            return []

    def cleanup(self):
        """Cleanup resources"""
        # Placeholder for any cleanup needed
        pass