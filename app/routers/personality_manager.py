# app/routers/personality_manager.py
"""
Personality Management API
Handles custom personality templates for Roampal
"""
import logging
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/personality", tags=["personality"])

# Template directories (relative to project root, not backend/backend/)
TEMPLATES_DIR = Path(__file__).parent.parent.parent / "backend" / "templates" / "personality"
PRESETS_DIR = TEMPLATES_DIR / "presets"
CUSTOM_DIR = TEMPLATES_DIR / "custom"
ACTIVE_FILE = TEMPLATES_DIR / "active.txt"

# Ensure directories exist
PRESETS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)


class PersonalityTemplate(BaseModel):
    """Personality template data"""
    template_id: str
    name: str
    content: str
    is_preset: bool


class SaveTemplateRequest(BaseModel):
    """Request to save a custom template"""
    name: str
    content: str


class ActivateTemplateRequest(BaseModel):
    """Request to activate a template"""
    template_id: str


def _validate_template(content: str) -> Dict[str, Any]:
    """Validate template YAML structure"""
    try:
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            raise ValueError("Template must be a YAML dictionary")

        # Check for REQUIRED sections (strict validation)
        required_sections = ["identity", "communication"]
        missing_sections = [s for s in required_sections if s not in data]
        if missing_sections:
            raise ValueError(f"Template missing required sections: {', '.join(missing_sections)}")

        # Validate identity section has 'name' field
        if not data.get("identity", {}).get("name"):
            raise ValueError("Template must have 'identity.name' field")

        # Check for recommended sections (warnings only)
        recommended_sections = ["response_behavior", "memory_usage"]
        for section in recommended_sections:
            if section not in data:
                logger.warning(f"Template missing recommended section: {section}")

        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax: {e}")


def _get_template_name(template_data: Dict[str, Any]) -> str:
    """Extract name from template data"""
    identity = template_data.get("identity", {})
    role = identity.get("role", "Custom Assistant")
    return role


@router.get("/presets")
async def list_presets() -> Dict[str, List[str]]:
    """List all available preset templates"""
    try:
        presets = []
        custom = []

        # List preset templates
        if PRESETS_DIR.exists():
            for file in PRESETS_DIR.glob("*.txt"):
                presets.append(file.stem)

        # List custom templates
        if CUSTOM_DIR.exists():
            for file in CUSTOM_DIR.glob("*.txt"):
                custom.append(file.stem)

        return {
            "presets": presets,
            "custom": custom
        }
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_template() -> PersonalityTemplate:
    """Get the currently active template"""
    try:
        if not ACTIVE_FILE.exists():
            # No active template, use default
            default_file = PRESETS_DIR / "default.txt"
            if default_file.exists():
                content = default_file.read_text(encoding="utf-8")
                return PersonalityTemplate(
                    template_id="default",
                    name="Default",
                    content=content,
                    is_preset=True
                )
            else:
                raise HTTPException(status_code=404, detail="No active template found")

        content = ACTIVE_FILE.read_text(encoding="utf-8")

        # Parse to get name first (needed regardless)
        try:
            data = yaml.safe_load(content)
            name = _get_template_name(data)
        except Exception:
            name = "Custom"

        # Optimize: Use content hash for matching instead of full string comparison
        content_hash = hash(content)

        # Check if it matches a preset (using hash for faster comparison)
        template_id = "custom"
        is_preset = False

        for preset_file in PRESETS_DIR.glob("*.txt"):
            preset_content = preset_file.read_text(encoding="utf-8")
            if hash(preset_content) == content_hash and preset_content == content:
                template_id = preset_file.stem
                is_preset = True
                break

        return PersonalityTemplate(
            template_id=template_id,
            name=name,
            content=content,
            is_preset=is_preset
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get current template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/template/{template_id}")
async def get_template(template_id: str) -> PersonalityTemplate:
    """Get a specific template by ID"""
    try:
        # Check presets first
        preset_file = PRESETS_DIR / f"{template_id}.txt"
        if preset_file.exists():
            content = preset_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            name = _get_template_name(data)
            return PersonalityTemplate(
                template_id=template_id,
                name=name,
                content=content,
                is_preset=True
            )

        # Check custom templates
        custom_file = CUSTOM_DIR / f"{template_id}.txt"
        if custom_file.exists():
            content = custom_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            name = _get_template_name(data)
            return PersonalityTemplate(
                template_id=template_id,
                name=name,
                content=content,
                is_preset=False
            )

        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {template_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_custom_template(request: SaveTemplateRequest) -> Dict[str, str]:
    """Save a custom template"""
    try:
        # Validate template
        _validate_template(request.content)

        # Sanitize filename
        safe_name = "".join(c for c in request.name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()

        if not safe_name:
            safe_name = "custom"

        # Check if template already exists
        custom_file = CUSTOM_DIR / f"{safe_name}.txt"
        overwrite = custom_file.exists()

        # Save to custom directory
        custom_file.write_text(request.content, encoding="utf-8")

        logger.info(f"{'Overwrote' if overwrite else 'Saved'} custom template: {safe_name}")

        return {
            "template_id": safe_name,
            "message": "Template updated successfully" if overwrite else "Template saved successfully",
            "overwrite": str(overwrite)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activate")
async def activate_template(request: ActivateTemplateRequest) -> Dict[str, str]:
    """Activate a template (preset or custom)"""
    try:
        # Find the template file
        preset_file = PRESETS_DIR / f"{request.template_id}.txt"
        custom_file = CUSTOM_DIR / f"{request.template_id}.txt"

        source_file = None
        if preset_file.exists():
            source_file = preset_file
        elif custom_file.exists():
            source_file = custom_file
        else:
            raise HTTPException(status_code=404, detail=f"Template '{request.template_id}' not found")

        # Validate before activating
        content = source_file.read_text(encoding="utf-8")
        _validate_template(content)

        # Copy to active.txt
        shutil.copy2(source_file, ACTIVE_FILE)

        logger.info(f"Activated template: {request.template_id}")

        return {
            "template_id": request.template_id,
            "message": "Template activated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_template(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a custom template file"""
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode("utf-8")

        # Validate template
        _validate_template(content_str)

        # Sanitize filename
        original_name = Path(file.filename).stem
        safe_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()

        if not safe_name:
            safe_name = "uploaded"

        # Save to custom directory
        custom_file = CUSTOM_DIR / f"{safe_name}.txt"
        custom_file.write_text(content_str, encoding="utf-8")

        logger.info(f"Uploaded template: {safe_name}")

        return {
            "template_id": safe_name,
            "message": "Template uploaded successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to upload template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom/{template_id}")
async def delete_custom_template(template_id: str) -> Dict[str, str]:
    """Delete a custom template (presets cannot be deleted)"""
    try:
        custom_file = CUSTOM_DIR / f"{template_id}.txt"

        if not custom_file.exists():
            raise HTTPException(status_code=404, detail=f"Custom template '{template_id}' not found")

        # Don't allow deleting if it's currently active
        if ACTIVE_FILE.exists():
            active_content = ACTIVE_FILE.read_text(encoding="utf-8")
            custom_content = custom_file.read_text(encoding="utf-8")
            if active_content == custom_content:
                raise HTTPException(status_code=400, detail="Cannot delete currently active template")

        custom_file.unlink()

        logger.info(f"Deleted custom template: {template_id}")

        return {
            "template_id": template_id,
            "message": "Template deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template: {e}")
        raise HTTPException(status_code=500, detail=str(e))