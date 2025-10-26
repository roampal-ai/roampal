import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError, Template
from typing import Dict, Optional, List

from core.interfaces.prompt_engine_interface import PromptEngineInterface, PromptContext
from config.settings import settings  # Import your settings

logger = logging.getLogger(__name__)

class PromptEngine(PromptEngineInterface):
    def __init__(self):
        self.jinja_env: Optional[Environment] = None
        self.template_dirs: List[Path] = []
        self.default_system_prompt_name: str = "default_system"
        logger.debug("PromptEngine instance created (uninitialized).")

    async def initialize(
        self,
        config: Optional[Dict] = None,
        template_directories: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the prompt engine with a prioritized list of template directories.
        Directories should be ordered from most specific (fragment/shard) to most general (shared/global).
        """
        # Support passing a single string, a list, or fallback to settings
        dirs = []
        if template_directories:
            # Allow str or list of str
            if isinstance(template_directories, str):
                dirs.append(Path(template_directories))
            else:
                dirs.extend([Path(d) for d in template_directories])
        else:
            # Fallback: [fragment_dir, shared_dir] or just [shared_dir]
            fragment_dir = getattr(settings.paths, "fragment_prompt_dir", None)
            shared_dir = getattr(settings.paths, "prompt_template_dir", None)
            if fragment_dir:
                dirs.append(Path(fragment_dir))
            if shared_dir:
                dirs.append(Path(shared_dir))
        
        # Remove duplicates and ensure directories exist
        seen = set()
        self.template_dirs = []
        for d in dirs:
            if d and str(d) not in seen:
                d.mkdir(parents=True, exist_ok=True)
                self.template_dirs.append(d)
                seen.add(str(d))
        if not self.template_dirs:
            raise RuntimeError("PromptEngine must have at least one valid template directory.")

        if config and "default_system_prompt_name" in config:
            self.default_system_prompt_name = config["default_system_prompt_name"]

        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader([str(d) for d in self.template_dirs]),
                autoescape=select_autoescape(['html', 'xml', 'jinja', 'txt']),
                enable_async=True
            )
            logger.info(f"PromptEngine initialized. Template directories: {self.template_dirs}, Default system prompt: '{self.default_system_prompt_name}.txt'")
        except Exception as e:
            logger.error(f"Failed to initialize Jinja2 Environment: {e}", exc_info=True)
            raise

    async def _load_template_jinja(self, template_name_with_ext: str) -> Template:
        if not self.jinja_env:
            logger.error("PromptEngine not initialized. Call initialize() first.")
            raise RuntimeError("PromptEngine not initialized.")
        
        # Log template search order
        logger.info(f"TEMPLATE DEBUG: Searching for '{template_name_with_ext}' in directories: {self.template_dirs}")
        
        # Check which directories actually contain the template
        found_in_dirs = []
        for template_dir in self.template_dirs:
            template_path = template_dir / template_name_with_ext
            if template_path.exists():
                found_in_dirs.append(str(template_path))
        
        logger.info(f"TEMPLATE DEBUG: Template '{template_name_with_ext}' found in: {found_in_dirs}")
        
        try:
            template = self.jinja_env.get_template(template_name_with_ext)
            # Log which template was actually loaded
            logger.info(f"TEMPLATE DEBUG: Jinja2 loaded '{template_name_with_ext}' - template object: {template}")
            return template
        except TemplateNotFound:
            logger.error(f"Jinja2 template file '{template_name_with_ext}' not found in any of {self.template_dirs}.")
            raise FileNotFoundError(f"Prompt template file '{template_name_with_ext}' not found.")
        except TemplateSyntaxError as e_syntax:
            logger.error(f"Syntax error in Jinja2 template '{template_name_with_ext}': {e_syntax.message} (line {e_syntax.lineno})", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error loading Jinja2 template '{template_name_with_ext}': {e}", exc_info=True)
            raise

    async def build_prompt(
        self,
        user_input: str,
        context: PromptContext,
        system_prompt_name: Optional[str] = None
    ) -> str:
        if not self.jinja_env:
            logger.error("PromptEngine not initialized for build_prompt. Call initialize() first.")
            raise RuntimeError("PromptEngine not initialized.")

        effective_system_prompt_name_no_ext = system_prompt_name if system_prompt_name else self.default_system_prompt_name
        effective_system_prompt_filename = f"{effective_system_prompt_name_no_ext}.txt"

        logger.info(
            f"PromptEngine.build_prompt: system_prompt_name_override='{system_prompt_name}', "
            f"default_system_prompt_name='{self.default_system_prompt_name}', "
            f"USING system_template_file='{effective_system_prompt_filename}' from dirs {self.template_dirs}"
        )

        try:
            system_template = await self._load_template_jinja(effective_system_prompt_filename)
            # Fix: PromptContext is a dictionary, not an object with __dict__
            if isinstance(context, dict):
                render_context_dict = context.copy()
            else:
                render_context_dict = context.__dict__.copy() if hasattr(context, '__dict__') else {}
            render_context_dict["user_input"] = user_input
            render_context_for_jinja = {k: v for k, v in render_context_dict.items() if v is not None}
            logger.debug(f"Context for Jinja rendering (keys): {list(render_context_for_jinja.keys())}")
            
            # Debug logging for system_prompt (reduced to debug level)
            if 'system_prompt' in render_context_for_jinja:
                system_prompt_content = str(render_context_for_jinja['system_prompt'])
                logger.debug(f"System prompt passed to template (length {len(system_prompt_content)})")
            else:
                logger.debug("No system_prompt in context - using default template")

            final_prompt_str = await system_template.render_async(**render_context_for_jinja)
            logger.debug(f"Template output generated (length: {len(final_prompt_str)})")
            logger.info(f"Final prompt generated via Jinja2 (length: {len(final_prompt_str)}) using system template '{effective_system_prompt_filename}'.")
            return final_prompt_str.strip()
        except FileNotFoundError:
            logger.error(f"System prompt template '{effective_system_prompt_filename}' not found in template dirs. Using hardcoded minimal prompt.")
            return f"System: You are a helpful assistant. Please assist the user with their query about: {user_input}"
        except Exception as e:
            logger.error(f"Error building prompt with Jinja2 template '{effective_system_prompt_filename}': {type(e).__name__} - {e}", exc_info=True)
            return f"Error building prompt. Please check system configuration.\nUser: {user_input}"

    async def add_template_directory(self, directory: str) -> None:
        """Add a template directory to the existing Jinja2 environment.
        New directories are added at the BEGINNING for highest priority."""
        if not self.jinja_env:
            logger.error("PromptEngine not initialized. Call initialize() first.")
            raise RuntimeError("PromptEngine not initialized.")
        
        new_dir = Path(directory)
        if str(new_dir) not in [str(d) for d in self.template_dirs]:
            new_dir.mkdir(parents=True, exist_ok=True)
            # PREPEND instead of append for highest priority
            self.template_dirs.insert(0, new_dir)
            
            # Recreate the Jinja2 environment with the new directory
            self.jinja_env = Environment(
                loader=FileSystemLoader([str(d) for d in self.template_dirs]),
                autoescape=select_autoescape(['html', 'xml', 'jinja', 'txt']),
                enable_async=True
            )
            logger.info(f"PRIORITY FIX: Added template directory at HIGHEST priority: {new_dir}. Directory order: {self.template_dirs}")
        else:
            logger.debug(f"Template directory {new_dir} already exists in environment.")

    async def close(self) -> None:
        pass
