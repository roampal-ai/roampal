# C:\RoampalAI\backend\utils\common_utils.py

from pathlib import Path
import json
import re
import ast
from typing import Dict, Any

def load_prompt_template(path: Path, **kwargs) -> str:
    """Load and format a prompt template from a file, ensuring all variables are present."""
    with open(path, 'r', encoding='utf-8') as f:
        template = f.read()
    try:
        # Check for any required fields missing in kwargs
        import string
        formatter = string.Formatter()
        required_keys = {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}
        missing = required_keys - kwargs.keys()
        if missing:
            raise KeyError(
                f"Missing required template variables: {missing} when formatting {path}\n"
                f"Supplied: {list(kwargs.keys())}"
            )
        return template.format(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Template format error in {path}:\n{e}\nTemplate: {template}\nWith: {kwargs}")

def _repair_json(raw: str) -> str:
    """Repair and normalize a potentially malformed JSON string."""
    raw = re.sub(r'^.*?(?=\{|\[)', '', raw, flags=re.DOTALL)  # Remove before { or [
    raw = re.sub(r'(?<=}|\]).*', '', raw, flags=re.DOTALL)  # Remove after } or ]
    raw = re.sub(r'^```json\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r"(?<!\\)'", '"', raw)
    raw = re.sub(r'([^\s"{}\[\],:]+)\s*:\s*', r'"\1": ', raw)
    raw = re.sub(r':\s*([^\s"{}\[\],:]+?)(?=[,\s}\]])', r': "\1"', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    raw = raw.replace('"""', '"').replace('""', '"').replace('\\"', '"')
    open_brace, close_brace = raw.count('{'), raw.count('}')
    open_bracket, close_bracket = raw.count('['), raw.count(']')
    raw += '}' * (open_brace - close_brace)
    raw += ']' * (open_bracket - close_bracket)
    if raw.strip().startswith('{') and not raw.strip().endswith(']'):
        raw = '[' + raw + ']'
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        try:
            ast.literal_eval(raw)
            return raw
        except:
            return '[]'

def _read_json(path: Path) -> Dict[str, Any]:
    """Safely reads a JSON file, returning an empty dict if not found or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _write_json(path: Path, data: Dict[str, Any]):
    """Writes data to a JSON file with pretty printing."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
