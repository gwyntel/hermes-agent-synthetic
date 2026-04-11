"""Synthetic.new healing models integration.

Provides self-healing for malformed JSON tool calls and failed diff edits
using Synthetic's free LoRA models:
- hf:syntheticlab/fix-json: Fixes broken JSON tool call arguments
- hf:syntheticlab/diff-apply: Fixes search-replace diffs with spacing issues

These models are FREE on Synthetic subscription and much faster/cheaper
than retrying the main model.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)

# Synthetic API configuration
SYNTHETIC_BASE_URL = "https://api.synthetic.new/openai/v1"

# Model IDs for healing
# Note: fix-json model returns 503, but diff-apply works for both JSON and diffs
HEALING_MODEL_JSON = "hf:syntheticlab/diff-apply"
HEALING_MODEL_DIFF = "hf:syntheticlab/diff-apply"

# Environment variable toggle (default: enabled)
HERMES_HEALING_ENABLED = os.environ.get("HERMES_HEALING_ENABLED", "true").lower() not in ("false", "0", "no")

# Prompt for JSON healing (from syntheticlab/fix-json model card)
# Note: Double curly braces {{ }} escape to single { in format strings
JSON_HEALING_PROMPT = """The following string may be broken JSON. Fix it if possible. Respond with JSON in the following
format, defined as TypeScript types:

// Success response:
type JsonFixSuccess = {{
  success: true,
  // The parsed JSON
  fixed: any,
}};

// Failure response:
type JsonFixFailure = {{
  success: false,
}};

If it's more-or-less JSON, fix it and respond with the success response. If it's not, respond with
the failure response. Here's the string:

{broken_json}"""

# Prompt for diff healing (from syntheticlab/diff-apply model card)
DIFF_HEALING_PROMPT = """This edit is invalid; please fix it. The search string does not match perfectly with the file contents.
Respond only with JSON, and only with the edit JSON, not the original file.
If the edit is ambiguous, respond with null.

File content:
{file_content}

Edit to fix:
{edit_json}"""


@dataclass
class HealingResult:
    """Result from a healing attempt."""
    success: bool
    fixed_data: Optional[Any] = None
    error: Optional[str] = None
    healing_model: Optional[str] = None


def get_synthetic_api_key() -> Optional[str]:
    """Get Synthetic API key from environment."""
    # Try env var first
    key = os.environ.get("SYNTHETIC_API_KEY")
    if key:
        return key
    
    # Try .env file
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("SYNTHETIC_API_KEY="):
                    return line.strip().split("=", 1)[1].strip('"').strip("'")
    
    return None


def call_healing_model(prompt: str, model: str = HEALING_MODEL_JSON) -> Optional[Dict]:
    """Call a Synthetic healing model.
    
    Args:
        prompt: The prompt to send
        model: Model ID (default: fix-json)
        
    Returns:
        Parsed JSON response or None on failure
    """
    if not HERMES_HEALING_ENABLED:
        logger.debug("Synthetic healing: disabled via HERMES_HEALING_ENABLED")
        return None
    
    api_key = get_synthetic_api_key()
    if not api_key:
        logger.debug("Synthetic healing: no API key configured")
        return None
    
    try:
        resp = requests.post(
            f"{SYNTHETIC_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 2000,
            },
            timeout=30
        )
        
        if resp.status_code != 200:
            logger.debug(f"Synthetic healing: API error {resp.status_code}")
            return None
        
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
        
    except Exception as e:
        logger.debug(f"Synthetic healing: {e}")
        return None


def heal_json_tool_arguments(broken_args: str) -> Optional[str]:
    """Attempt to heal malformed JSON tool call arguments.
    
    This is called BEFORE retrying the main model, saving:
    - Time (healing model is fast)
    - Cost (healing model is free on Synthetic)
    - Model confusion (self-correction can confuse the model)
    
    Args:
        broken_args: The malformed JSON string from tool call arguments
        
    Returns:
        Fixed JSON string if successful, None otherwise
    """
    if not broken_args or not broken_args.strip():
        return None
    
    # Try standard JSON parsing first (fast path)
    try:
        json.loads(broken_args)
        return broken_args  # Already valid
    except json.JSONDecodeError:
        pass  # Need to heal
    
    # Call healing model
    prompt = JSON_HEALING_PROMPT.format(broken_json=broken_args)
    response = call_healing_model(prompt, model=HEALING_MODEL_JSON)
    
    if not response:
        return None
    
    # Check if healing succeeded
    if not response.get("success"):
        return None
    
    fixed = response.get("fixed")
    if fixed is None:
        return None
    
    # Return as JSON string
    try:
        if isinstance(fixed, str):
            # Validate it's valid JSON
            json.loads(fixed)
            return fixed
        else:
            return json.dumps(fixed)
    except (json.JSONDecodeError, TypeError):
        return None


def heal_diff_edit(edit_info: Union[Dict, Tuple[str, str, str]]) -> Optional[Dict]:
    """Attempt to heal a failing search-replace diff edit.
    
    When fuzzy_match.py fails to find a match, this can often fix
    spacing/indentation issues automatically.
    
    Args:
        edit_info: Either a dict with keys (file_path, search, replace, context_hint, file_content)
                   or a tuple of (file_path, search, replace)
        
    Returns:
        Dict with 'search' and 'replace' keys if successful, None otherwise
    """
    # Handle both dict and tuple inputs
    if isinstance(edit_info, dict):
        file_path = edit_info.get("file_path", "")
        search = edit_info.get("search", "")
        replace = edit_info.get("replace", "")
        file_content = edit_info.get("file_content", "")
    else:
        file_path, search, replace = edit_info[:3]
        file_content = edit_info[3] if len(edit_info) > 3 else ""
    
    if not search:
        return None
    
    edit_json = json.dumps({
        "file": file_path,
        "edit": {
            "type": "diff",
            "search": search,
            "replace": replace
        }
    }, indent=2)
    
    prompt = DIFF_HEALING_PROMPT.format(
        file_content=file_content[:5000] if file_content else "(file content not provided)",
        edit_json=edit_json
    )
    response = call_healing_model(prompt, model=HEALING_MODEL_DIFF)
    
    if not response:
        return None
    
    # Check for null response (ambiguous) - only fail if no edit data at all
    if "edit" not in response and "search" not in response:
        return None
    
    # Extract fixed edit from nested structure
    if "edit" in response:
        edit = response["edit"]
        fixed_search = edit.get("search")
        fixed_replace = edit.get("replace")
    else:
        fixed_search = response.get("search")
        fixed_replace = response.get("replace")
    
    if not fixed_search:
        return None
    
    return {
        "search": fixed_search,
        "replace": fixed_replace or replace
    }


# Full result versions for advanced usage
def heal_json_tool_arguments_full(broken_args: str) -> HealingResult:
    """Full healing result with metadata."""
    result = heal_json_tool_arguments(broken_args)
    if result:
        return HealingResult(success=True, fixed_data=json.loads(result), healing_model=HEALING_MODEL_JSON)
    return HealingResult(success=False, error="Healing failed")


def heal_diff_edit_full(file_path: str, search: str, replace: str) -> HealingResult:
    """Full healing result with metadata."""
    result = heal_diff_edit({"file_path": file_path, "search": search, "replace": replace})
    if result:
        return HealingResult(success=True, fixed_data=result, healing_model=HEALING_MODEL_DIFF)
    return HealingResult(success=False, error="Healing failed")


# Integration point for run_agent.py
def should_use_synthetic_healing(provider: str) -> bool:
    """Check if Synthetic healing should be used for a provider.
    
    Healing is free on Synthetic, so always enable for Synthetic provider.
    For other providers, it requires a Synthetic API key to be configured.
    """
    if not HERMES_HEALING_ENABLED:
        return False
    
    if provider == "synthetic":
        return True
    
    # For other providers, check if we have Synthetic API key configured
    # as a backup healing service
    return get_synthetic_api_key() is not None
