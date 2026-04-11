"""Response healing auxiliary model.

Provides self-healing for malformed LLM outputs using configurable backends.
Part of the auxiliary model infrastructure — can use any OpenAI-compatible endpoint.

Supported healing tasks:
- JSON repair: Fixes broken tool call arguments (truncated, unescaped, malformed)
- Diff repair: Fixes failing search-replace edits (whitespace, indentation mismatches)

Config (config.yaml):
  auxiliary:
    response_healing:
      provider: "auto"        # auto | openrouter | nous | synthetic | custom
      model: ""               # e.g., "hf:syntheticlab/diff-apply"
      base_url: ""            # direct endpoint for local/custom healing
      api_key: ""             # paired with base_url
      timeout: 15             # healing should be fast
      enabled: true           # toggle healing on/off

Environment overrides:
  AUXILIARY_RESPONSE_HEALING_PROVIDER
  AUXILIARY_RESPONSE_HEALING_MODEL
  AUXILIARY_RESPONSE_HEALING_BASE_URL
  AUXILIARY_RESPONSE_HEALING_API_KEY
  HERMES_HEALING_ENABLED=false  # global toggle
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Global toggle (default: enabled)
HERMES_HEALING_ENABLED = os.environ.get("HERMES_HEALING_ENABLED", "true").lower() not in (
    "false", "0", "no"
)

# Prompts for healing models
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
    model_used: Optional[str] = None


def _call_healing_model(prompt: str) -> Optional[Dict]:
    """Call healing model via auxiliary_client.
    
    Returns:
        Parsed JSON response or None on failure
    """
    if not HERMES_HEALING_ENABLED:
        logger.debug("Response healing disabled via HERMES_HEALING_ENABLED")
        return None
    
    try:
        # Lazy import to avoid circular dependency
        from agent.auxiliary_client import call_llm
        
        response = call_llm(
            task="response_healing",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
            extra_body={"response_format": {"type": "json_object"}},
        )
        
        if response is None:
            logger.debug("Healing model returned None (no provider available)")
            return None
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        logger.debug(f"Healing model call failed: {e}")
        return None


def heal_json(broken_args: str) -> HealingResult:
    """Heal malformed JSON tool call arguments.

    Args:
        broken_args: The malformed JSON string from tool call arguments

    Returns:
        HealingResult with fixed JSON string if successful
    """
    if not broken_args or not broken_args.strip():
        return HealingResult(success=False, error="Empty input")

    # Fast path: already valid JSON
    try:
        json.loads(broken_args)
        return HealingResult(success=True, fixed_data=broken_args)
    except json.JSONDecodeError:
        pass  # Need healing

    # Call healing model
    prompt = JSON_HEALING_PROMPT.format(broken_json=broken_args)
    response = _call_healing_model(prompt)

    if not response:
        return HealingResult(success=False, error="No healing provider available")

    # Check if healing succeeded
    if not response.get("success"):
        return HealingResult(success=False, error="Healing model returned failure")

    fixed = response.get("fixed")
    if fixed is None:
        return HealingResult(success=False, error="No fixed data in response")

    # Return as JSON string
    try:
        if isinstance(fixed, str):
            json.loads(fixed)  # Validate
            return HealingResult(success=True, fixed_data=fixed)
        else:
            return HealingResult(success=True, fixed_data=json.dumps(fixed))
    except (json.JSONDecodeError, TypeError) as e:
        return HealingResult(success=False, error=f"Fixed data not valid JSON: {e}")


def heal_diff(edit_info: Union[Dict, tuple]) -> HealingResult:
    """Heal a failing search-replace diff edit.

    Args:
        edit_info: Dict with keys (file_path, search, replace, file_content)
                   or tuple (file_path, search, replace[, file_content])

    Returns:
        HealingResult with fixed edit dict if successful
    """
    # Normalize input
    if isinstance(edit_info, dict):
        file_path = edit_info.get("file_path", "")
        search = edit_info.get("search", "")
        replace = edit_info.get("replace", "")
        file_content = edit_info.get("file_content", "")
    else:
        if len(edit_info) < 3:
            return HealingResult(success=False, error="edit_info tuple must have at least 3 elements")
        file_path, search, replace = edit_info[:3]
        file_content = edit_info[3] if len(edit_info) > 3 else ""

    if not search:
        return HealingResult(success=False, error="Empty search string")

    edit_json = json.dumps(
        {
            "file": file_path,
            "edit": {"type": "diff", "search": search, "replace": replace},
        },
        indent=2,
    )

    prompt = DIFF_HEALING_PROMPT.format(
        file_content=file_content[:5000] if file_content else "(file content not provided)",
        edit_json=edit_json,
    )
    
    response = _call_healing_model(prompt)

    if not response:
        return HealingResult(success=False, error="No healing provider available")

    # Check for null response (ambiguous)
    if "edit" not in response and "search" not in response:
        return HealingResult(success=False, error="Edit is ambiguous")

    # Extract fixed edit from nested structure
    if "edit" in response:
        edit = response["edit"]
        fixed_search = edit.get("search")
        fixed_replace = edit.get("replace")
    else:
        fixed_search = response.get("search")
        fixed_replace = response.get("replace")

    if not fixed_search:
        return HealingResult(success=False, error="No fixed search in response")

    return HealingResult(
        success=True,
        fixed_data={"search": fixed_search, "replace": fixed_replace or replace},
    )


# Convenience functions for integration points
def try_heal_json(broken_args: str) -> Optional[str]:
    """Try to heal JSON, return fixed string or None.

    This is the main entry point for JSON healing in parsers.
    """
    if not HERMES_HEALING_ENABLED:
        return None
    
    result = heal_json(broken_args)
    if result.success:
        logger.info("Healed malformed JSON")
        return result.fixed_data
    return None


def try_heal_diff(edit_info: Union[Dict, tuple]) -> Optional[Dict]:
    """Try to heal diff edit, return fixed dict or None.

    This is the main entry point for diff healing in patch_parser.
    """
    if not HERMES_HEALING_ENABLED:
        return None
    
    result = heal_diff(edit_info)
    if result.success:
        logger.info("Healed malformed diff edit")
        return result.fixed_data
    return None
