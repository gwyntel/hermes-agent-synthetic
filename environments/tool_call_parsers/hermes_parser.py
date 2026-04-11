"""
Hermes tool call parser.

Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
Based on VLLM's Hermes2ProToolParser.extract_tool_calls()
"""

import json
import logging
import re
import uuid
from typing import List, Optional, Tuple

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
_heal_json = None

def _get_heal_json():
    """Lazy loader for heal_json_tool_arguments."""
    global _heal_json
    if _heal_json is None:
        try:
            from agent.synthetic_healing import heal_json_tool_arguments
            _heal_json = heal_json_tool_arguments
        except ImportError:
            _heal_json = lambda x: None  # No-op if healing unavailable
    return _heal_json



@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    """
    Parser for Hermes-format tool calls.

    Matches <tool_call>...</tool_call> tags containing JSON with "name" and "arguments".
    Also handles unclosed <tool_call> at end-of-string (truncated generation).
    """

    # Matches both closed and unclosed tool_call tags
    PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)", re.DOTALL
    )

    def parse(self, text: str) -> ParseResult:
        if "<tool_call>" not in text:
            return text, None

        try:
            matches = self.PATTERN.findall(text)
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            for match in matches:
                # match is a tuple: (closed_content, unclosed_content)
                raw_json = match[0] if match[0] else match[1]
                if not raw_json.strip():
                    continue

                try:
                    tc_data = json.loads(raw_json)
                except json.JSONDecodeError:
                    # Attempt healing via Synthetic fix-json model
                    heal_fn = _get_heal_json()
                    healed_json = heal_fn(raw_json)
                    
                    if healed_json:
                        logger.info(f"Healed malformed JSON tool call")
                        try:
                            tc_data = json.loads(healed_json)
                        except json.JSONDecodeError:
                            logger.warning(f"Healing produced invalid JSON, skipping")
                            continue
                    else:
                        logger.warning(f"JSON parse failed, healing unavailable")
                        continue
                
                if "name" not in tc_data:
                    continue
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=tc_data["name"],
                            arguments=json.dumps(
                                tc_data.get("arguments", {}), ensure_ascii=False
                            ),
                        ),
                    )
                )

            if not tool_calls:
                return text, None

            # Content is everything before the first <tool_call> tag
            content = text[: text.find("<tool_call>")].strip()
            return content if content else None, tool_calls

        except Exception:
            return text, None
