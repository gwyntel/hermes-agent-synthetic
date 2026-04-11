#!/usr/bin/env python3
"""Tests for Synthetic healing models integration.

Validates JSON and diff healing with intentionally malformed inputs.
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.synthetic_healing import (
    heal_json_tool_arguments,
    heal_diff_edit,
    should_use_synthetic_healing,
    HERMES_HEALING_ENABLED,
)


class TestJSONHealing:
    """Test JSON healing with various malformed inputs."""
    
    # Test cases: (name, broken_json, expected_valid)
    TEST_CASES = [
        ("truncated_object", '{"key": "value', True),
        ("missing_closing_brace", '{"key": "value"', True),
        ("unescaped_newline", '{"key": "val\\nue"}', True),  # This is actually valid
        ("single_quotes", "{'key': 'value'}", True),
        ("trailing_comma", '{"key": "value",}', True),
        ("missing_quotes", '{key: "value"}', True),
        ("empty_string", "", False),
        ("whitespace_only", "   ", False),
        ("already_valid", '{"name": "test", "args": {"x": 1}}', True),
    ]
    
    def test_valid_json_passthrough(self):
        """Valid JSON should pass through unchanged."""
        valid_json = '{"name": "test", "arguments": {"x": 1}}'
        result = heal_json_tool_arguments(valid_json)
        assert result is not None
        # Should be parseable
        parsed = json.loads(result)
        assert parsed["name"] == "test"
    
    def test_truncated_json(self):
        """Truncated JSON should be healable."""
        broken = '{"name": "test", "arguments": {"x":'
        result = heal_json_tool_arguments(broken)
        # With mock or real API, should attempt healing
        # Without API key, returns None
        if result is not None:
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
    
    def test_single_quotes_json(self):
        """JSON with single quotes should be healable."""
        broken = "{'name': 'test', 'arguments': {'x': 1}}"
        result = heal_json_tool_arguments(broken)
        # Should attempt healing
        if result is not None:
            parsed = json.loads(result)
            assert parsed["name"] == "test"
    
    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert heal_json_tool_arguments("") is None
        assert heal_json_tool_arguments("   ") is None
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_healing_model_success(self, mock_call):
        """Test successful healing model response."""
        mock_call.return_value = {
            "success": True,
            "fixed": {"name": "healed", "arguments": {"x": 1}}
        }
        
        result = heal_json_tool_arguments('{"broken": "json')
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == "healed"
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_healing_model_failure(self, mock_call):
        """Test healing model returning failure."""
        mock_call.return_value = {"success": False}
        
        result = heal_json_tool_arguments('{"broken": "json')
        assert result is None
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_healing_model_unavailable(self, mock_call):
        """Test handling when healing model is unavailable."""
        mock_call.return_value = None
        
        result = heal_json_tool_arguments('{"broken": "json')
        assert result is None


class TestDiffHealing:
    """Test diff healing with various malformed edits."""
    
    def test_valid_edit_passthrough(self):
        """Valid edit info should be processed."""
        # Without API key, returns None (no healing)
        edit_info = {
            "file_path": "test.py",
            "search": "def foo():",
            "replace": "def bar():",
        }
        result = heal_diff_edit(edit_info)
        # Without API key, returns None
        if result is not None:
            assert "search" in result
            assert "replace" in result
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_healing_success(self, mock_call):
        """Test successful diff healing."""
        mock_call.return_value = {
            "edit": {
                "search": "def foo():",
                "replace": "def bar():"
            }
        }
        
        edit_info = {
            "file_path": "test.py",
            "search": "def  foo():",  # Extra space
            "replace": "def bar():",
        }
        result = heal_diff_edit(edit_info)
        assert result is not None
        assert result["search"] == "def foo():"
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_healing_tuple_input(self, mock_call):
        """Test tuple input format."""
        mock_call.return_value = {
            "search": "fixed search",
            "replace": "fixed replace"
        }
        
        result = heal_diff_edit(("test.py", "broken search", "replace"))
        assert result is not None
    
    def test_empty_search_returns_none(self):
        """Empty search should return None."""
        result = heal_diff_edit({"file_path": "test.py", "search": "", "replace": "x"})
        assert result is None


class TestHealingToggle:
    """Test HERMES_HEALING_ENABLED environment variable."""
    
    def test_should_use_synthetic_provider(self):
        """Synthetic provider should use healing if enabled."""
        # This depends on env var
        result = should_use_synthetic_healing("synthetic")
        assert result == HERMES_HEALING_ENABLED
    
    @patch('agent.synthetic_healing.get_synthetic_api_key')
    def test_other_provider_with_key(self, mock_key):
        """Other providers should use healing if API key is set."""
        mock_key.return_value = "test-key"
        result = should_use_synthetic_healing("openai")
        assert result == HERMES_HEALING_ENABLED
    
    @patch('agent.synthetic_healing.get_synthetic_api_key')
    def test_other_provider_without_key(self, mock_key):
        """Other providers should not use healing without API key."""
        mock_key.return_value = None
        result = should_use_synthetic_healing("openai")
        assert result is False


class TestIntegrationScenarios:
    """Real-world integration scenarios."""
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_hermes_tool_call_truncated(self, mock_call):
        """Simulate truncated Hermes tool call."""
        mock_call.return_value = {
            "success": True,
            "fixed": {"name": "read_file", "arguments": {"path": "/etc/passwd"}}
        }
        
        # Simulate truncated tool call from LLM
        broken = '{"name": "read_file", "arguments": {"path": "/etc/passwd'
        result = heal_json_tool_arguments(broken)
        
        assert result is not None
        parsed = json.loads(result)
        assert parsed["name"] == "read_file"
        assert parsed["arguments"]["path"] == "/etc/passwd"
    
    @patch('agent.synthetic_healing.call_healing_model')
    def test_diff_with_wrong_indentation(self, mock_call):
        """Simulate diff with wrong indentation."""
        mock_call.return_value = {
            "edit": {
                "search": "    def foo():\n        pass",
                "replace": "    def bar():\n        return 1"
            }
        }
        
        edit_info = {
            "file_path": "module.py",
            "search": "def foo():\n        pass",  # Missing leading spaces
            "replace": "def bar():\n        return 1",
            "file_content": "class A:\n    def foo():\n        pass\n"
        }
        
        result = heal_diff_edit(edit_info)
        assert result is not None
        assert "    def foo():" in result["search"]


def run_live_tests():
    """Run tests with live API (requires SYNTHETIC_API_KEY)."""
    print("\n=== Live API Tests ===\n")
    
    # Test 1: Truncated JSON
    print("Test 1: Truncated JSON tool call")
    broken = '{"name": "terminal", "arguments": {"command": "ls -la'
    result = heal_json_tool_arguments(broken)
    if result:
        print(f"  ✓ Healed: {result[:100]}...")
    else:
        print("  ✗ Healing failed (API unavailable or healing failed)")
    
    # Test 2: Single quotes JSON
    print("\nTest 2: Single quotes JSON")
    broken = "{'name': 'read_file', 'arguments': {'path': '/tmp'}}"
    result = heal_json_tool_arguments(broken)
    if result:
        print(f"  ✓ Healed: {result}")
    else:
        print("  ✗ Healing failed")
    
    # Test 3: Diff healing
    print("\nTest 3: Diff healing")
    edit_info = {
        "file_path": "test.py",
        "search": "def  foo():",  # Extra space
        "replace": "def bar():",
        "file_content": "def foo():\n    pass\n"
    }
    result = heal_diff_edit(edit_info)
    if result:
        print(f"  ✓ Healed search: {result['search']}")
        print(f"  ✓ Healed replace: {result['replace']}")
    else:
        print("  ✗ Healing failed")
    
    print("\n=== Done ===\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Synthetic healing")
    parser.add_argument("--live", action="store_true", help="Run live API tests")
    parser.add_argument("--pytest", action="store_true", help="Run pytest suite")
    args = parser.parse_args()
    
    if args.live:
        run_live_tests()
    elif args.pytest:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Default: run pytest
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
