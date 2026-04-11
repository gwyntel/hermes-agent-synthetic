#!/usr/bin/env python3
"""Tests for response_healing auxiliary model integration.

Validates JSON and diff healing with intentionally malformed inputs.
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.response_healing import (
    heal_json,
    heal_diff,
    try_heal_json,
    try_heal_diff,
    HealingResult,
    HERMES_HEALING_ENABLED,
)


class TestJSONHealing:
    """Test JSON healing with various malformed inputs."""

    def test_valid_json_passthrough(self):
        """Valid JSON should pass through unchanged."""
        valid_json = '{"name": "test", "arguments": {"x": 1}}'
        result = heal_json(valid_json)
        assert result.success
        parsed = json.loads(result.fixed_data)
        assert parsed["name"] == "test"

    def test_empty_string_returns_failure(self):
        """Empty string should return failure."""
        result = heal_json("")
        assert not result.success
        assert "Empty input" in result.error

    def test_whitespace_only_returns_failure(self):
        """Whitespace only should return failure."""
        result = heal_json("   ")
        assert not result.success

    @patch('agent.response_healing._call_healing_model')
    def test_healing_model_success(self, mock_call):
        """Test successful healing model response."""
        mock_call.return_value = {
            "success": True,
            "fixed": {"name": "healed", "arguments": {"x": 1}},
        }

        result = heal_json('{"broken": "json')
        assert result.success
        parsed = json.loads(result.fixed_data)
        assert parsed["name"] == "healed"

    @patch('agent.response_healing._call_healing_model')
    def test_healing_model_failure(self, mock_call):
        """Test healing model returning failure."""
        mock_call.return_value = {"success": False}

        result = heal_json('{"broken": "json')
        assert not result.success

    @patch('agent.response_healing._call_healing_model')
    def test_healing_model_unavailable(self, mock_call):
        """Test handling when healing model is unavailable."""
        mock_call.return_value = None

        result = heal_json('{"broken": "json')
        assert not result.success
        assert "No healing provider" in result.error


class TestDiffHealing:
    """Test diff healing with various malformed edits."""

    @patch('agent.response_healing._call_healing_model')
    def test_healing_success(self, mock_call):
        """Test successful diff healing."""
        mock_call.return_value = {
            "edit": {
                "search": "def foo():",
                "replace": "def bar():",
            }
        }

        edit_info = {
            "file_path": "test.py",
            "search": "def  foo():",  # Extra space
            "replace": "def bar():",
        }
        result = heal_diff(edit_info)
        assert result.success
        assert "def foo():" in result.fixed_data["search"]

    @patch('agent.response_healing._call_healing_model')
    def test_healing_tuple_input(self, mock_call):
        """Test tuple input format."""
        mock_call.return_value = {
            "search": "fixed search",
            "replace": "fixed replace",
        }

        result = heal_diff(("test.py", "broken search", "replace"))
        assert result.success
        assert result.fixed_data["search"] == "fixed search"

    def test_empty_search_returns_failure(self):
        """Empty search should return failure."""
        result = heal_diff({"file_path": "test.py", "search": "", "replace": "x"})
        assert not result.success

    def test_short_tuple_returns_failure(self):
        """Tuple with less than 3 elements should return failure."""
        result = heal_diff(("test.py", "search"))
        assert not result.success

    @patch('agent.response_healing._call_healing_model')
    def test_healing_ambiguous(self, mock_call):
        """Test handling when edit is ambiguous (null response)."""
        mock_call.return_value = {"type": None}

        result = heal_diff({"file_path": "test.py", "search": "ambiguous", "replace": "x"})
        assert not result.success
        assert "ambiguous" in result.error.lower()


class TestConvenienceFunctions:
    """Test try_heal_json and try_heal_diff convenience functions."""

    @patch('agent.response_healing.HERMES_HEALING_ENABLED', True)
    @patch('agent.response_healing.heal_json')
    def test_try_heal_json_success(self, mock_heal):
        """try_heal_json returns string on success."""
        mock_heal.return_value = HealingResult(success=True, fixed_data='{"fixed": true}')

        result = try_heal_json('{"broken"')
        assert result == '{"fixed": true}'

    @patch('agent.response_healing.HERMES_HEALING_ENABLED', True)
    @patch('agent.response_healing.heal_json')
    def test_try_heal_json_failure(self, mock_heal):
        """try_heal_json returns None on failure."""
        mock_heal.return_value = HealingResult(success=False, error="test error")

        result = try_heal_json('{"broken"')
        assert result is None

    @patch('agent.response_healing.HERMES_HEALING_ENABLED', False)
    def test_try_heal_json_disabled(self):
        """try_heal_json returns None when healing disabled."""
        result = try_heal_json('{"broken"')
        assert result is None

    @patch('agent.response_healing.HERMES_HEALING_ENABLED', True)
    @patch('agent.response_healing.heal_diff')
    def test_try_heal_diff_success(self, mock_heal):
        """try_heal_diff returns dict on success."""
        mock_heal.return_value = HealingResult(
            success=True,
            fixed_data={"search": "fixed", "replace": "fixed2"},
        )

        result = try_heal_diff({"file_path": "test.py", "search": "broken", "replace": "x"})
        assert result == {"search": "fixed", "replace": "fixed2"}


class TestIntegrationScenarios:
    """Real-world integration scenarios."""

    @patch('agent.response_healing._call_healing_model')
    def test_hermes_tool_call_truncated(self, mock_call):
        """Simulate truncated Hermes tool call."""
        mock_call.return_value = {
            "success": True,
            "fixed": {"name": "read_file", "arguments": {"path": "/etc/passwd"}},
        }

        # Simulate truncated tool call from LLM
        broken = '{"name": "read_file", "arguments": {"path": "/etc/passwd'
        result = heal_json(broken)

        assert result.success
        parsed = json.loads(result.fixed_data)
        assert parsed["name"] == "read_file"
        assert parsed["arguments"]["path"] == "/etc/passwd"

    @patch('agent.response_healing._call_healing_model')
    def test_diff_with_wrong_indentation(self, mock_call):
        """Simulate diff with wrong indentation."""
        mock_call.return_value = {
            "edit": {
                "search": "    def foo():\n        pass",
                "replace": "    def bar():\n        return 1",
            }
        }

        edit_info = {
            "file_path": "module.py",
            "search": "def foo():\n        pass",  # Missing leading spaces
            "replace": "def bar():\n        return 1",
            "file_content": "class A:\n    def foo():\n        pass\n",
        }

        result = heal_diff(edit_info)
        assert result.success
        assert "    def foo():" in result.fixed_data["search"]


def run_live_tests():
    """Run tests with live API (requires configured healing backend)."""
    print("\n=== Live API Tests ===\n")

    # Test 1: Truncated JSON
    print("Test 1: Truncated JSON tool call")
    broken = '{"name": "terminal", "arguments": {"command": "ls -la'
    result = try_heal_json(broken)
    if result:
        print(f"  ✓ Healed: {result[:100]}...")
    else:
        print("  ✗ Healing failed (API unavailable or healing failed)")

    # Test 2: Single quotes JSON
    print("\nTest 2: Single quotes JSON")
    broken = "{'name': 'read_file', 'arguments': {'path': '/tmp'}}"
    result = try_heal_json(broken)
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
        "file_content": "def foo():\n    pass\n",
    }
    result = try_heal_diff(edit_info)
    if result:
        print(f"  ✓ Healed search: {result['search']}")
        print(f"  ✓ Healed replace: {result['replace']}")
    else:
        print("  ✗ Healing failed")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test response healing")
    parser.add_argument("--live", action="store_true", help="Run live API tests")
    parser.add_argument("--pytest", action="store_true", help="Run pytest suite")
    args = parser.parse_args()

    if args.live:
        run_live_tests()
    elif args.pytest:
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
    else:
        # Default: run pytest
        import pytest

        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
