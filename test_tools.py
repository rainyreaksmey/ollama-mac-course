"""Checks that run_tool rejects bad tool calls before any handler runs. Run: python test_tools.py"""
from tools import run_tool

calls = []


def needs_city(city: str) -> str:
    calls.append(city)
    return f"weather in {city}"


TEST_TOOLS = {"needs_city": needs_city}


def test_missing_argument():
    assert run_tool("needs_city", {}, TEST_TOOLS).startswith("error: bad arguments")


def test_unexpected_argument():
    assert run_tool("needs_city", {"city": "Phnom Penh", "units": "C"}, TEST_TOOLS).startswith("error: bad arguments")


def test_unknown_tool():
    assert run_tool("delete_files", {}, TEST_TOOLS) == "error: no tool named delete_files"


def test_valid_call_runs_once():
    assert run_tool("needs_city", {"city": "Phnom Penh"}, TEST_TOOLS) == "weather in Phnom Penh"


if __name__ == "__main__":
    test_missing_argument()
    test_unexpected_argument()
    test_unknown_tool()
    assert calls == [], "a handler ran on a bad call"
    test_valid_call_runs_once()
    assert calls == ["Phnom Penh"]
    print("all 4 checks passed")
