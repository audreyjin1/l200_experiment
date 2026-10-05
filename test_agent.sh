#!/usr/bin/env bash
# Healthcare Agent Test Runner
set -e

# Load .env variables if present
if [ -f .env ]; then
    set -a
    # Source .env ignoring comments and empty lines
    eval "$(grep -v '^#' .env | grep '=' | sed -E 's/^([^=]+)=(.*)$/export \1="\2"/')" 2>/dev/null || true
    set +a
fi

# Select working Python interpreter
if command -v /Library/GoogleCorpSupport/bin/python3 >/dev/null 2>&1; then
    PYTHON="/Library/GoogleCorpSupport/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "Error: Python 3 not found."
    exit 1
fi

echo "========================================================"
echo "🏥 Healthcare ADK 2.0 Agent - Test Menu"
echo "========================================================"
echo "Select how you would like to test:"
echo "  1) Interactive Chat (Type your own symptoms/prompts)"
echo "  2) Run 5-Scenario Automated Demo"
echo "  3) Run Full Unit Test Suite (19 tests)"
echo "  4) Exit"
echo "========================================================"

if [ -n "$1" ]; then
    CHOICE="$1"
else
    read -p "Enter choice [1-4]: " CHOICE
fi

case "$CHOICE" in
    1|chat|interactive)
        "$PYTHON" src/healthcare_agent/cli.py
        ;;
    2|demo)
        "$PYTHON" src/healthcare_agent/demo.py
        ;;
    3|test|unit)
        "$PYTHON" -m unittest discover tests/
        ;;
    4|exit|quit)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid selection: $CHOICE"
        exit 1
        ;;
esac
