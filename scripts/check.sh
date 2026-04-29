#!/bin/bash
# GEO-Digest Pre/Post Change Helper
# Usage: bash ~/.hermes/geo_digest/scripts/check.sh [file1.py file2.py ...]
# If no files specified — checks all core files

set -e
cd ~/.hermes/geo_digest
export PATH="$HOME/.local/bin:$PATH"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FILES="${@:-engine/schemas.py engine/agents/writer.py engine/agents/base.py engine/orchestrator_v2.py engine/agents/reviewer.py engine/agents/reader.py engine/llm/config.py}"

echo ""
echo "══════════════════════════════════════════════"
echo "  GEO-Digest Codebase Check"
echo "══════════════════════════════════════════════"
echo ""

# 1. LeanKG impact for changed files
echo -e "${YELLOW}[1/3] Impact Analysis (LeanKG)${NC}"
for f in $FILES; do
    if [ -f "$f" ]; then
        count=$(leankg impact "$f" 2>&1 | grep -c "^  -" || true)
        echo "  $f → $count dependents"
    fi
done
echo ""

# 2. Pyright type check
echo -e "${YELLOW}[2/3] Type Check (pyright)${NC}"
pyright $FILES 2>&1 | grep -E '^[0-9]+ error|No errors' | tail -1
echo ""

# 3. Unit tests
echo -e "${YELLOW}[3/3] Unit Tests (pytest)${NC}"
python -m pytest tests/ -q \
    -k 'not test_cli and not test_e2e_full_pipeline and not test_editor_api' \
    --tb=line 2>&1 | tail -5
echo ""

echo "══════════════════════════════════════════════"
echo "  Check complete"
echo "══════════════════════════════════════════════"
