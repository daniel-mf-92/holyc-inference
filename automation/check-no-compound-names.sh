#!/usr/bin/env bash
# check-no-compound-names.sh — reject identifier compounding
# Usage: check-no-compound-names.sh [<rev>]
# Exits 0 if no violations in the given revision's diff vs parent; 1 otherwise.

set -euo pipefail

REV="${1:-HEAD}"
MAX_LEN=40
MAX_TOKENS=5

violations=0

# Extract files added/modified in this commit
files=$(git diff-tree --no-commit-id --name-only -r "$REV" 2>/dev/null || git diff --name-only HEAD)

while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  [[ ! -f "$f" ]] && continue

  # Check filename itself (basename without extension)
  base=$(basename "$f")
  name="${base%.*}"
  len=${#name}
  tokens=$(echo "$name" | tr '_-' '\n' | wc -l | tr -d ' ')

  if (( len > MAX_LEN )); then
    echo "VIOLATION: filename too long ($len > $MAX_LEN): $f"
    violations=$((violations+1))
  fi
  if (( tokens > MAX_TOKENS )); then
    echo "VIOLATION: filename has too many tokens ($tokens > $MAX_TOKENS): $f"
    violations=$((violations+1))
  fi

  # For .HC and .sh files, check function/identifier names added in this diff
  case "$f" in
    *.HC|*.sh|*.py)
      # Extract added lines from current working diff against the target rev.
      # This enforces naming on changes being made now, not historical commits.
      added=$(git diff "$REV" -- "$f" 2>/dev/null | grep -E "^\+" | grep -v "^+++") || true

      # HolyC: Type FuncName(...)  ; bash: function_name() {  ; python: def func(
      idents=$(echo "$added" | grep -oE '(\b[A-Za-z_][A-Za-z0-9_]+)\s*\(' | sed 's/[[:space:]]*($//' | sort -u) || true

      while IFS= read -r ident; do
        [[ -z "$ident" ]] && continue
        ilen=${#ident}
        if (( ilen > MAX_LEN )); then
          echo "VIOLATION: identifier too long ($ilen > $MAX_LEN) in $f: $ident"
          violations=$((violations+1))
        fi
      done <<< "$idents"
      ;;
  esac
done <<< "$files"

if (( violations > 0 )); then
  echo ""
  echo "TOTAL VIOLATIONS: $violations"
  echo "Identifier-compounding ban: max ${MAX_LEN} chars, max ${MAX_TOKENS} tokens. Revise names."
  exit 1
fi

echo "check-no-compound-names: OK"
exit 0
