#!/usr/bin/env bash
# bin/codex-worktree.sh — run Codex CLI inside a git worktree, then merge
#                         changes back to the current branch.
#
# This is the standard workflow for all Codex-driven coding in this repo.
# It prevents `git commit` from racing with Codex's own edits to the same
# files, and gives every Codex session its own branch for clean rollback.
#
# Usage:
#   bin/codex-worktree.sh <branch-name> <prompt-file-or-string> [...]
#
# Examples:
#   bin/codex-worktree.sh step5-codegen /tmp/codex_step5_prompt.md
#   bin/codex-worktree.sh fix-readme-typos "Fix typos in README.md"
#   bin/codex-worktree.sh step5-codegen /tmp/p.md --model gpt-5.4-mini --reasoning low
#
# What it does:
#   1. From the current branch (typically master), create a new branch in a
#      sibling worktree at ../<repo>-<branch>/.
#   2. Run `codex exec` inside that worktree with the provided prompt and any
#      extra flags forwarded to codex exec.
#   3. If Codex exits 0, show a diff summary, then offer to merge the branch
#      back into the original branch (with --no-ff so it's a real merge commit).
#   4. Remove the worktree and delete the temporary branch on merge success.
#
# Failure handling:
#   - If Codex exits non-zero, the worktree is left in place for inspection.
#   - If you say 'n' to the merge prompt, the worktree is also left in place
#     so you can review / rebase / squash manually.
#   - Run `git worktree remove --force <path>` to discard the worktree.
#
# Requirements:
#   - git >= 2.30 (for `git worktree remove`)
#   - codex CLI in PATH
#   - The current directory must be the root of a git repository.
#   - The working tree must be clean (no uncommitted changes).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage: bin/codex-worktree.sh <branch-name> <prompt> [extra-codex-flags...]

  <branch-name>     A new git branch name to create (e.g. step5-codegen)
  <prompt>          Either a path to a file containing the prompt, or a
                    string. If the path exists and is readable, its contents
                    are used; otherwise the literal string is passed.
  [extra flags]     Anything after the prompt is forwarded to `codex exec`.

Examples:
  bin/codex-worktree.sh step5-codegen /tmp/prompt.md
  bin/codex-worktree.sh fix-typos "Fix typos in README.md"
  bin/codex-worktree.sh step5-codegen /tmp/prompt.md --model gpt-5.4-mini
USAGE
  exit 64
fi

BRANCH_NAME="$1"
shift
PROMPT_ARG="$1"
shift

# Resolve the prompt: if it's a path to a readable file, use its contents.
if [[ -f "$PROMPT_ARG" ]]; then
  PROMPT="$(cat "$PROMPT_ARG")"
  PROMPT_SOURCE="file: $PROMPT_ARG"
else
  PROMPT="$PROMPT_ARG"
  PROMPT_SOURCE="string literal"
fi

# Sanity checks.
if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found in PATH" >&2
  exit 127
fi

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: not inside a git repository" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
REPO_NAME="$(basename "$REPO_ROOT")"
ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked files have uncommitted changes. Commit or stash before running." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

# Untracked files are allowed — they don't conflict with the worktree, which
# is a separate checkout. If a file with the same name exists in the worktree,
# that's the worktree's problem, not ours.

# Pick a worktree location. Sibling to the repo so it doesn't pollute the
# repo dir, and easy to spot in `git worktree list`.
WORKTREE_DIR="${REPO_ROOT}/../${REPO_NAME}.wt.${BRANCH_NAME}"

# Refuse to clobber an existing worktree.
if [[ -d "$WORKTREE_DIR" ]]; then
  echo "ERROR: worktree already exists at $WORKTREE_DIR" >&2
  echo "  Remove it with: git worktree remove --force $WORKTREE_DIR" >&2
  exit 1
fi

# Refuse to clobber an existing branch (other than the one we're about to create).
if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
  echo "ERROR: branch '$BRANCH_NAME' already exists" >&2
  echo "  Delete it with: git branch -D $BRANCH_NAME" >&2
  exit 1
fi

echo "==> Creating worktree"
echo "    repo:      $REPO_ROOT"
echo "    original:  $ORIGINAL_BRANCH"
echo "    new:       $BRANCH_NAME"
echo "    location:  $WORKTREE_DIR"
echo "    prompt:    $PROMPT_SOURCE ($(wc -l < <(echo "$PROMPT")) lines)"

# Add the worktree on a new branch based on the current HEAD.
git worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" "$ORIGINAL_BRANCH" >/dev/null

# Run codex inside the worktree. We forward all extra args to `codex exec`.
echo
echo "==> Running codex exec in worktree"
set +e
(
  cd "$WORKTREE_DIR"
  codex exec -C "$WORKTREE_DIR" "$@" "$PROMPT"
)
CODEX_RC=$?
set -e

echo
if [[ $CODEX_RC -ne 0 ]]; then
  echo "==> Codex exited with code $CODEX_RC"
  echo "    Worktree left in place at: $WORKTREE_DIR"
  echo "    Inspect with:  cd $WORKTREE_DIR && git status"
  echo "    Discard with: git worktree remove --force $WORKTREE_DIR && git branch -D $BRANCH_NAME"
  exit $CODEX_RC
fi

# Show what changed so the human can sanity-check before merge.
echo "==> Diff summary vs $ORIGINAL_BRANCH"
(
  cd "$WORKTREE_DIR"
  git fetch .. "$ORIGINAL_BRANCH:$ORIGINAL_BRANCH" 2>/dev/null || true
  git diff --stat "$ORIGINAL_BRANCH"..HEAD
  echo
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Uncommitted changes still in worktree:"
    git status --short --untracked-files=no
  else
    echo "(no uncommitted changes in worktree; everything was committed)"
  fi
)

# Prompt for merge. Default = no, so the human stays in the loop.
echo
read -r -p "==> Merge '$BRANCH_NAME' into '$ORIGINAL_BRANCH'? [y/N] " MERGE_ANSWER
MERGE_ANSWER="${MERGE_ANSWER:-N}"

if [[ ! "$MERGE_ANSWER" =~ ^[Yy]$ ]]; then
  echo "    Skipping merge. Worktree left in place at: $WORKTREE_DIR"
  echo "    To merge later: cd $REPO_ROOT && git merge --no-ff $BRANCH_NAME"
  exit 0
fi

# Merge with --no-ff so this is a real merge commit (visible in history).
echo "==> Merging $BRANCH_NAME into $ORIGINAL_BRANCH (--no-ff)"
(
  cd "$REPO_ROOT"
  git merge --no-ff "$BRANCH_NAME" -m "Merge branch '$BRANCH_NAME' (codex)"
)

# Clean up: remove worktree, delete the temporary branch.
echo "==> Cleaning up worktree and branch"
git worktree remove "$WORKTREE_DIR"
git branch -d "$BRANCH_NAME"

echo
echo "==> Done. Branch $BRANCH_NAME merged into $ORIGINAL_BRANCH and removed."
