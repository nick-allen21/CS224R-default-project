#!/bin/bash
# Pre-push cleanup script. Run from repo root.
# Implements the changes recommended by the security + hygiene reviewers.
#
# What this does:
#   1. Redacts the WANDB API key from the three launch shell scripts (they're
#      untracked — secret never entered git history, but rotate the key anyway).
#   2. Moves internal notes (PROJECT_SUMMARY, CONJECTURER_PLAN) to a gitignored
#      notes/ dir so they survive locally but aren't pushed.
#   3. Removes stray .zip files and tracked .pyc / .DS_Store files.
#   4. Moves user-launch scripts into joint_trainer/scripts/.
#   5. Updates .gitignore with missing patterns.
#
# What you still need to do MANUALLY after this script:
#   * Rotate the WANDB API key at https://wandb.ai/authorize
#   * Review the diff (`git status` + `git diff`) before committing
#   * Add a "Joint Trainer Extension" section to README.md
#   * Verify with: grep -rE "wandb_v1_|hf_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{20,}" . --include="*.py" --include="*.sh" --include="*.md" --exclude-dir=.git

set -euo pipefail

REPO_ROOT="/Users/finnstablein/code/CS224R-default-project"
cd "$REPO_ROOT"

echo "===> Step 1/6: redact WANDB key from shell scripts"
for script in launch_v7.sh eval_v7.sh eval_fair.sh; do
    if [ -f "$script" ]; then
        sed -i '' 's|export WANDB_API_KEY=wandb_v1_[A-Za-z0-9_]*|# Set WANDB_API_KEY in your shell before running: export WANDB_API_KEY=...|' "$script"
        echo "  redacted $script"
    fi
done

echo "===> Step 2/6: move internal notes to gitignored notes/"
mkdir -p notes
for f in poster_assets/PROJECT_SUMMARY.md poster_assets/PROJECT_SUMMARY.docx \
         poster_assets/PROJECT_SUMMARY.md.zip \
         joint_trainer/CONJECTURER_PLAN.md; do
    if [ -e "$f" ]; then
        mv "$f" "notes/$(basename "$f")"
        echo "  moved $f -> notes/"
    fi
done

echo "===> Step 3/6: delete stray .zip files at root and one-offs"
for f in default_proj_sft.zip rloo_trainer.zip; do
    if [ -f "$f" ]; then
        rm "$f"
        echo "  deleted $f"
    fi
done
# Don't touch milestone .tex.zip — leaves that decision to user
# Don't touch the .swp swap file (vim leftover)
if [ -f ".claude/skills/dcc1_rewrite_jira_ticket/.SKILL.md.swp" ]; then
    : # not in this repo
fi

echo "===> Step 4/6: untrack .pyc and .DS_Store files"
git rm --cached __pycache__/modal_train.cpython-312.pyc 2>/dev/null || true
git rm --cached __pycache__/modal_train.cpython-39.pyc 2>/dev/null || true
git rm --cached evaluation/__pycache__/countdown_eval.cpython-39.pyc 2>/dev/null || true
git rm --cached sft_trainer/.DS_Store 2>/dev/null || true
git rm --cached rloo_trainer/.DS_Store 2>/dev/null || true
git rm --cached .DS_Store 2>/dev/null || true

echo "===> Step 5/6: move launch scripts into joint_trainer/scripts/"
mkdir -p joint_trainer/scripts
for f in launch_v7.sh eval_v7.sh eval_fair.sh; do
    if [ -f "$f" ]; then
        mv "$f" "joint_trainer/scripts/$f"
        echo "  moved $f -> joint_trainer/scripts/$f"
    fi
done

echo "===> Step 6/6: append patterns to .gitignore"
# Append only entries that aren't already present
add_to_gitignore() {
    local pat="$1"
    if ! grep -qxF "$pat" .gitignore 2>/dev/null; then
        echo "$pat" >> .gitignore
    fi
}

# Header for our additions
if ! grep -q "# --- added by cleanup_for_push.sh" .gitignore 2>/dev/null; then
    cat >> .gitignore <<'EOF'

# --- added by cleanup_for_push.sh ---
# OS / editors
**/.DS_Store
*.swp
*.swo

# Archives / large artifacts
*.zip

# Eval results & datasets (regeneratable)
eval_results_joint/
downloaded_eval_results
rloo_step10.json
rloo_trainer/rloo_eval.json
data/

# Build / binary artifacts
*.docx
poster_assets/*.pdf

# Internal notes (not for public repo)
notes/

# Local checkpoints / wandb dirs
checkpoints/
wandb/
EOF
fi

echo
echo "===> CLEANUP COMPLETE."
echo
echo "Next steps (manual):"
echo "  1. ROTATE THE WANDB KEY at https://wandb.ai/authorize"
echo "  2. Review changes: cd $REPO_ROOT && git status && git diff"
echo "  3. Verify no secrets remain: grep -rE 'wandb_v1_|hf_[a-zA-Z0-9]{30,}|sk-[a-zA-Z0-9]{20,}' . --include='*.py' --include='*.sh' --include='*.md' --exclude-dir=.git"
echo "  4. Update README.md to mention joint_trainer/ extension"
echo "  5. Commit: git add -A && git commit -m 'Add joint conjecturer/prover trainer (CS224R extension)'"
echo "  6. Push: git push"
