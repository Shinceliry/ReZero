#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="InvisibleMic"
MESSAGE="$1"
BRANCH="$2"

# Assuming USER_NAME is exported in ~/.bash_profile
TARGET_DIR="/home/${USER_NAME}/${REPOSITORY}"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory does not exist: $TARGET_DIR"
    exit 1
fi

cd "$TARGET_DIR"
git add .
git commit -m "$MESSAGE"
git push origin "$BRANCH"