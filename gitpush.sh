#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <repository> <commit-message> <branch>"
    exit 1
fi

REPOSITORY="$1"
MESSAGE="$2"
BRANCH="$3"

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