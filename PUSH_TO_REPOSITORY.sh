#!/bin/bash

# MCADS Codebase Push Script
# Repository: https://github.com/PauleNajus/MCADS.git
# Branch: prod

echo "=========================================="
echo "MCADS Codebase Push to GitHub Repository"
echo "Repository: https://github.com/PauleNajus/MCADS.git"
echo "Branch: prod"
echo "=========================================="

# Navigate to the repository directory
cd /opt/mcads/app

# Set safe directory (in case of ownership issues)
echo "1. Setting safe directory..."
git config --global --add safe.directory /opt/mcads/app

# Check current status
echo -e "\n2. Checking repository status..."
git status

# Add all files (modified and untracked)
echo -e "\n3. Adding all files to git..."
git add .

# Show what will be committed
echo -e "\n4. Files staged for commit:"
git status --short

# Commit all changes
echo -e "\n5. Committing changes..."
git commit -m "Update codebase - deployment ready version with all modifications and new files

- Updated Django settings and configuration
- Enhanced X-ray analysis functionality  
- Added deployment scripts and monitoring tools
- Updated static files and templates
- Added comprehensive logging and debugging utilities
- Ready for production deployment"

# Fetch latest changes (to check for conflicts)
echo -e "\n6. Fetching latest from remote..."
git fetch origin

# Push to prod branch with force (to overwrite any conflicts as requested)
echo -e "\n7. Pushing to prod branch..."
echo "Attempting force-with-lease first (safer option)..."
if git push --force-with-lease origin prod; then
    echo "✅ Successfully pushed with force-with-lease!"
else
    echo "Force-with-lease failed, trying regular force push..."
    if git push --force origin prod; then
        echo "✅ Successfully pushed with force!"
    else
        echo "❌ Push failed! Check network connection and authentication."
        exit 1
    fi
fi

echo -e "\n=========================================="
echo "✅ SUCCESS: Codebase successfully pushed!"
echo "Repository: https://github.com/PauleNajus/MCADS.git"
echo "Branch: prod"
echo "All local changes have been committed and pushed."
echo "Any previous conflicts have been overwritten as requested."
echo "==========================================" 