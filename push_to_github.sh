#!/bin/bash

# Script to push MultiPhishGuard to GitHub
# Usage: ./push_to_github.sh <your-github-repo-url>
# Example: ./push_to_github.sh https://github.com/username/MultiPhishGuard.git

if [ -z "$1" ]; then
    echo "❌ Error: Please provide your GitHub repository URL"
    echo ""
    echo "Usage: ./push_to_github.sh <your-github-repo-url>"
    echo ""
    echo "If you haven't created a repository yet:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository (e.g., 'MultiPhishGuard')"
    echo "3. Do NOT initialize it with README, .gitignore, or license"
    echo "4. Copy the repository URL"
    echo "5. Run this script with that URL"
    exit 1
fi

REPO_URL=$1

echo "🚀 Setting up GitHub remote and pushing..."
echo ""

# Add remote
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your code is now on GitHub at: $REPO_URL"

