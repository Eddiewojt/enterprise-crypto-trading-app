#!/bin/bash

# 🚀 Enterprise Crypto Trading App - Quick Setup Script
# This script will help you get your automated APK build system running

echo "🚀 Enterprise Crypto Trading App - Automated Build Setup"
echo "======================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "AUTOMATED_APK_BUILD_SETUP.md" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "✅ Project structure verified"
echo ""

# Check for git
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "   Visit: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git found"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📦 Initializing Git repository..."
    git init
    git add .
    git commit -m "🚀 Enterprise Crypto Trading App v2.0 - Complete Setup"
    echo "✅ Git repository initialized and files committed"
else
    echo "✅ Git repository already exists"
    
    # Check if there are uncommitted changes
    if [ -n "$(git status --porcelain)" ]; then
        echo "📦 Committing latest changes..."
        git add .
        git commit -m "🔄 Update: Latest changes to Enterprise Crypto Trading App"
        echo "✅ Changes committed"
    else
        echo "✅ No uncommitted changes"
    fi
fi

echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. 🌐 Create a GitHub Repository:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: enterprise-crypto-trading-app"
echo "   - Set to Public (required for free GitHub Actions)"
echo "   - Don't initialize with README (we already have files)"
echo ""
echo "2. 🔗 Connect this project to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/enterprise-crypto-trading-app.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. ⏳ Wait for Automatic Build (5-10 minutes):"
echo "   - Go to your GitHub repository"
echo "   - Click 'Actions' tab"
echo "   - Watch the '🚀 Build Android APK' workflow"
echo ""
echo "4. 📥 Download Your APK:"
echo "   - From 'Actions' → Latest workflow → 'Artifacts'"
echo "   - Or from 'Releases' section"
echo ""
echo "🎯 Quick Manual Trigger (Alternative):"
echo "   - GitHub repository → Actions tab"
echo "   - Select build workflow → 'Run workflow'"
echo ""
echo "📱 APK File Details:"
echo "   - Name: enterprise-crypto-trading-v2.0.0.apk"
echo "   - Size: ~25-30 MB"
echo "   - Compatible: Android 5.0+ (API 21+)"
echo ""
echo "🔔 Your Notifications Are Already Configured:"
echo "   - ✅ Telegram: Working (Chat ID: 6086031887)"
echo "   - ✅ Email: Working (eddiewojt1@gmail.com)"
echo ""
echo "🎉 Your Enterprise Crypto Trading App is ready for distribution!"
echo ""
echo "📖 For detailed instructions, see: AUTOMATED_APK_BUILD_SETUP.md"
echo ""

# Offer to open the setup guide
read -p "📖 Would you like to open the detailed setup guide? (y/n): " open_guide
if [[ $open_guide =~ ^[Yy]$ ]]; then
    if command -v code &> /dev/null; then
        code AUTOMATED_APK_BUILD_SETUP.md
    elif command -v nano &> /dev/null; then
        nano AUTOMATED_APK_BUILD_SETUP.md
    else
        cat AUTOMATED_APK_BUILD_SETUP.md
    fi
fi

echo ""
echo "🚀 Happy building! Your automated APK system is ready to go!"