#!/bin/bash

# Acceleray Deployment Script
# Deploy to Databricks workspace: /Workspace/Shared/acceleray/

set -e  # Exit on error

echo "============================================"
echo "   Acceleray Deployment to Databricks"
echo "============================================"
echo ""
echo "⚠️  This will deploy Acceleray to your Databricks workspace"
echo "   Path: /Workspace/Shared/acceleray"
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Deployment cancelled"
    exit 0
fi

# Configuration
WORKSPACE_URL="https://dbc-47a3dcaa-ae3e.cloud.databricks.com"
WORKSPACE_PATH="/Workspace/Shared/acceleray"
PROJECT_DIR="/Users/analytics360/databricks/ray"

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI not found. Installing..."
    pip install databricks-cli
    echo "✅ Databricks CLI installed"
fi

# Check if configured
if [ ! -f ~/.databrickscfg ]; then
    echo "⚙️  Configuring Databricks CLI..."
    echo ""
    echo "Please provide your Databricks configuration:"
    echo "Host: $WORKSPACE_URL"
    echo ""
    databricks configure --token
else
    echo "✅ Databricks CLI already configured"
fi

echo ""
echo "📦 Deploying Acceleray to $WORKSPACE_PATH..."
echo ""

# Navigate to project directory
cd "$PROJECT_DIR"

# Deploy to Databricks
databricks workspace import-dir . "$WORKSPACE_PATH" --overwrite --exclude-hidden-files

echo ""
echo "✅ Deployment complete!"
echo ""
echo "============================================"
echo "   Verification Steps"
echo "============================================"
echo ""
echo "1. Check deployment:"
echo "   databricks workspace ls $WORKSPACE_PATH"
echo ""
echo "2. Access in browser:"
echo "   $WORKSPACE_URL/browse?o=7474660441663212#folder/$WORKSPACE_PATH"
echo ""
echo "3. Next steps:"
echo "   - Create cluster (see DEPLOYMENT.md)"
echo "   - Install libraries (see requirements.txt)"
echo "   - Run Demo 1: $WORKSPACE_PATH/notebooks/01_hyperparameter_tuning"
echo ""
echo "============================================"
echo "   🚀 Acceleray is ready to use!"
echo "============================================"
