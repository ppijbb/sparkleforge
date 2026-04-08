#!/bin/bash
# SparkleForge - MCP Only, No Fallbacks
# Run with: bash run.sh

export ENABLE_AUTO_FALLBACK=false
export MCP_ENABLED=true

echo "Starting SparkleForge - MCP Only - No Fallbacks"
python main.py "$@"
