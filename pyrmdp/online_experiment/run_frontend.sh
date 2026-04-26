#!/usr/bin/env bash
# Convenience launcher for the Streamlit frontend.
set -e
cd "$(dirname "$0")/../.."
export PYRMDP_BACKEND_URL="${PYRMDP_BACKEND_URL:-http://localhost:8000}"
exec streamlit run pyrmdp/online_experiment/frontend/app.py "$@"
