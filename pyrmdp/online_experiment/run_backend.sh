#!/usr/bin/env bash
# Convenience launcher for the backend.
set -e
cd "$(dirname "$0")/../.."
export PYRMDP_DOMAIN_PATH="${PYRMDP_DOMAIN_PATH:-./pipeline_output/robustified.ppddl}"
export PYRMDP_EPS_PHYS="${PYRMDP_EPS_PHYS:-0.35}"
export PYRMDP_EPS_SPECTRAL="${PYRMDP_EPS_SPECTRAL:-0.02}"
export PYRMDP_OUTPUT_DIR="${PYRMDP_OUTPUT_DIR:-./online_output}"
# Bind to all interfaces so phones / other devices on the same WiFi
# can reach the backend at http://<laptop-IP>:8000.
exec uvicorn pyrmdp.online_experiment.backend.server:app \
    --host 0.0.0.0 --port 8000 "$@"
