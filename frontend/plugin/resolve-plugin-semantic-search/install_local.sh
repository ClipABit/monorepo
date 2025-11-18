#!/usr/bin/env bash
set -euo pipefail

# Install this plugin locally for DaVinci Resolve on macOS.
# Default mode installs as a Workflow Integration bundle.
# Use --script to install the Python script into a page's Scripts menu instead.
#   Optional: --page <Comp|Edit|Color|Deliver|Media|Cut|Fairlight> (default: Edit)
# Use --uninstall to remove previously installed files for the selected mode.
#
# Examples:
#   ./install_local.sh                # Install Workflow Integration
#   ./install_local.sh --script       # Install as Script (Workspace > Scripts > Media)
#   ./install_local.sh --uninstall    # Uninstall Workflow Integration
#   ./install_local.sh --script --uninstall   # Uninstall Script install
#
# Notes:
# - This script only targets macOS paths.
# - Restart Resolve after installing/uninstalling.

MODE="workflow"
UNINSTALL="false"
PAGE="Edit"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script)
      MODE="script"; shift ;;
    --uninstall)
      UNINSTALL="true"; shift ;;
    --page)
      PAGE="${2:-Edit}"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--script] [--uninstall] [--page <Comp|Edit|Color|Deliver|Media|Cut|Fairlight>]" >&2
      exit 1 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer currently supports macOS only." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="${SCRIPT_DIR}/plugin"

if [[ ! -d "${PLUGIN_DIR}" ]]; then
  echo "Plugin directory not found: ${PLUGIN_DIR}" >&2
  exit 1
fi

BASE="$HOME/Library/Application Support/Blackmagic Design/DaVinci Resolve"

if [[ "${MODE}" == "workflow" ]]; then
  TARGET_DIR="${BASE}/Workflow Integrations/semantic-search"
  if [[ "${UNINSTALL}" == "true" ]]; then
    echo "Uninstalling Workflow Integration from: ${TARGET_DIR}"
    rm -rf "${TARGET_DIR}"
    echo "Done. Restart DaVinci Resolve."
    exit 0
  fi
  echo "Installing Workflow Integration to:"
  echo "  ${TARGET_DIR}"
  mkdir -p "${TARGET_DIR}"
  rsync -a --delete --exclude=".*" "${PLUGIN_DIR}/" "${TARGET_DIR}/"
  echo "Installed. Restart DaVinci Resolve and open the Media page."
else
  # Script install to selected page(s)
  PAGES=("${PAGE}")
  if [[ "${PAGE}" == "All" ]]; then
    PAGES=("Comp" "Edit" "Color" "Deliver" "Media" "Cut" "Fairlight")
  fi
  # Support both common Resolve layouts (with and without 'Support')
  BASES=(
    "${BASE}/Support"
    "${BASE}"
  )
  for p in "${PAGES[@]}"; do
    for b in "${BASES[@]}"; do
      SCRIPTS_DIR="${b}/Fusion/Scripts/${p}"
      TARGET_PY="${SCRIPTS_DIR}/semantic_search.py"
      if [[ "${UNINSTALL}" == "true" ]]; then
        echo "Uninstalling Script from: ${TARGET_PY}"
        rm -f "${TARGET_PY}" || true
        continue
      fi
      echo "Installing Script to:"
      echo "  ${TARGET_PY}"
      mkdir -p "${SCRIPTS_DIR}"
      cp -f "${PLUGIN_DIR}/Scripts/semantic_search.py" "${TARGET_PY}"
    done
  done
  if [[ "${UNINSTALL}" == "true" ]]; then
    echo "Done. Restart DaVinci Resolve."
  else
    echo "Installed. Restart DaVinci Resolve, then check Workspace → Scripts for your chosen page(s)."
  fi
fi


