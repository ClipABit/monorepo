#!/usr/bin/env bash
set -euo pipefail

# Package the plugin contents into a .drfx archive suitable for DaVinci Resolve.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="${SCRIPT_DIR}/plugin"
OUTPUT="${SCRIPT_DIR}/resolve-plugin-semantic-search.drfx"

if [[ ! -d "${PLUGIN_DIR}" ]]; then
  echo "Plugin directory not found: ${PLUGIN_DIR}" >&2
  exit 1
fi

echo "Building ${OUTPUT} ..."
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Copy plugin contents so the archive root contains manifest.json, Scripts/, UI/
rsync -a --exclude=".*" "${PLUGIN_DIR}/" "${tmpdir}/"

cd "${tmpdir}"
zip -r -X "plugin.drfx" . -x "**/.DS_Store" "**/__pycache__/**"
mv -f "plugin.drfx" "${OUTPUT}"
cd - >/dev/null
echo "Done: ${OUTPUT}"


