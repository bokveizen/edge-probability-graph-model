#!/usr/bin/env bash

set -uo pipefail

# Location of the compiled 4-clique counter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNTER_BIN="${SCRIPT_DIR}/count_4_cliques"

if [[ ! -x "${COUNTER_BIN}" ]]; then
  echo "count_4_cliques binary not found at ${COUNTER_BIN}." >&2
  echo "Compile it with:" >&2
  echo "  cd ${SCRIPT_DIR} && g++ -O2 -fopenmp -o count_4_cliques count_4_cliques.cpp" >&2
  exit 1
fi

# Root directory containing original ER graphs (default to data/orig_er relative to script dir)
if [[ -n "${1:-}" ]]; then
  ROOT_DIR="${1}"
else
  ROOT_DIR="${SCRIPT_DIR}/../data/orig_er"
fi

# Resolve to absolute path
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"

if [[ ! -d "${ROOT_DIR}" ]]; then
  echo "Error: Root directory not found: ${ROOT_DIR}" >&2
  exit 1
fi

echo "# dataset graph n_nodes n_4_cliques"

shopt -s nullglob

for graph_file in "${ROOT_DIR}"/*.txt; do
  [[ -f "${graph_file}" ]] || continue

  graph_base="$(basename "${graph_file}")"
  # Remove trailing .txt
  name_no_ext="${graph_base%.txt}"
  # Split on last underscore to separate dataset and index
  dataset="${name_no_ext%_*}"

  # Run the counter
  output="$("${COUNTER_BIN}" "${graph_file}" 2>/dev/null)"
  n_nodes="$(printf "%s\n" "${output}" | awk '/Number of nodes:/ {print $NF}')"
  n_4c="$(printf "%s\n" "${output}" | awk '/Number of 4-cliques:/ {print $NF}')"

  if [[ -z "${n_nodes}" || -z "${n_4c}" ]]; then
    echo "Warning: Failed to parse counts for ${graph_file}" >&2
    continue
  fi

  echo "${dataset} ${graph_base} ${n_nodes} ${n_4c}"
done

shopt -u nullglob


