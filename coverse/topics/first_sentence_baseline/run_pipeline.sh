#!/usr/bin/env bash
# 这个脚本用于本地调试第一启动句基线课题：
# 按 llm_sample -> embedding_similarity -> analysis 的顺序串联执行三步。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-30}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-}"
EMBEDDING_MODEL_PATH="${EMBEDDING_MODEL_PATH:-${REPO_ROOT}/data/models/Qwen/Qwen3-Embedding-0.6B}"

cd "${REPO_ROOT}"

sample_cmd=(
  "${PYTHON_BIN}" "coverse/topics/first_sentence_baseline/llm_sample.py"
  "--output-dir" "${OUTPUT_DIR}"
  "--samples-per-prompt" "${SAMPLES_PER_PROMPT}"
  "--temperature" "${TEMPERATURE}"
)

if [[ -n "${PROMPTS_FILE}" ]]; then
  sample_cmd+=("--prompts-file" "${PROMPTS_FILE}")
fi

if [[ -n "${SEED}" ]]; then
  sample_cmd+=("--seed" "${SEED}")
fi

echo "[1/3] 运行 llm_sample"
sample_json="$("${sample_cmd[@]}")"
echo "${sample_json}"
samples_path="$(
  printf '%s\n' "${sample_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["samples_path"])'
)"

echo
echo "[2/3] 运行 embedding_similarity"
embedding_json="$(
  "${PYTHON_BIN}" "coverse/topics/first_sentence_baseline/embedding_similarity.py" \
    --samples-path "${samples_path}" \
    --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}"
)"
echo "${embedding_json}"
similarities_path="$(
  printf '%s\n' "${embedding_json}" | "${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["similarities_path"])'
)"

echo
echo "[3/3] 运行 analysis"
analysis_json="$(
  "${PYTHON_BIN}" "coverse/topics/first_sentence_baseline/analysis.py" \
    --similarities-path "${similarities_path}" \
    --output-dir "${OUTPUT_DIR}"
)"
echo "${analysis_json}"

echo
echo "流水线完成。"
