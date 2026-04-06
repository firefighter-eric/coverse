#!/usr/bin/env bash
# 这个脚本用于本地调试第一启动句基线课题：
# 按 llm_sample -> embedding_similarity -> analysis 的顺序串联执行三步。

set -euo pipefail

source ".venv/bin/activate"

SAMPLES_PER_PROMPT=30
TEMPERATURE=1.0
EMBEDDING_MODEL_PATH="data/models/Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR="data/first_sentence_analysis/v1"
PROMPTS_FILE="${OUTPUT_DIR}/prompt.json"
SYSTEM_PROMPT_FILE="${OUTPUT_DIR}/system_prompt.md"
SAMPLES_PATH="${OUTPUT_DIR}/llm_samples.json"
SIMILARITIES_PATH="${OUTPUT_DIR}/embedding_similarities.json"
ANALYSIS_PATH="${OUTPUT_DIR}/analysis.json"

mkdir -p "${OUTPUT_DIR}"


echo "[1/3] 运行 llm_sample"
python "coverse/topics/first_sentence_analysis/llm_sample.py" \
  --output-path "${SAMPLES_PATH}" \
  --prompts-file "${PROMPTS_FILE}" \
  --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --temperature "${TEMPERATURE}"

echo
echo "[2/3] 运行 embedding_similarity"
python "coverse/topics/first_sentence_analysis/embedding_similarity.py" \
  --samples-path "${SAMPLES_PATH}" \
  --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
  --output-path "${SIMILARITIES_PATH}"

echo
echo "[3/3] 运行 analysis"
analysis_output_json="${OUTPUT_DIR}/analysis_output.json"
python "coverse/topics/first_sentence_analysis/analysis.py" \
  --similarities-path "${SIMILARITIES_PATH}" \
  --output-path "${ANALYSIS_PATH}"

echo
echo "流水线完成。"
