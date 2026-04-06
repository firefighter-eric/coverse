#!/usr/bin/env bash
# 这个脚本用于本地调试第一启动句基线课题：
# 按 llm_sample -> embedding_similarity -> analysis 的顺序串联执行三步。

set -euo pipefail

PROMPTS_FILE="data/first_sentence_analysis/prompt.json"
SAMPLES_PER_PROMPT=30
TEMPERATURE=1.0
EMBEDDING_MODEL_PATH="data/models/Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR="data/first_sentence_analysis/v1"
SAMPLES_PATH="${OUTPUT_DIR}/llm_samples.json"
SIMILARITIES_PATH="${OUTPUT_DIR}/embedding_similarity.json"
ANALYSIS_PATH="${OUTPUT_DIR}/analysis_details.json"
mkdir -p "${OUTPUT_DIR}"


echo "[1/3] 运行 llm_sample"
sample_output_json="${OUTPUT_DIR}/llm_sample_output.json"
python "coverse/topics/first_sentence_analysis/llm_sample.py" \
  --output-path "${SAMPLES_PATH}" \
  --prompts-file "${PROMPTS_FILE}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --temperature "${TEMPERATURE}" > "${sample_output_json}"
cat "${sample_output_json}"

echo
echo "[2/3] 运行 embedding_similarity"
embedding_output_json="${OUTPUT_DIR}/embedding_similarity_output.json"
python "coverse/topics/first_sentence_analysis/embedding_similarity.py" \
  --samples-path "${SAMPLES_PATH}" \
  --embedding-model-path "${EMBEDDING_MODEL_PATH}" \
  --output-path "${SIMILARITIES_PATH}" > "${embedding_output_json}"
cat "${embedding_output_json}"

echo
echo "[3/3] 运行 analysis"
analysis_output_json="${OUTPUT_DIR}/analysis_output.json"
python "coverse/topics/first_sentence_analysis/analysis.py" \
  --similarities-path "${SIMILARITIES_PATH}" \
  --output-path "${ANALYSIS_PATH}" > "${analysis_output_json}"
cat "${analysis_output_json}"

echo
echo "流水线完成。"
