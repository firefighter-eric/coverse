# 第一启动句启发性基线

## 研究目的

这个课题用于检验“第一句启动词”的启发性和发散性。  
核心问题是：不同启动句是否会系统性影响语言模型生成下一句时的多样性。

## 实验材料

实验默认从 `data/first_sentence_baseline/prompt.json` 读取 20 个启动句，整理自三个日常情境：

- 公共空间与街道
- 室内与办公场景
- 居家与个人动作

场景标签仅用于分析输出，不进入模型提示词。

## 流水线

这个 topic 现在拆成 3 个独立脚本阶段：

1. `llm_sample.py`
   对每个启动句采样 30 个下一句，输出原始回答和清洗后回答。
2. `embedding_similarity.py`
   用 embedding 模型分别编码“原始 prompt”和“回答”，计算 prompt-response cosine similarity。
3. `analysis.py`
   按每个 prompt 聚合相似度与距离统计，并生成排序结果。

## 固定口径

- 默认参数：`temperature=1.0`，`samples_per_prompt=30`
- 生成模型：默认读取当前 `.env` 中的 DeepSeek 配置
- 生成目标：只续写“紧接着的下一句”
- 文本约束：`7-20` 字、单句、不换行
- 相似度定义：`cosine_similarity(prompt_embedding, response_embedding)`
- 距离定义：`1 - cosine_similarity`
- 去重策略：完全相同回答先去重，再做 embedding 相似度计算
- 分析方式：按每个 prompt 聚合 30 个回答对应的 similarity / distance 均值和方差
- 排序方式：按 `prompt_response_distance_variance` 从高到低排序

## 输入要求

运行时需要提供本地句向量模型路径：

- 默认路径：`data/models/Qwen/Qwen3-Embedding-0.6B`
- 可通过 `--embedding-model-path /path/to/local/model` 覆盖

仓库不内置句向量模型。实现会用 `transformers` 载入本地模型，并对句向量做 mean pooling + L2 normalize。

## 运行方式

```bash
.venv/bin/python coverse/topics/first_sentence_baseline/llm_sample.py \
  --output-dir outputs
```

```bash
.venv/bin/python coverse/topics/first_sentence_baseline/embedding_similarity.py \
  --samples-path outputs/first_sentence_baseline/<run>/llm_samples.json
```

```bash
.venv/bin/python coverse/topics/first_sentence_baseline/analysis.py \
  --similarities-path outputs/first_sentence_baseline_embeddings/<run>/embedding_similarity.json
```

如果只是本地调试，也可以直接运行串联脚本：

```bash
bash coverse/topics/first_sentence_baseline/run_pipeline.sh
```

可通过环境变量覆盖默认值，例如：

```bash
OUTPUT_DIR=outputs_debug \
SAMPLES_PER_PROMPT=5 \
TEMPERATURE=1.0 \
EMBEDDING_MODEL_PATH=data/models/Qwen/Qwen3-Embedding-0.6B \
bash coverse/topics/first_sentence_baseline/run_pipeline.sh
```

如果需要替换材料，可以在第一步传入 JSON 文件：

```bash
.venv/bin/python coverse/topics/first_sentence_baseline/llm_sample.py \
  --prompts-file /path/to/prompts.json
```

`prompts.json` 结构为：

```json
[
  {"scenario": "公共空间与街道", "text": "我在路边等红绿灯"}
]
```

如果不传 `--prompts-file`，程序默认读取：

```text
data/first_sentence_baseline/prompt.json
```

## 输出文件

第一步 `llm-sample`：
- `metadata.json`
- `llm_samples.json`

第二步 `embedding-similarity`：
- `metadata.json`
- `embedding_similarity.json`
- `embedding_similarity.csv`

第三步 `analyze`：
- `metadata.json`
- `analysis_ranking.csv`
- `analysis_details.json`

## 结果解释

建议优先查看 `analysis_ranking.csv` 中的字段：

- `avg_prompt_response_cosine_similarity`
- `prompt_response_similarity_variance`
- `avg_prompt_response_cosine_distance`
- `prompt_response_distance_variance`

其中 `prompt_response_distance_variance` 较大的启动句，说明同一启动句引出的回答与原始 prompt 的语义距离波动更大，可优先作为发散性候选材料。  
如果某个 prompt 去重后少于 2 个回答，结果会被标记为不可计算。
