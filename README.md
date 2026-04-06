# coverse

`coverse` 是一个研究型实验仓库，核心目标是支持多个课题并行推进，同时把可复用的能力沉淀到最小公共层。当前仓库重点覆盖两类研究任务：

- 多 Agent 对话共创：让多个 Agent 轮流续写故事，批量生成 transcript 和故事样本。
- 概率分析课题：用 BERT masked LM 评估特定短语在给定上下文中的概率。

## 项目结构

```text
coverse/
  core/      共享基础能力：agent、backend、实验 IO
  topics/    按课题拆分的研究目录
  apps/      可交互 demo，例如 Gradio
```

当前主要目录：

- `coverse/core/agents/`: 单 Agent 和多 Agent 编排。
- `coverse/core/backends/`: OpenAI-compatible 模型后端。
- `coverse/core/io/`: 实验结果、metadata、CSV/JSON 输出。
- `coverse/topics/first_sentence_baseline/`: 第一启动句启发性基线实验，文档与代码同目录。
- `coverse/topics/multi_chat/`: 多 Agent 对话生成课题。
- `coverse/topics/prob_detect/`: 概率分析课题。
- `coverse/apps/gradio_app.py`: 可交互聊天 demo。

## 安装

```bash
uv sync
```

或使用现有虚拟环境后直接安装：

```bash
pip install -e .
```

## 运行方式

直接运行对应脚本文件。

### 1. 多 Agent 批量生成

```bash
.venv/bin/python coverse/topics/multi_chat/runner.py \
  --prompts-path data/coverse_pe/story_prompt.txt \
  --agent-name agent_1 \
  --agent-name agent_2 \
  --n-turns 5 \
  --concurrency 4 \
  --tag baseline
```

默认会从 `.env` 读取这组 LLM 配置：

```env
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=...
LLM_MODEL=deepseek-chat
```

输出内容：

- `metadata.json`: 本次运行的参数和模型信息
- `results.json`: 完整 transcript
- `results.csv`: 便于后处理的扁平化结果

### 2. 启动句启发性基线

```bash
.venv/bin/python coverse/topics/first_sentence_baseline/llm_sample.py \
  --output-dir outputs
```

实验说明文档见 [coverse/topics/first_sentence_baseline/README.md](/Users/eric/projects/coverse/coverse/topics/first_sentence_baseline/README.md)。
默认材料文件在 [prompt.json](/Users/eric/projects/coverse/data/first_sentence_baseline/prompt.json)。
默认 embedding 模型路径是 `data/models/Qwen/Qwen3-Embedding-0.6B`。

### 3. 概率分析

```bash
.venv/bin/python coverse/topics/prob_detect/runner.py \
  --model-path data/models/google-bert/bert-base-chinese \
  --target 嚼馒头 \
  --text "学习，就像[MASK][MASK][MASK]，因为久了方觉甜"
```

也可以通过 `--text-file` 传入多条样本。

### 4. 启动 Gradio demo

## 研究扩展约定

- 新课题默认放在 `coverse/topics/<topic_name>/`。
- 每个独立实验目录都应自带 `README.md`，把实验设计与运行方式和代码放在一起。
- 共享层只沉淀至少被两个课题复用的能力。
- 关键实验参数优先通过脚本参数传入，不依赖手改源码常量。
- 每次运行默认写出 metadata，保证实验留痕和复现。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
