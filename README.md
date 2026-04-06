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
  cli/       统一命令入口
```

当前主要目录：

- `coverse/core/agents/`: 单 Agent 和多 Agent 编排。
- `coverse/core/backends/`: OpenAI-compatible 模型后端。
- `coverse/core/io/`: 实验结果、metadata、CSV/JSON 输出。
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

## CLI 用法

安装后统一使用 `coverse` 命令。

### 1. 多 Agent 批量生成

```bash
coverse topic multi-chat \
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

### 2. 概率分析

```bash
coverse topic prob-detect \
  --model-path data/models/google-bert/bert-base-chinese \
  --target 嚼馒头 \
  --text "学习，就像[MASK][MASK][MASK]，因为久了方觉甜"
```

也可以通过 `--text-file` 传入多条样本。

### 3. 启动 Gradio demo

```bash
coverse app serve \
  --output-dir outputs
```

## 研究扩展约定

- 新课题默认放在 `coverse/topics/<topic_name>/`。
- 共享层只沉淀至少被两个课题复用的能力。
- 关键实验参数优先通过 CLI 传入，不依赖手改源码常量。
- 每次运行默认写出 metadata，保证实验留痕和复现。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```
