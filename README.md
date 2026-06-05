# MM-ReAct

MM-ReAct 是一个面向视觉问答（VQA）的多模态 ReAct Agent。它让视觉语言模型先判断当前图像是否足以回答问题；如果不够，再按需调用图像处理或检测工具，随后基于工具观察继续推理并给出最终答案。

当前仓库包含：

- `mm_react/`：Agent、Planner、Executor、Memory 和工具注册表。
- `evals/vizwiz/`：VizWiz 普通 VLM baseline、数据加载和官方指标封装。
- `evals/vizwiz_agent/`：VizWiz 上的 MM-ReAct Agent 评测入口。
- `sft/`：基于 VizWiz 生成 ReAct SFT 数据、校验数据、LoRA/QLoRA 微调。
- `data/VizWiz/`：VizWiz 标注、图片和官方 Python 评测工具。
- `tests/`：单图 Planner/Agent 调试脚本。

## 工作流

MM-ReAct 的一次运行由三部分组成：

1. `ImagePlanner` 调用模型生成结构化决策。
2. `ImageExecutor` 按决策执行一个工具调用。
3. `AgentMemory` 记录用户请求、模型输出、工具观察和最终答案。

模型输出必须符合以下格式：

```text
<thought>
One short sentence.
</thought>
<tool>
{"tool_name": "...", "args": {...}}
</tool>
<final_answer>

</final_answer>
```

最终回答时，`<tool>` 必须为 `null`，`<final_answer>` 写入答案。具体规则在 `mm_react/prompts/` 中维护。

## 环境安装

建议使用独立 Python 环境：

```bash
conda create -n mm-react python=3.10 -y
conda activate mm-react
```

仓库目前没有 `requirements.txt`，可按使用场景安装依赖：

```bash
# 基础运行、OpenAI-compatible backend、本地图像工具
pip install openai pillow

# transformers backend 和训练
pip install torch torchvision transformers accelerate peft bitsandbytes

# VizWiz 评测工具依赖
pip install numpy scikit-learn matplotlib scikit-image
```

如果只调用 OpenAI-compatible 服务，不需要安装本地 `transformers` 模型依赖。若使用 `--caption-metrics`，官方 VizWiz caption 指标还需要 Java，因为其中包含 Stanford CoreNLP 和 METEOR jar。

## 配置

项目会自动读取仓库根目录的 `.env.local`，并且不会覆盖 shell 中已经存在的环境变量。

### OpenAI-compatible 后端

适用于 vLLM、SGLang、DashScope compatible mode 或其他兼容 OpenAI Chat Completions 的服务。

```bash
VIZWIZ_PLANNER_BACKEND=openai
VIZWIZ_AGENT_BACKEND=openai

OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=EMPTY
OPENAI_MODEL=/model
OPENAI_SEND_IMAGE=true
OPENAI_ENABLE_THINKING=false
```

说明：

- `OPENAI_BASE_URL`：OpenAI-compatible 服务地址。
- `OPENAI_API_KEY`：本地服务通常可填 `EMPTY`。
- `OPENAI_MODEL`：传给接口的模型名。
- `OPENAI_SEND_IMAGE`：是否把当前图像作为 `image_url` data URL 发送。
- `OPENAI_ENABLE_THINKING`：若后端支持 Qwen thinking 参数，会写入 `extra_body.chat_template_kwargs.enable_thinking`。

### Transformers 后端

适用于本地 Hugging Face/ModelScope 模型，尤其是 InternVL 类带 `.chat()` 接口的模型。

```bash
VIZWIZ_PLANNER_BACKEND=transformers
VIZWIZ_AGENT_BACKEND=transformers

MM_REACT_TRANSFORMERS_MODEL=/path/to/local/vlm
MM_REACT_TRANSFORMERS_STRATEGY=chat
MM_REACT_TRANSFORMERS_SEND_IMAGE=true
MM_REACT_TRANSFORMERS_DEVICE=cuda
MM_REACT_TRANSFORMERS_TORCH_DTYPE=bfloat16
```

常用变量：

- `MM_REACT_TRANSFORMERS_STRATEGY=chat` 或 `internvl`：使用 `AutoModel.chat()`。
- `MM_REACT_TRANSFORMERS_STRATEGY=vision2seq` 或 `processor`：使用 `AutoProcessor` 和 `AutoModelForImageTextToText`。
- `MM_REACT_TRANSFORMERS_MAX_NEW_TOKENS`：默认 `512`。
- `MM_REACT_TRANSFORMERS_DO_SAMPLE`：默认 `false`。
- `MM_REACT_TRANSFORMERS_TEMPERATURE`、`MM_REACT_TRANSFORMERS_TOP_P`、`MM_REACT_TRANSFORMERS_TOP_K`：可选生成参数。
- `MM_REACT_TRANSFORMERS_IMAGE_SIZE`：InternVL 图像 tile 尺寸，默认 `448`。
- `MM_REACT_TRANSFORMERS_MAX_IMAGE_TILES`：默认 `12`。
- `MM_REACT_TRANSFORMERS_TRUST_REMOTE_CODE`：默认 `true`。
- `MM_REACT_TRANSFORMERS_USE_FLASH_ATTN`：默认 `false`。

使用 transformers 后端跑数据集时请保持 `--workers 1`，代码会在一个 `ImagePlanner` 中缓存本地模型。

## 工具服务

Agent 可调用的工具由 `mm_react/tools/__init__.py` 注册：

| 工具名 | 功能 | 默认实现 |
| --- | --- | --- |
| `object_detection_image` | 开放词表目标检测，返回 box JSON | HTTP API |
| `low_light_enhance` | 低光照增强 | HTTP API |
| `nafnet_image_restoration` | 去噪/去模糊 | HTTP API |
| `rotate_image` | 旋转图像 | 本地 Pillow |
| `super_resolution_image` | 超分辨率 | HTTP API |
| `zoom_in_image` | 按检测框裁剪放大 | 本地 Pillow |

HTTP 工具默认地址如下，可在 `.env.local` 覆盖：

```bash
MM_REACT_GROUNDING_DINO_API_URL=http://127.0.0.1:8004/detect
MM_REACT_LOW_LIGHT_API_URL=http://127.0.0.1:8005/api/enhance
MM_REACT_NAFNET_API_URL=http://127.0.0.1:8001/nafnet/process
MM_REACT_SUPER_RESOLUTION_API_URL=http://127.0.0.1:8002/enhance
MM_REACT_TOOL_TIMEOUT_SECONDS=900
```

兼容旧变量：

```bash
SUPER_RESOLUTION_API_URL=http://127.0.0.1:8002/enhance
TOOL_TIMEOUT_SECONDS=900
```

如果某个 HTTP 工具服务未启动，只有当 Planner 选择该工具时运行才会失败。普通 baseline 不调用这些工具。

## 数据集布局

默认 VizWiz 根目录为 `data/VizWiz`，当前代码优先识别：

```text
data/VizWiz/
  annotations/
    train.json
    val.json
    test.json
  images/
    train/
    val/
    test/
  API/
```

也兼容旧布局：

```text
data/VizWiz/
  Annotations/{split}.json
  Images/{split}/
```

如果本机某些图片缺失，可给评测脚本加 `--skip-missing`。

## 单图调试

`tests/test_planner.py` 和 `tests/test_react.py` 是可直接运行的调试脚本。运行前先在脚本顶部修改 `IMAGE_PATH`、`QUESTION` 和 `PLANNER_BACKEND`。

只测试 Planner：

```bash
python tests/test_planner.py
```

运行完整 ReAct Agent：

```bash
python tests/test_react.py
```

Agent 输出图片默认写入 `outputs/react_test/`。如果模型输出格式错误，脚本会打印最后一次 prompt 和原始模型输出，便于修正 prompt 或后端配置。

## VizWiz 普通 VLM Baseline

运行少量样本：

```bash
python evals/vizwiz/run_vlm_baseline.py \
  --split val \
  --limit 10 \
  --backend openai \
  --overwrite
```

Transformers 后端示例：

```bash
python evals/vizwiz/run_vlm_baseline.py \
  --split val \
  --limit 10 \
  --backend transformers \
  --workers 1 \
  --overwrite
```

默认输出：

```text
outputs/vizwiz_planner/predictions.jsonl
```

评分：

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_planner/predictions.jsonl
```

如果预测文件不包含 `gt_answers`，需显式传入标注：

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_planner/predictions.jsonl \
  --annotations data/VizWiz/annotations/val.json
```

写出评分 JSON：

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_planner/predictions.jsonl \
  --output outputs/vizwiz_planner/metrics.json
```

## VizWiz Agent 评测

运行少量样本：

```bash
python evals/vizwiz_agent/run_agent.py \
  --split val \
  --limit 10 \
  --backend openai \
  --max-turns 4 \
  --overwrite
```

Transformers 后端：

```bash
python evals/vizwiz_agent/run_agent.py \
  --split val \
  --limit 10 \
  --backend transformers \
  --workers 1 \
  --max-turns 4 \
  --overwrite
```

常用参数：

- `--start-index`：从第几个样本开始。
- `--limit`：处理样本数；`0` 表示处理到 split 末尾。
- `--output`：预测 JSONL 路径，默认 `outputs/vizwiz_agent/predictions.jsonl`。
- `--image-output-dir`：工具生成图片目录，默认 `outputs/vizwiz_agent/images`。
- `--max-turns`：每个样本最多 ReAct 轮数。
- `--include-prompt`：在输出中保存实际用户 prompt。
- `--overwrite`：覆盖已有输出文件。

评分：

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_agent/predictions.jsonl
```

Agent 输出除 baseline 字段外，还会包含 `final_image`、`num_steps`、`tools`、`steps` 和 `trace`，用于排查每次工具调用。

## SFT 数据生成

使用 OpenAI-compatible teacher planner 跑完整 Agent，并把每一轮 student-visible prompt 和 assistant ReAct 输出保存为 SFT JSONL：

```bash
python sft/generation/generate_vizwiz_sft.py \
  --vizwiz-root data/VizWiz \
  --split train \
  --limit 100 \
  --output outputs/sft_data/vizwiz_train_react_sft.jsonl \
  --image-output-dir outputs/sft_generation/images \
  --max-turns 8 \
  --continue-on-error
```

可选保存 trace：

```bash
python sft/generation/generate_vizwiz_sft.py \
  --split train \
  --limit 100 \
  --save-traces \
  --trace-dir outputs/sft_generation/traces
```

生成脚本会把 VizWiz gold answers 作为 teacher-only supervision 传给 Planner，用于指导工具选择和停止时机；这些信息不应出现在 student-visible SFT 记录中。

## SFT 数据校验

```bash
python sft/validation/validate_vizwiz_sft.py \
  --data outputs/sft_data/vizwiz_train_react_sft.jsonl \
  --vizwiz-root data/VizWiz \
  --split train
```

常用选项：

- `--no-check-images`：不检查记录里的图片路径是否存在。
- `--show-issues 0`：只打印统计，不打印具体问题。
- `--fail-on-warning`：存在 warning 时也返回非零退出码。

校验会检查 JSONL 格式、消息结构、图片路径、ReAct 标签格式、工具名、teacher-only 信息泄漏和最大轮数失败文本。

## LoRA/QLoRA 训练

示例：

```bash
python sft/training/train_lora_sft.py \
  --model-path /path/to/base-vlm \
  --train-data outputs/sft_data/vizwiz_train_react_sft.jsonl \
  --output-dir outputs/sft_lora/vizwiz_react_lora \
  --load-in-4bit \
  --bf16 \
  --gradient-checkpointing
```

常用参数：

- `--eval-data`：验证集 JSONL。
- `--image-root`：当 JSONL 中图片路径来自另一台机器时，用该目录按文件名重定位图片。
- `--model-loader auto|vision2seq|causal-lm|auto-model`：选择模型加载器。
- `--max-length`：最大 token 长度，默认 `4096`。
- `--epochs`、`--batch-size`、`--grad-accum`、`--learning-rate`：训练超参数。
- `--lora-r`、`--lora-alpha`、`--lora-dropout`、`--lora-target-modules`：LoRA 配置。
- `--resume-from-checkpoint`：从 checkpoint 继续训练。

训练脚本会保存 LoRA adapter，以及 processor/tokenizer 到 `--output-dir`。

## 输出文件

常见输出路径：

```text
outputs/
  vizwiz_planner/
    predictions.jsonl
  vizwiz_agent/
    predictions.jsonl
    images/
  sft_data/
    vizwiz_train_react_sft.jsonl
  sft_generation/
    images/
    traces/
    errors.jsonl
  sft_lora/
```

## 常见问题

### `Output already exists`

评测脚本默认不覆盖已有输出。加 `--overwrite`，或换一个 `--output` 路径。

### Transformers 后端并发报错

本地模型被缓存到一个 Planner 实例里。使用 transformers 后端时保持 `--workers 1`。

### 模型输出解析失败

Planner 要求模型严格输出 `<thought>`、`<tool>`、`<final_answer>` 三段。运行 `tests/test_planner.py` 查看实际 prompt 和 raw output，再调整模型、prompt 或生成参数。

### 工具 API 调用失败

确认对应服务已启动，并且 `.env.local` 中的 API URL 指向正确端口。`rotate_image` 和 `zoom_in_image` 不需要 HTTP 服务；其他工具需要。

### 图片路径不存在

检查 `data/VizWiz/images/{split}/` 是否包含对应图片。临时跑通可以加 `--skip-missing`。

## 开发提示

- 修改工具列表时，同步更新 `mm_react/tools/__init__.py` 和 `mm_react/prompts/tools_prompt`。
- 修改 Planner 输出格式时，同步更新 `mm_react/prompts/output_format_prompt`、`ImagePlanner._parse_decision()` 和 SFT 校验脚本。
- 数据集评测共用 `evals/vizwiz/data.py` 和 `evals/vizwiz/metrics.py`，新增 split 或布局兼容优先放在那里。
