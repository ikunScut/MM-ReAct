# VizWiz Agent Eval

This directory evaluates the MM-ReAct tool-using agent on VizWiz VQA. It is
kept separate from `evals/vizwiz`, which contains the plain VLM baseline,
shared VizWiz data loading, and shared scoring code.

## Run

The dataset root defaults to `data/VizWiz` and uses the same layout as the
baseline:

```text
data/VizWiz/
  annotations/{train,val,test}.json
  images/{train,val,test}/
```

Run a small local split:

```bash
python evals/vizwiz_agent/run_agent.py \
  --split train \
  --limit 10 \
  --backend transformers \
  --max-turns 4 \
  --output outputs/vizwiz_agent/predictions.jsonl
```

For an OpenAI-compatible backend:

```bash
python evals/vizwiz_agent/run_agent.py \
  --split val \
  --backend openai \
  --workers 4 \
  --max-turns 4
```

Use `--workers 1` for the transformers backend because the local model is kept
in one shared `ImagePlanner` instance.

## Evaluate

The prediction JSONL keeps the same fields used by the baseline scorer:

```json
{
  "image": "VizWiz_train_00000000.jpg",
  "question": "...",
  "prediction": "...",
  "gt_answers": [],
  "answerable": 1,
  "answer_type": "..."
}
```

It also records agent-specific debugging fields such as `final_image`, `steps`,
`tools`, and `trace`.

Evaluate with the existing VizWiz evaluator:

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_agent/predictions.jsonl
```

Agent runs never pass VizWiz gold answers into `ReActAgent.run()`. Gold answers
are stored only in the output JSONL for offline scoring.
