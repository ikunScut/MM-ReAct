# VizWiz Planner Eval

This directory runs VizWiz VQA through `mm_react.agent.planner.ImagePlanner`.
Model calls are shared with the agent instead of maintaining a separate VLM
HTTP client in `evals`.

## Prompt

The baseline uses a fixed, zero-shot prompt designed for reproducible VizWiz
VQA experiments. It asks the model to answer with a single word or phrase and
emit `Unanswerable` when the provided information is insufficient. The exact
template is defined in
`evals/vizwiz/vlm_client.py` as `VIZWIZ_PAPER_PROMPT_TEMPLATE`.

```text
{question}
When the provided information is insufficient, respond with 'Unanswerable'.
Answer the question using a single word or phrase.
```

This prompt is intentionally dataset-specific and should be reported alongside
the model, backend, decoding settings, and split when used in a paper.

## Configuration

The local dataset root defaults to `data/VizWiz`. The expected current layout is:

```text
data/VizWiz/
  annotations/{train,val,test}.json
  images/{train,val,test}/
```

Older layouts using `Annotations/{split}.json` and `{split}/` image directories
are still accepted.

In the current local checkout, only `images/train/` contains images. Use
`--split train` with the checked-in data, or download the matching `val`/`test`
images before running those splits.

Configure the planner backend in `.env.local` or the shell. For the local
transformers backend:

```bash
VIZWIZ_PLANNER_BACKEND=transformers
MM_REACT_TRANSFORMERS_MODEL=/path/to/local/vlm
MM_REACT_TRANSFORMERS_STRATEGY=chat
MM_REACT_TRANSFORMERS_SEND_IMAGE=true
```

For an OpenAI-compatible local service, use the same variables as the agent:

```bash
VIZWIZ_PLANNER_BACKEND=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=
OPENAI_MODEL=your-model-name
OPENAI_SEND_IMAGE=true
```

Run a small split:

```bash
python evals/vizwiz/run_vlm_baseline.py --split train --limit 10 --backend transformers
```

Evaluate saved predictions:

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_planner/predictions.jsonl
```

Scoring is routed through the official VizWiz API checked in under
`data/VizWiz/API`. The default output keeps the old fractional `accuracy`
field and also reports the official percentage under `accuracy_percent` and
`official_accuracy`.

If a prediction file does not contain embedded `gt_answers`, pass the matching
annotation JSON:

```bash
python evals/vizwiz/evaluate.py outputs/vizwiz_planner/predictions.jsonl \
  --annotations data/VizWiz/annotations/val.json
```

Use `--workers 1` for the transformers backend. The local model is loaded and
cached inside one `ImagePlanner` instance.
