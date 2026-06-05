"""Minimal LoRA/QLoRA SFT for MM-ReAct planner records.

The training data is the JSONL produced by sft/generation/generate_vizwiz_sft.py:
each line has a user image/text message and one assistant ReAct target.

Example:
    python sft/training/train_lora_sft.py \
        --model-path /mnt/models/InternVL2-8B \
        --train-data outputs/sft_data/vizwiz_train_react_sft.jsonl \
        --output-dir outputs/sft_lora/internvl2_vizwiz_react \
        --load-in-4bit \
        --bf16
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "outputs" / "sft_data" / "vizwiz_train_react_sft.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft_lora" / "vizwiz_react_lora"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SftExample:
    user_text: str
    assistant_text: str
    image_path: Path | None
    line_number: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for MM-ReAct JSONL records.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Fallback image directory when JSONL image paths are not valid here.",
    )
    parser.add_argument(
        "--model-loader",
        choices=("auto", "vision2seq", "causal-lm", "auto-model"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--device-map", default="auto")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


class ReactSftDataset:
    def __init__(self, path: Path, image_root: Path | None = None, max_samples: int = 0) -> None:
        self.path = path.expanduser().resolve()
        self.image_root = image_root.expanduser().resolve() if image_root else None
        self.examples = self._load_examples(max_samples)
        if not self.examples:
            raise ValueError(f"No SFT examples loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SftExample:
        return self.examples[index]

    def _load_examples(self, max_samples: int) -> list[SftExample]:
        if not self.path.exists():
            raise FileNotFoundError(f"SFT data does not exist: {self.path}")

        examples: list[SftExample] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                example = self._parse_record(json.loads(line), line_number)
                if example is not None:
                    examples.append(example)
                if max_samples > 0 and len(examples) >= max_samples:
                    break
        return examples

    def _parse_record(self, record: dict[str, Any], line_number: int) -> SftExample | None:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Line {line_number}: expected user and assistant messages.")

        user_message = messages[0]
        assistant_message = messages[-1]
        if not isinstance(user_message, dict) or user_message.get("role") != "user":
            raise ValueError(f"Line {line_number}: first message must be user.")
        if not isinstance(assistant_message, dict) or assistant_message.get("role") != "assistant":
            raise ValueError(f"Line {line_number}: last message must be assistant.")

        user_text, raw_image_path = extract_user_text_and_image(user_message)
        assistant_text = assistant_message.get("content")
        if not user_text.strip() or not isinstance(assistant_text, str) or not assistant_text.strip():
            return None

        return SftExample(
            user_text=user_text.strip(),
            assistant_text=assistant_text.strip(),
            image_path=self._resolve_image_path(raw_image_path, line_number),
            line_number=line_number,
        )

    def _resolve_image_path(self, raw_path: str | None, line_number: int) -> Path | None:
        if raw_path is None:
            return None

        raw = Path(raw_path).expanduser()
        candidates = [raw]
        if self.image_root is not None:
            candidates.append(self.image_root / raw.name)
            if not raw.is_absolute():
                candidates.append(self.image_root / raw)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        raise FileNotFoundError(
            f"Line {line_number}: image path does not exist: {raw_path}. "
            "Use --image-root if the dataset was generated on another machine."
        )


def extract_user_text_and_image(user_message: dict[str, Any]) -> tuple[str, str | None]:
    content = user_message.get("content")
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return "", None

    texts: list[str] = []
    image_path: str | None = None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
        if item.get("type") == "image" and isinstance(item.get("image"), str):
            image_path = image_path or item["image"]
    return "\n".join(texts), image_path


class ReactCollator:
    def __init__(self, processor: Any, tokenizer: Any, model: Any, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.internvl_model = unwrap_model(model)
        self.internvl_template = getattr(self.internvl_model, "conv_template", None)
        self.internvl_num_image_token = getattr(self.internvl_model, "num_image_token", None)
        self.internvl_image_size = internvl_image_size(self.internvl_model)
        self.internvl_pixel_dtype = internvl_pixel_dtype(self.internvl_model)

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None)
        self._set_internvl_image_context_token()

    def __call__(self, examples: list[SftExample]) -> dict[str, Any]:
        if self.uses_internvl:
            return self._collate_internvl(examples)
        return self._collate_with_processor(examples)

    @property
    def uses_internvl(self) -> bool:
        return self.internvl_template is not None and self.internvl_num_image_token is not None

    def _collate_with_processor(self, examples: list[SftExample]) -> dict[str, Any]:
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        images: list[Any] = []

        for example in examples:
            image = load_pil_image(example.image_path)
            prompt_text, full_text = self._format_processor_text(example)
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            images.append(image)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            prompt_len = self._processor_length(prompt_text, image)
            labels[row, : min(prompt_len, labels.shape[1])] = IGNORE_INDEX
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"].eq(0), IGNORE_INDEX)
        batch["labels"] = labels
        return batch

    def _collate_internvl(self, examples: list[SftExample]) -> dict[str, Any]:
        import torch

        prompt_texts: list[str] = []
        full_texts: list[str] = []
        pixel_values: list[Any] = []

        for example in examples:
            if example.image_path is None:
                raise ValueError(f"Line {example.line_number}: InternVL training requires an image.")
            image_tensor = self._load_internvl_image_tensor(example.image_path)
            prompt_text, full_text = self._format_internvl_text(example, image_tensor.shape[0])
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            pixel_values.append(image_tensor)

        batch = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row, prompt_text in enumerate(prompt_texts):
            prompt_len = self._tokenized_length(prompt_text)
            labels[row, : min(prompt_len, labels.shape[1])] = IGNORE_INDEX
        if "attention_mask" in batch:
            labels = labels.masked_fill(batch["attention_mask"].eq(0), IGNORE_INDEX)

        batch["labels"] = labels
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        batch["image_flags"] = torch.ones(batch["pixel_values"].shape[0], 1, dtype=torch.long)
        return batch

    def _format_processor_text(self, example: SftExample) -> tuple[str, str]:
        user_message = {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": example.user_text}],
        }
        assistant_message = {"role": "assistant", "content": example.assistant_text}

        if hasattr(self.processor, "apply_chat_template"):
            prompt = self.processor.apply_chat_template(
                [user_message],
                tokenize=False,
                add_generation_prompt=True,
            )
            full = self.processor.apply_chat_template(
                [user_message, assistant_message],
                tokenize=False,
                add_generation_prompt=False,
            )
            return prompt, full

        prompt = f"<image>\n{example.user_text}\nAssistant:"
        return prompt, f"{prompt} {example.assistant_text}"

    def _format_internvl_text(self, example: SftExample, num_patches: int) -> tuple[str, str]:
        question = example.user_text
        if "<image>" not in question:
            question = f"<image>\n{question}"
        return (
            self._build_internvl_query(question, None, num_patches),
            self._build_internvl_query(question, example.assistant_text, num_patches),
        )

    def _build_internvl_query(
        self,
        question: str,
        assistant_text: str | None,
        num_patches: int,
    ) -> str:
        import copy

        template = copy.deepcopy(self.internvl_template)
        system_message = getattr(self.internvl_model, "system_message", None)
        if system_message is not None:
            template.system_message = system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], assistant_text)

        image_tokens = "<img>" + "<IMG_CONTEXT>" * int(self.internvl_num_image_token) * num_patches + "</img>"
        return template.get_prompt().replace("<image>", image_tokens, 1)

    def _processor_length(self, text: str, image: Any) -> int:
        inputs = self.processor(
            text=[text],
            images=[image],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return attention_length(inputs)

    def _tokenized_length(self, text: str) -> int:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return attention_length(inputs)

    def _load_internvl_image_tensor(self, image_path: Path) -> Any:
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        image = Image.open(image_path).convert("RGB")
        tiles = dynamic_preprocess_internvl_image(
            image=image,
            image_size=self.internvl_image_size,
            max_tiles=int(os.getenv("MM_REACT_TRANSFORMERS_MAX_IMAGE_TILES", "12")),
        )
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.internvl_image_size, self.internvl_image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        tensor = torch.stack([transform(tile) for tile in tiles])
        if self.internvl_pixel_dtype is not None:
            tensor = tensor.to(dtype=self.internvl_pixel_dtype)
        return tensor

    def _set_internvl_image_context_token(self) -> None:
        if not hasattr(self.internvl_model, "img_context_token_id"):
            return
        token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if token_id is None or token_id == getattr(self.tokenizer, "unk_token_id", None):
            raise ValueError("Tokenizer does not contain the InternVL <IMG_CONTEXT> token.")
        self.internvl_model.img_context_token_id = token_id


def attention_length(inputs: dict[str, Any]) -> int:
    if "attention_mask" in inputs:
        return int(inputs["attention_mask"][0].sum().item())
    return int(inputs["input_ids"].shape[-1])


def load_pil_image(path: Path | None) -> Any:
    if path is None:
        return None
    from PIL import Image

    return Image.open(path).convert("RGB")


def unwrap_model(model: Any) -> Any:
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        model = get_base_model()
    for attr in ("model", "base_model"):
        nested = getattr(model, attr, None)
        if nested is not None and hasattr(nested, "num_image_token"):
            return nested
    return model


def internvl_image_size(model: Any) -> int:
    env_value = os.getenv("MM_REACT_TRANSFORMERS_IMAGE_SIZE")
    if env_value:
        return int(env_value)
    config = getattr(model, "config", None)
    for source in (config, getattr(config, "vision_config", None)):
        image_size = getattr(source, "force_image_size", None) or getattr(source, "image_size", None)
        if image_size:
            return int(image_size)
    return 448


def internvl_pixel_dtype(model: Any) -> Any | None:
    parameters = getattr(getattr(model, "vision_model", model), "parameters", None)
    if not callable(parameters):
        return None
    for parameter in parameters():
        if getattr(parameter, "is_floating_point", lambda: False)():
            return parameter.dtype
    return None


def dynamic_preprocess_internvl_image(image: Any, image_size: int, max_tiles: int) -> list[Any]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = {
        (cols, rows)
        for n in range(1, max_tiles + 1)
        for cols in range(1, n + 1)
        for rows in range(1, n + 1)
        if 1 <= cols * rows <= max_tiles
    }
    cols, rows = min(
        target_ratios,
        key=lambda ratio: (abs(aspect_ratio - ratio[0] / ratio[1]), ratio[0] * ratio[1]),
    )

    resized = image.resize((cols * image_size, rows * image_size))
    tiles = [
        resized.crop(
            (
                col * image_size,
                row * image_size,
                (col + 1) * image_size,
                (row + 1) * image_size,
            )
        )
        for row in range(rows)
        for col in range(cols)
    ]
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_processor_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, AutoTokenizer

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=False,
        )
    return processor, tokenizer


def load_model(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else "auto"
    kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map
    if args.load_in_4bit:
        quant_kwargs: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if dtype != "auto":
            quant_kwargs["bnb_4bit_compute_dtype"] = dtype
        kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)
    if args.load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    loader_map = {
        "vision2seq": [AutoModelForImageTextToText],
        "causal-lm": [AutoModelForCausalLM],
        "auto-model": [AutoModel],
        "auto": [AutoModelForImageTextToText, AutoModelForCausalLM, AutoModel],
    }
    last_error: Exception | None = None
    for loader in loader_map[args.model_loader]:
        try:
            print(f"Loading model with {loader.__name__}: {args.model_path}")
            return loader.from_pretrained(args.model_path, **kwargs)
        except Exception as exc:
            last_error = exc
            if args.model_loader != "auto":
                raise
            print(f"[warn] {loader.__name__} failed: {exc!r}")
    raise RuntimeError(f"Unable to load model from {args.model_path}") from last_error


def prepare_lora_model(args: argparse.Namespace, model: Any) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    target_modules: str | list[str] = args.lora_target_modules
    if target_modules != "all-linear":
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]

    lora_kwargs: dict[str, Any] = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": target_modules,
    }
    if not hasattr(unwrap_model(model), "conv_template"):
        lora_kwargs["task_type"] = "CAUSAL_LM"

    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def training_args(args: argparse.Namespace, has_eval: bool) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "tf32": args.tf32,
        "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments).parameters
    eval_strategy_name = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs[eval_strategy_name] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = args.eval_steps
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()

    import torch
    from transformers import Trainer

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load-in-4bit or --load-in-8bit.")
    if args.tf32 and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = ReactSftDataset(
        args.train_data,
        image_root=args.image_root,
        max_samples=args.max_train_samples,
    )
    eval_dataset = (
        ReactSftDataset(args.eval_data, image_root=args.image_root, max_samples=args.max_eval_samples)
        if args.eval_data
        else None
    )

    processor, tokenizer = load_processor_and_tokenizer(args)
    base_model = load_model(args)
    model = prepare_lora_model(args, base_model)
    collator = ReactCollator(
        processor=processor,
        tokenizer=tokenizer,
        model=base_model,
        max_length=args.max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args(args, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("MM-ReAct LoRA SFT")
    print(f"model_path: {args.model_path}")
    print(f"train_examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval_examples: {len(eval_dataset)}")
    print(f"output_dir: {args.output_dir}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Saved LoRA adapter and processor/tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
