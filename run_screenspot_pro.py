#!/usr/bin/env python3
"""Run a Qwen2.5-VL or Qwen3-VL model on ScreenSpot-Pro."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor


COORDINATE_RE = re.compile(
    r"\(\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)

SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL or Qwen3-VL on ScreenSpot-Pro."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--architecture",
        choices=("auto", "qwen2_5_vl", "qwen3_vl"),
        default="auto",
        help="Model architecture. 'auto' reads model_type from the checkpoint.",
    )
    parser.add_argument(
        "--coordinate_mode",
        choices=("absolute", "relative"),
        default="absolute",
        help="Use pixel coordinates or relative coordinates on a 0-1000 scale.",
    )
    parser.add_argument("--annotation_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "cn"), default="en")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default=None,
    )
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_pixels", type=int, default=4096 * 2160)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()


def load_annotations(annotation_dir: Path) -> list[dict[str, Any]]:
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    samples: list[dict[str, Any]] = []
    paths = sorted(annotation_dir.glob("*.json")) + sorted(annotation_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSON or JSONL files found in: {annotation_dir}")

    for path in paths:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as file:
                rows = [json.loads(line) for line in file if line.strip()]
        else:
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
            rows = value if isinstance(value, list) else [value]

        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"Expected objects in {path}, got {type(row).__name__}")
            sample = dict(row)
            sample["_annotation_file"] = path.name
            samples.append(sample)

    return samples


def get_instruction(sample: dict[str, Any], language: str) -> str:
    keys = ("instruction_cn", "prompt_to_evaluate_cn") if language == "cn" else ()
    keys += ("instruction", "prompt_to_evaluate", "query", "prompt", "text")
    for key in keys:
        value = sample.get(key)
        if value:
            return str(value)
    raise ValueError(f"Sample {sample.get('id', '<unknown>')} has no instruction")


def get_image_path(sample: dict[str, Any], image_dir: Path) -> Path:
    value = sample.get("img_filename") or sample.get("img_path")
    if not value:
        raise ValueError(f"Sample {sample.get('id', '<unknown>')} has no image path")
    path = Path(str(value))
    return path if path.is_absolute() else image_dir / path


def get_bbox(sample: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    bbox = sample.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def parse_prediction(
    response: str,
    coordinate_mode: str,
    width: int,
    height: int,
) -> tuple[list[float] | None, list[float] | None]:
    match = COORDINATE_RE.search(response)
    if match is None:
        return None, None

    x, y = float(match.group(1)), float(match.group(2))
    if coordinate_mode == "relative":
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            return None, None
        pixel_x = min(x / 1000.0 * width, width - 1.0)
        pixel_y = min(y / 1000.0 * height, height - 1.0)
    else:
        if not (0.0 <= x < width and 0.0 <= y < height):
            return None, None
        pixel_x, pixel_y = x, y

    return [x, y], [pixel_x, pixel_y]


def point_in_bbox(point: list[float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def resolve_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str]:
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    detected_architecture = str(getattr(config, "model_type", ""))
    architecture = detected_architecture if args.architecture == "auto" else args.architecture

    if args.architecture != "auto" and architecture != detected_architecture:
        raise ValueError(
            f"--architecture={architecture} does not match checkpoint "
            f"model_type={detected_architecture!r}"
        )

    if architecture == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_class = Qwen2_5_VLForConditionalGeneration
    elif architecture == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL is unavailable. Install a transformers version that supports Qwen3-VL."
            ) from error
        model_class = Qwen3VLForConditionalGeneration
    else:
        raise ValueError(
            f"Unsupported model_type={detected_architecture!r}; expected qwen2_5_vl or qwen3_vl"
        )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "device_map": {"": args.device},
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_class.from_pretrained(args.model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor, architecture


def run_one(
    model: Any,
    processor: Any,
    image: Image.Image,
    instruction: str,
    device: str,
    max_new_tokens: int,
) -> str:
    width, height = image.size
    user_prompt = SYSTEM_PROMPT.format(height=height, width=width).strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt + "\n\n"},
                {"type": "image"},
                {"type": "text", "text": "\n" + instruction},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    tokenizer = getattr(processor, "tokenizer", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated = generated[:, prompt_length:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable, but --device={args.device}")

    samples = load_annotations(args.annotation_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    print(f"Loading {args.model_path}")
    model, processor, architecture = load_model(args)
    print(f"Architecture: {architecture}")
    print(f"Coordinate mode: {args.coordinate_mode}")
    print(f"Samples: {len(samples)}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    scored = 0
    correct = 0
    parse_failures = 0

    with args.output_path.open("w", encoding="utf-8") as output_file:
        for index, sample in enumerate(tqdm(samples, desc="ScreenSpot-Pro")):
            sample_id = str(sample.get("id", index))
            image_path = get_image_path(sample, args.image_dir)
            instruction = get_instruction(sample, args.language)
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")

            width, height = image.size
            bbox = get_bbox(sample, width, height)
            try:
                response = run_one(
                    model=model,
                    processor=processor,
                    image=image,
                    instruction=instruction,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as error:
                raise RuntimeError(f"Inference failed for sample {sample_id}") from error

            output_coordinate, pixel_coordinate = parse_prediction(
                response=response,
                coordinate_mode=args.coordinate_mode,
                width=width,
                height=height,
            )
            is_correct = (
                pixel_coordinate is not None and point_in_bbox(pixel_coordinate, bbox)
                if bbox is not None
                else None
            )
            if output_coordinate is None:
                parse_failures += 1
            if bbox is not None:
                scored += 1
            if is_correct:
                correct += 1
            total += 1

            result = {
                "id": sample_id,
                "annotation_file": sample["_annotation_file"],
                "image_path": str(image_path),
                "image_size": [width, height],
                "instruction": instruction,
                "bbox": list(bbox) if bbox is not None else None,
                "architecture": architecture,
                "coordinate_mode": args.coordinate_mode,
                "raw_response": response,
                "predicted_coordinate": output_coordinate,
                "predicted_pixel": pixel_coordinate,
                "correct": is_correct,
            }

            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()

    accuracy = correct / scored if scored else 0.0
    summary = {
        "model_path": args.model_path,
        "architecture": architecture,
        "coordinate_mode": args.coordinate_mode,
        "total": total,
        "scored": scored,
        "unscored": total - scored,
        "correct": correct,
        "parse_failures": parse_failures,
        "accuracy": accuracy,
    }
    summary_path = Path(f"{args.output_path}.summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Predictions: {args.output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
