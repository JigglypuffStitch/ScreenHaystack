#!/usr/bin/env python3
"""Evaluate UI-Venus-Ground on all ScreenSpot-Pro samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
from ast import literal_eval
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


PROMPT_TEMPLATE = 'Outline the position corresponding to the instruction: {}. The output should be only [x1,y1,x2,y2].'
BOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
PATCH_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UI-Venus-Ground on ScreenSpot-Pro.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--screenspot_imgs", type=Path, required=True)
    parser.add_argument("--screenspot_test", type=Path, required=True)
    parser.add_argument("--task", default="all", help="Annotation basename, comma-separated names, or all.")
    parser.add_argument("--log_path", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu.")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=2_000_000)
    parser.add_argument("--max_pixels", type=int, default=4_800_000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Continue a matching JSONL run.")
    mode.add_argument("--overwrite", "--no_resume", action="store_true", help="Replace an existing JSONL run.")
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max_samples must be -1 or a positive integer")
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    if args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        parser.error("pixel limits must satisfy 1 <= min_pixels <= max_pixels")
    return args


def load_samples(annotation_dir: Path, task: str, max_samples: int) -> list[dict[str, Any]]:
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")

    if task == "all":
        paths = sorted(annotation_dir.glob("*.json"))
    else:
        names = [name.strip().removesuffix(".json") for name in task.split(",")]
        if not names or any(not name or Path(name).name != name for name in names):
            raise ValueError("--task must contain annotation basenames")
        paths = [annotation_dir / f"{name}.json" for name in names]
    if not paths:
        raise FileNotFoundError(f"No annotation JSON files found in: {annotation_dir}")

    samples: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            rows = json.load(file)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of samples in {path}")
        for local_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Expected a sample object in {path} at index {local_index}")
            if not row.get("instruction") or not row.get("img_filename"):
                raise ValueError(f"Missing instruction or img_filename: {path}:{local_index}")

            key = f"{path.stem}:{row.get('id', local_index)}"
            if key in seen_keys:
                raise ValueError(f"Duplicate sample key: {key}")
            seen_keys.add(key)
            samples.append({**row, "_key": key, "_task_filename": path.stem})
            if max_samples > 0 and len(samples) >= max_samples:
                return samples
    return samples


def run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name_or_path": args.model_name_or_path,
        "screenspot_imgs": str(args.screenspot_imgs.resolve()),
        "screenspot_test": str(args.screenspot_test.resolve()),
        "task": args.task,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "prompt_template": PROMPT_TEMPLATE,
        "scoring": "bbox_if_available",
    }


def run_id_for(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_resume_results(path: Path, run_id: str, expected_keys: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_id") != run_id:
                raise ValueError(f"Log configuration differs at {path}:{line_number}; use a new log path")
            key = record.get("resume_key")
            if key not in expected_keys or key in results:
                raise ValueError(f"Unknown or duplicate sample key at {path}:{line_number}: {key!r}")
            results[key] = record
    return results


def ensure_trailing_newline(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as file:
        file.seek(-1, 2)
        if file.read(1) != b"\n":
            file.write(b"\n")


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {args.device}")
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device.startswith("cuda")
    logging.info("Loading model: %s", args.model_name_or_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device == "auto" else None,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if use_cuda else "sdpa",
    ).eval()
    if args.device != "auto":
        model = model.to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    return model, processor


def parse_bbox(text: str) -> list[float] | None:
    try:
        value = literal_eval(text.strip())
        if isinstance(value, (list, tuple)) and len(value) == 4:
            box = [float(number) for number in value]
            return box if all(math.isfinite(number) for number in box) else None
    except (SyntaxError, ValueError, TypeError, OverflowError, MemoryError):
        pass
    match = BOX_RE.search(text)
    if match is None:
        return None
    box = [float(number) for number in match.groups()]
    return box if all(math.isfinite(number) for number in box) else None


def predict(model: Any, processor: Any, image_path: Path, instruction: str, args: argparse.Namespace) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    if instruction.endswith("."):
        instruction = instruction[:-1]
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(image_path),
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            },
            {"type": "text", "text": PROMPT_TEMPLATE.format(instruction)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0
        )
    new_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    raw_response = decoded[0] if decoded else ""
    raw_box = parse_bbox(raw_response)
    if raw_box is None:
        return {"raw_response": raw_response, "raw_box": None, "bbox": None, "point": None, "input_size": None}

    grid = inputs["image_grid_thw"][0]
    input_height = float(grid[1].item() * PATCH_SIZE)
    input_width = float(grid[2].item() * PATCH_SIZE)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"Invalid model input size for {image_path}: {input_width}x{input_height}")
    bbox = [
        raw_box[0] / input_width,
        raw_box[1] / input_height,
        raw_box[2] / input_width,
        raw_box[3] / input_height,
    ]
    point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return {
        "raw_response": raw_response,
        "raw_box": raw_box,
        "bbox": bbox,
        "point": point,
        "input_size": [input_width, input_height],
    }


def score(bbox: Any, image_size: tuple[int, int], point: list[float] | None) -> str:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "unscored"
    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError):
        return "unscored"
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
        return "unscored"
    if point is None:
        return "wrong_format"
    width, height = image_size
    return "correct" if x1 / width <= point[0] <= x2 / width and y1 / height <= point[1] <= y2 / height else "wrong"


def metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    scored_rows = [row for row in results if row["correctness"] != "unscored"]
    scored = len(scored_rows)
    correct = sum(row["correctness"] == "correct" for row in results)
    wrong_format = sum(row["correctness"] == "wrong_format" for row in results)
    text_rows = [row for row in scored_rows if row.get("ui_type") == "text"]
    icon_rows = [row for row in scored_rows if row.get("ui_type") == "icon"]

    def accuracy(rows: list[dict[str, Any]]) -> float:
        return sum(row["correctness"] == "correct" for row in rows) / len(rows) if rows else 0.0

    return {
        "num_total": total,
        "num_scored": scored,
        "num_unscored": total - scored,
        "num_correct_action": correct,
        "wrong_format_num": wrong_format,
        "action_acc": correct / scored if scored else 0.0,
        "text_acc": accuracy(text_rows),
        "icon_acc": accuracy(icon_rows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    samples = load_samples(args.screenspot_test, args.task, args.max_samples)
    if not samples:
        raise ValueError("No samples found")
    if not args.screenspot_imgs.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.screenspot_imgs}")

    config = run_config(args)
    run_id = run_id_for(config)
    expected_keys = {sample["_key"] for sample in samples}
    if args.resume:
        results = load_resume_results(args.log_path, run_id, expected_keys)
    else:
        if args.log_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {args.log_path}; use --resume or --overwrite")
        results = {}

    remaining = len(samples) - len(results)
    logging.info("Samples: %d; remaining: %d", len(samples), remaining)
    if remaining:
        model, processor = load_model(args)

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        ensure_trailing_newline(args.log_path)
    mode = "a" if args.resume else "w"
    with args.log_path.open(mode, encoding="utf-8") as output:
        for idx, sample in enumerate(tqdm(samples, desc="ScreenSpot-Pro")):
            key = sample["_key"]
            if key in results:
                continue

            image_path = args.screenspot_imgs / str(sample["img_filename"])
            with Image.open(image_path) as image:
                width, height = image.size
            response = predict(model, processor, image_path, str(sample["instruction"]), args)
            point = response["point"]
            pred_pixel = [point[0] * width, point[1] * height] if point is not None else None
            result = {
                "idx": idx,
                "key": str(sample.get("id", idx)),
                "resume_key": key,
                "run_id": run_id,
                "img_path": str(image_path),
                "platform": sample.get("platform"),
                "application": sample.get("application"),
                "group": sample.get("group"),
                "lang": "en",
                "instruction_style": "instruction",
                "ui_type": sample.get("ui_type"),
                "task_filename": sample["_task_filename"],
                "prompt_to_evaluate": sample["instruction"],
                "bbox_orig": sample.get("bbox"),
                "bbox": sample.get("bbox"),
                "img_size": [width, height],
                "raw_response": response["raw_response"],
                "pred_pixel": pred_pixel,
                "pred": pred_pixel,
                "pred_bbox_norm01": response["bbox"],
                "pred_raw_box": response["raw_box"],
                "model_input_size": response["input_size"],
                "correctness": score(sample.get("bbox"), (width, height), point),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            results[key] = result

    ordered_results = [results[sample["_key"]] for sample in samples]
    overall = metrics(ordered_results)
    summary_path = Path(f"{args.log_path}.summary.json")
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump({"config": config, "metrics": {"overall": overall}}, output, indent=2, ensure_ascii=False)
    logging.info(
        "Accuracy: %.4f (%d/%d scored; %d total)",
        overall["action_acc"],
        overall["num_correct_action"],
        overall["num_scored"],
        overall["num_total"],
    )
    logging.info("Results: %s; summary: %s", args.log_path, summary_path)


if __name__ == "__main__":
    main()
