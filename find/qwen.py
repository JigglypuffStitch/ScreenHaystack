from __future__ import annotations

from common import (
    dist_init,
    rank0_print,
    dist_barrier,
    cleanup_dist,
    _get_pad_token_id,
    compute_icon_placement,
    record_to_accuracy,
    load_existing_jsonl_progress,
    ensure_jsonl_append_boundary,
)

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import smart_resize


# ========== Default configuration ==========
DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'background')
DEFAULT_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/qwen')

MIN_PIXELS = 3136
MAX_PIXELS = 4096 * 2160
MODEL_DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = True
DEFAULT_ICON_BLEND_ALPHA = 0.82

SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description,
provide the coordinates of the specified element as a single (x,y) point.
The image resolution is height {height} and width {width}.
For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

_COORD_RE = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")


def get_icons(icon_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        "star": {
            "path": os.path.join(icon_dir, "star_icon_40.png"),
            "width": 40,
            "height": 40,
            "prompt": "a red five-pointed star shape",
            "name": "star",
        },
        "circle_ok": {
            "path": os.path.join(icon_dir, "circle_ok_40.png"),
            "width": 40,
            "height": 40,
            "prompt": "the OK icon, a red circle with white text 'OK'",
            "name": "circle_ok",
        },
        "clock": {
            "path": os.path.join(icon_dir, "clock_text_60x40.png"),
            "width": 60,
            "height": 40,
            "prompt": "the clock icon, a white circle with black hour hand pointing to 3 o'clock and minute hand pointing to 12 o'clock, with text 'clock' on the right",
            "name": "clock",
        },
    }


def extract_coordinates(raw_string: str) -> Optional[Tuple[float, float]]:
    try:
        m = _COORD_RE.search(raw_string or "")
        if m:
            return float(m.group(1)), float(m.group(2))

        nums = re.findall(r"-?\d*\.?\d+", raw_string or "")
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])

        return None
    except Exception:
        return None


def make_prompt_text(processor: AutoProcessor, resized_width: int, resized_height: int, instruction: str) -> str:
    prompt = SYSTEM_PROMPT.format(height=resized_height, width=resized_width).strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt + "\n\n"},
                {"type": "image"},
                {"type": "text", "text": "\nElement description: " + instruction},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def composite_icon_alpha_blend(
    base_bg: Image.Image,
    icon: Image.Image,
    px: int,
    py: int,
    blend_alpha: float,
) -> Image.Image:
    bg_rgba = base_bg.convert("RGBA")
    icon_rgba = icon.convert("RGBA")
    if blend_alpha < 1.0:
        r, g, b, a = icon_rgba.split()
        a = a.point(lambda v: int(v * blend_alpha))
        icon_rgba = Image.merge("RGBA", (r, g, b, a))

    icon_w, icon_h = icon_rgba.size
    region_box = (px, py, px + icon_w, py + icon_h)
    bg_region = bg_rgba.crop(region_box)
    blended_region = Image.alpha_composite(bg_region, icon_rgba)
    bg_rgba.paste(blended_region, (px, py))
    return bg_rgba.convert("RGB")


def record_matches_placement(
    record: Dict[str, Any],
    px: int,
    py: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    icon_blend_alpha: float,
) -> bool:
    record_blend_alpha = record.get("icon_blend_alpha")
    if record_blend_alpha is None:
        return False
    try:
        if abs(float(record_blend_alpha) - icon_blend_alpha) > 1e-6:
            return False
    except (TypeError, ValueError):
        return False

    placement = record.get("icon_placement")
    if isinstance(placement, dict):
        return (
            placement.get("px") == px
            and placement.get("py") == py
            and placement.get("x_min") == x_min
            and placement.get("x_max") == x_max
            and placement.get("y_min") == y_min
            and placement.get("y_max") == y_max
        )

    placement_xy = record.get("placement_xy_original")
    bbox = record.get("bbox_original")
    if isinstance(placement_xy, list) and len(placement_xy) == 2 and isinstance(bbox, list) and len(bbox) == 4:
        return placement_xy == [px, py] and bbox == [x_min, y_min, x_max, y_max]

    return False


def get_qwen_jsonl_paths(output_dir: str, icon_name: str, bg_index: int, rank: int) -> Tuple[str, str]:
    merged_jsonl_path = os.path.join(output_dir, f"{icon_name}_qwen_outputs_{bg_index}.jsonl")
    rank_jsonl_path = os.path.join(output_dir, f"{icon_name}_qwen_outputs_{bg_index}.rank{rank}.jsonl")
    return merged_jsonl_path, rank_jsonl_path


def load_merged_qwen_progress(
    *,
    output_dir: str,
    rows: int,
    cols: int,
    bg_name: str,
    bg_index: int,
    icon_name: str,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    merged_jsonl_path = os.path.join(output_dir, f"{icon_name}_qwen_outputs_{bg_index}.jsonl")

    return load_existing_jsonl_progress(
        jsonl_path=merged_jsonl_path,
        rows=rows,
        cols=cols,
        bg_name=bg_name,
        bg_index=bg_index,
        icon_name=icon_name,
        clean_malformed=False,
    )


def load_rank_qwen_progress(
    *,
    output_dir: str,
    rows: int,
    cols: int,
    bg_name: str,
    bg_index: int,
    icon_name: str,
    rank: int,
    clean_malformed: bool = True,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    _, rank_jsonl_path = get_qwen_jsonl_paths(output_dir, icon_name, bg_index, rank)
    return load_existing_jsonl_progress(
        jsonl_path=rank_jsonl_path,
        rows=rows,
        cols=cols,
        bg_name=bg_name,
        bg_index=bg_index,
        icon_name=icon_name,
        clean_malformed=clean_malformed,
    )


def count_missing_cells_for_rank(
    *,
    bg_path: str,
    bg_index: int,
    icon_info: Dict[str, Any],
    output_dir: str,
    rank: int,
    world_size: int,
    icon_blend_alpha: float,
) -> Tuple[int, int]:
    base_bg = Image.open(bg_path).convert("RGB")
    width, height = base_bg.size

    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])
    cols = math.ceil(width / icon_w)
    rows = height // icon_h
    if rows == 0:
        return 0, 0

    grid_w = width / cols
    grid_h = height / rows
    total_cells = rows * cols
    my_cell_indices = [i for i in range(total_cells) if (i % world_size) == rank]
    existing_records = load_merged_qwen_progress(
        output_dir=output_dir,
        rows=rows,
        cols=cols,
        bg_name=os.path.basename(bg_path),
        bg_index=bg_index,
        icon_name=icon_info["name"],
    )

    completed = 0
    for cell_idx in my_cell_indices:
        r = cell_idx // cols
        c = cell_idx % cols
        record = existing_records.get((r, c))
        if record is None:
            continue
        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, width, height
        )
        if record_matches_placement(record, px, py, x_min, x_max, y_min, y_max, icon_blend_alpha):
            completed += 1

    return len(my_cell_indices) - completed, len(my_cell_indices)


def load_model(
    *,
    model_path: str,
    device: torch.device,
    rank: int,
    min_pixels: int,
    max_pixels: int,
    attn_impl: str,
) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    model_kwargs = dict(
        dtype=MODEL_DTYPE,
        device_map={"": str(device)},
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    rank0_print(rank, f"[Rank {rank}] Loading Qwen3-VL on {device} from {model_path} ...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model.eval()
    rank0_print(rank, f"[Rank {rank}] Model loaded.")
    return model, processor


def run_batch(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    device: torch.device,
    batch_items: List[Dict[str, Any]],
    prompt_text: str,
    pad_token_id: int,
    max_new_tokens: int,
    original_width: int,
    original_height: int,
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    if not batch_items:
        return []

    texts = [prompt_text] * len(batch_items)
    images = [it["image"] for it in batch_items]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        out_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        errors: List[Optional[str]] = [None] * len(batch_items)
    except Exception as exc:
        out_texts = [""] * len(batch_items)
        errors = [str(exc)] * len(batch_items)

    results: List[Dict[str, Any]] = []
    for it, raw, error_msg in zip(batch_items, out_texts, errors):
        coords = extract_coordinates(raw)
        parsed_raw_x = None
        parsed_raw_y = None
        pred = None
        hit = 0

        if error_msg is None and coords is not None:
            raw_x, raw_y = coords
            parsed_raw_x, parsed_raw_y = raw_x, raw_y
            if raw_x <= 1005 and raw_y <= 1005:
                pred_x = raw_x * (original_width / 1000.0)
                pred_y = raw_y * (original_height / 1000.0)
            else:
                pred_x = raw_x * scale_x
                pred_y = raw_y * scale_y
            pred = (pred_x, pred_y)

            x_min, y_min, x_max, y_max = it["bbox"]
            if x_min <= pred_x <= x_max and y_min <= pred_y <= y_max:
                hit = 1
        elif error_msg is None:
            error_msg = f"Cannot parse coordinate from output: {raw}"

        results.append(
            {
                "cell_idx": it["cell_idx"],
                "r": it["r"],
                "c": it["c"],
                "px": it["px"],
                "py": it["py"],
                "bbox": it["bbox"],
                "hit": hit,
                "raw": raw,
                "parsed_raw_x": parsed_raw_x,
                "parsed_raw_y": parsed_raw_y,
                "pred": pred,
                "error": error_msg,
            }
        )

    return results


def run_one_background(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    device: torch.device,
    rank: int,
    world_size: int,
    dist_enabled: bool,
    bg_path: str,
    bg_index: int,
    icon_info: Dict[str, Any],
    output_dir: str,
    batch_size: int,
    max_new_tokens: int,
    min_pixels: int,
    max_pixels: int,
    icon_blend_alpha: float,
) -> None:
    base_bg = Image.open(bg_path).convert("RGB")
    width, height = base_bg.size

    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])
    cols = math.ceil(width / icon_w)
    rows = height // icon_h
    if rows == 0:
        rank0_print(rank, "⚠️ The background is shorter than the icon; cannot place the icon, skipping")
        return

    grid_w = width / cols
    grid_h = height / rows

    image_processor = processor.image_processor
    patch_size = getattr(image_processor, "patch_size", 14)
    merge_size = getattr(image_processor, "merge_size", 2)
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    scale_x = width / resized_width
    scale_y = height / resized_height

    prompt_text = make_prompt_text(
        processor=processor,
        resized_width=resized_width,
        resized_height=resized_height,
        instruction=icon_info["prompt"],
    )

    icon = Image.open(icon_info["path"]).convert("RGBA").resize((icon_w, icon_h))
    pad_token_id = _get_pad_token_id(processor)

    rank0_print(rank, f"\n🖼️ Background [{bg_index}]: {os.path.basename(bg_path)} ({width}x{height})")
    rank0_print(rank, f"📐 Grid: {cols} columns x {rows} rows; icon size {icon_w}x{icon_h}px")
    rank0_print(rank, f"🔁 Qwen3-VL input size: {resized_width}x{resized_height}")
    rank0_print(rank, f"🔁 Coordinate scale: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    rank0_print(rank, f"🚀 batch inference: batch_size={batch_size}, world_size={world_size}")

    total_cells = rows * cols
    my_cell_indices = [i for i in range(total_cells) if (i % world_size) == rank]
    rank_matrix = np.full((rows, cols), np.nan, dtype=np.float32)

    _, rank_jsonl_path = get_qwen_jsonl_paths(output_dir, icon_info["name"], bg_index, rank)
    existing_records = load_merged_qwen_progress(
        output_dir=output_dir,
        rows=rows,
        cols=cols,
        bg_name=os.path.basename(bg_path),
        bg_index=bg_index,
        icon_name=icon_info["name"],
    )

    valid_existing_records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    stale_records = 0
    for cell_idx in my_cell_indices:
        r = cell_idx // cols
        c = cell_idx % cols
        record = existing_records.get((r, c))
        if record is None:
            continue
        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, width, height
        )
        if record_matches_placement(record, px, py, x_min, x_max, y_min, y_max, icon_blend_alpha):
            valid_existing_records[(r, c)] = record
            rank_matrix[r, c] = record_to_accuracy(record)
        else:
            stale_records += 1
    existing_records = valid_existing_records

    if existing_records:
        print(f"[Rank {rank}] Automatic resume: loaded {len(existing_records)} completed grid cells; these cells will be skipped")
    if stale_records:
        print(f"[Rank {rank}] Found {stale_records} stale records with mismatched placements; recomputing them")
    ensure_jsonl_append_boundary(rank_jsonl_path)

    start_time = time.time()
    batch_items: List[Dict[str, Any]] = []
    new_cells_done = 0
    pbar = tqdm(my_cell_indices, desc=f"BG{bg_index}-rank{rank}", disable=(rank != 0))

    def flush_batch() -> None:
        nonlocal batch_items, new_cells_done
        if not batch_items:
            return

        results = run_batch(
            model=model,
            processor=processor,
            device=device,
            batch_items=batch_items,
            prompt_text=prompt_text,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            original_width=width,
            original_height=height,
            scale_x=scale_x,
            scale_y=scale_y,
        )

        jsonl_rows: List[Dict[str, Any]] = []
        for res in results:
            rank_matrix[res["r"], res["c"]] = float(res["hit"])
            x_min, y_min, x_max, y_max = res["bbox"]
            pred_x = res["pred"][0] if res["pred"] is not None else None
            pred_y = res["pred"][1] if res["pred"] is not None else None
            jsonl_rows.append(
                {
                    "background": os.path.basename(bg_path),
                    "background_file": os.path.basename(bg_path),
                    "background_index": bg_index,
                    "background_path": bg_path,
                    "icon_name": icon_info["name"],
                    "icon_prompt": icon_info["prompt"],
                    "rank": rank,
                    "world_size": world_size,
                    "cell_idx": res["cell_idx"],
                    "grid_row": res["r"],
                    "grid_col": res["c"],
                    "row": res["r"],
                    "col": res["c"],
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "original_width": width,
                    "original_height": height,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "icon_width": icon_w,
                    "icon_height": icon_h,
                    "icon_blend_alpha": icon_blend_alpha,
                    "placement_xy_original": [res["px"], res["py"]],
                    "bbox_original": [x_min, y_min, x_max, y_max],
                    "icon_placement": {
                        "px": res["px"],
                        "py": res["py"],
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    },
                    "model_output": res["raw"],
                    "model_raw_output": res["raw"],
                    "parsed_coord_resized": (
                        [res["parsed_raw_x"], res["parsed_raw_y"]]
                        if res["parsed_raw_x"] is not None and res["parsed_raw_y"] is not None
                        else None
                    ),
                    "parsed_coords": {
                        "raw_x": res["parsed_raw_x"],
                        "raw_y": res["parsed_raw_y"],
                        "pred_x": pred_x,
                        "pred_y": pred_y,
                    },
                    "pred_coord_original": [pred_x, pred_y] if pred_x is not None and pred_y is not None else None,
                    "hit": int(res["hit"]),
                    "error": res["error"],
                }
            )

        with open(rank_jsonl_path, "a", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        new_cells_done += len(batch_items)
        batch_items = []

    for cell_idx in pbar:
        r = cell_idx // cols
        c = cell_idx % cols
        if (r, c) in existing_records:
            continue

        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, width, height
        )

        temp_img = composite_icon_alpha_blend(base_bg, icon, px, py, icon_blend_alpha)
        resized_img = temp_img.resize((resized_width, resized_height), Image.BILINEAR)

        batch_items.append(
            {
                "cell_idx": cell_idx,
                "r": r,
                "c": c,
                "px": px,
                "py": py,
                "image": resized_img,
                "bbox": (x_min, y_min, x_max, y_max),
            }
        )

        if len(batch_items) >= batch_size:
            flush_batch()

    flush_batch()

    rank_path = os.path.join(
        output_dir,
        f"{icon_info['name']}_accuracy_relative_matrix_{bg_index}.rank{rank}.npy",
    )
    np.save(rank_path, rank_matrix)

    elapsed = time.time() - start_time
    print(f"[Rank {rank}] Background {bg_index} shard time: {elapsed/60:.2f} min; added {new_cells_done} records")

    dist_barrier(dist_enabled)

    if rank == 0:
        merged = np.full((rows, cols), np.nan, dtype=np.float32)
        for rr in range(world_size):
            p = os.path.join(
                output_dir,
                f"{icon_info['name']}_accuracy_relative_matrix_{bg_index}.rank{rr}.npy",
            )
            if not os.path.exists(p):
                print(f"⚠️ Missing rank file: {p}")
                continue
            m = np.load(p)
            mask = ~np.isnan(m)
            merged[mask] = m[mask]

        missing = int(np.isnan(merged).sum())
        if missing > 0:
            print(f"⚠️ {missing} cells still have no result after merging; setting them to 0")
            merged = np.nan_to_num(merged, nan=0.0)

        npy_filename = f"{icon_info['name']}_accuracy_relative_matrix_{bg_index}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        np.save(npy_path, merged)
        print(f"💾 Saved merged matrix: {npy_path}")

        png_filename = f"{icon_info['name']}_blind_spot_relative_{bg_index}.png"
        png_path = os.path.join(output_dir, png_filename)
        plt.figure(figsize=(16, 9))
        plt.imshow(merged, cmap="RdYlGn", origin="upper", vmin=0, vmax=1)
        plt.colorbar(label="Accuracy")
        plt.title(
            f'Qwen3-VL Blind Spot Heatmap: {icon_info["name"]}\n'
            f'Background: {os.path.basename(bg_path)} ({width}x{height})'
        )
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"🖼️ Saved heatmap: {png_path}")

        merged_jsonl_filename = f"{icon_info['name']}_qwen_outputs_{bg_index}.jsonl"
        merged_jsonl_path = os.path.join(output_dir, merged_jsonl_filename)
        merged_records_for_jsonl = load_merged_qwen_progress(
            output_dir=output_dir,
            rows=rows,
            cols=cols,
            bg_name=os.path.basename(bg_path),
            bg_index=bg_index,
            icon_name=icon_info["name"],
        )
        for rr in range(world_size):
            rank_records = load_rank_qwen_progress(
                output_dir=output_dir,
                rows=rows,
                cols=cols,
                bg_name=os.path.basename(bg_path),
                bg_index=bg_index,
                icon_name=icon_info["name"],
                rank=rr,
                clean_malformed=False,
            )
            merged_records_for_jsonl.update(rank_records)
        with open(merged_jsonl_path, "w", encoding="utf-8") as fout:
            for cell_idx in range(rows * cols):
                r = cell_idx // cols
                c = cell_idx % cols
                record = merged_records_for_jsonl.get((r, c))
                if record is None:
                    continue
                px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
                    r, c, grid_w, grid_h, icon_w, icon_h, width, height
                )
                if not record_matches_placement(record, px, py, x_min, x_max, y_min, y_max, icon_blend_alpha):
                    continue
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"📝 Saved model outputs JSONL: {merged_jsonl_path}")

    dist_barrier(dist_enabled)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist_enabled, rank, world_size, local_rank, device = dist_init()

    parser = argparse.ArgumentParser()
    parser.add_argument("icon_name", type=str, choices=["star", "circle_ok", "clock"])
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--bg_dir", type=str, default=DEFAULT_BG_DIR)
    parser.add_argument("--icon_dir", type=str, default=DEFAULT_ICON_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--icon_blend_alpha", type=float, default=DEFAULT_ICON_BLEND_ALPHA)
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--max_backgrounds", type=int, default=-1)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not 0.0 <= args.icon_blend_alpha <= 1.0:
        raise ValueError("--icon_blend_alpha must be between 0.0 and 1.0")

    os.makedirs(args.output_dir, exist_ok=True)

    icons = get_icons(args.icon_dir)
    icon_info = icons[args.icon_name]
    if not os.path.exists(icon_info["path"]):
        raise FileNotFoundError(f"Icon file does not exist: {icon_info['path']}")
    if not os.path.isdir(args.bg_dir):
        raise FileNotFoundError(f"Background directory does not exist: {args.bg_dir}")

    rank0_print(rank, f"[Rank {rank}] local_rank={local_rank}, device={device}, world_size={world_size}")

    bg_files = sorted([
        f for f in os.listdir(args.bg_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.max_backgrounds and args.max_backgrounds > 0:
        bg_files = bg_files[: args.max_backgrounds]
    if not bg_files:
        raise FileNotFoundError(f"No background images found in {args.bg_dir}")

    rank0_print(
        rank,
        f"\n📁 Found {len(bg_files)} background images; testing icon: "
        f"{args.icon_name}, size {icon_info['width']}x{icon_info['height']}, "
        f"blend alpha={args.icon_blend_alpha:.2f}",
    )
    rank0_print(rank, "=" * 60)

    bg_jobs = list(enumerate(bg_files, start=1))
    local_missing_by_bg: List[int] = []
    local_total_by_bg: List[int] = []
    for idx, bg_file in bg_jobs:
        bg_path = os.path.join(args.bg_dir, bg_file)
        missing, total = count_missing_cells_for_rank(
            bg_path=bg_path,
            bg_index=idx,
            icon_info=icon_info,
            output_dir=args.output_dir,
            rank=rank,
            world_size=world_size,
            icon_blend_alpha=args.icon_blend_alpha,
        )
        local_missing_by_bg.append(missing)
        local_total_by_bg.append(total)

    if dist_enabled:
        missing_tensor = torch.tensor(local_missing_by_bg, dtype=torch.long, device=device)
        total_tensor = torch.tensor(local_total_by_bg, dtype=torch.long, device=device)
        dist.all_reduce(missing_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        global_missing_by_bg = [int(x) for x in missing_tensor.cpu().tolist()]
        global_total_by_bg = [int(x) for x in total_tensor.cpu().tolist()]
    else:
        global_missing_by_bg = local_missing_by_bg
        global_total_by_bg = local_total_by_bg

    pending_bg_jobs = [
        job
        for job, missing in zip(bg_jobs, global_missing_by_bg)
        if missing > 0
    ]
    total_missing = sum(global_missing_by_bg)
    total_cells = sum(global_total_by_bg)
    rank0_print(
        rank,
        f"🔎 Automatic-resume precheck: {len(bg_jobs) - len(pending_bg_jobs)}/{len(bg_jobs)} backgrounds are complete; "
        f"{total_missing}/{total_cells} cells or finalization tasks remain",
    )

    if not pending_bg_jobs:
        rank0_print(rank, f"✅ All target backgrounds are complete; skipping model loading. Results are saved in {args.output_dir}")
        cleanup_dist(dist_enabled)
        return

    model, processor = load_model(
        model_path=args.model_path,
        device=device,
        rank=rank,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_impl=args.attn_impl,
    )

    try:
        for idx, bg_file in pending_bg_jobs:
            bg_path = os.path.join(args.bg_dir, bg_file)
            run_one_background(
                model=model,
                processor=processor,
                device=device,
                rank=rank,
                world_size=world_size,
                dist_enabled=dist_enabled,
                bg_path=bg_path,
                bg_index=idx,
                icon_info=icon_info,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
                icon_blend_alpha=args.icon_blend_alpha,
            )
    finally:
        cleanup_dist(dist_enabled)

    rank0_print(rank, f"\n🎉 Complete! Results are saved in {args.output_dir}")


if __name__ == "__main__":
    main()
