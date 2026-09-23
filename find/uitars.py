from __future__ import annotations

from common import (
    dist_init,
    rank0_print,
    dist_barrier,
    cleanup_dist,
    _get_pad_token_id,
    compute_icon_placement,
    record_to_accuracy,
    ensure_jsonl_append_boundary,
    load_existing_jsonl_progress,
)

import argparse
import glob
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from qwen_vl_utils import smart_resize


# ========== Default configuration ==========
DEFAULT_MODEL_PATH = "ByteDance-Seed/UI-TARS-1.5-7B"

DEFAULT_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'background')
DEFAULT_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/uitars')

MIN_PIXELS = 3136
MAX_PIXELS = 4096 * 2160

SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, 
provide the coordinates of the specified element as a single (x,y) point. 
The image resolution is height {height} and width {width}. 
For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

_COORD_RE = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")


# -------------------------
# Distributed helpers
# -------------------------


# -------------------------
# General helpers
# -------------------------
def get_icons(icon_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        "star": {
            "path": os.path.join(icon_dir, "star_icon_40.png"),
            "width": 40,
            "height": 40,
            "prompt": "a red five-pointed star shape.",
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
            "path": os.path.join(icon_dir, "clock_text_40x60.png"),
            "width": 40,
            "height": 60,
            "prompt": "the clock icon, a white circle with black hour hand pointing to 3 o'clock and minute hand pointing to 12 o'clock, with text 'clock' on the right",
            "name": "clock",
        },
    }


def extract_coordinates(raw_string: str) -> Optional[Tuple[float, float]]:
    """
    Parse (x,y) first; if that fails, fall back to extracting the first two numbers.
    GTA1 treats these as pixel coordinates and no longer interprets values <=1005 as normalized 0-1000 coordinates.
    """
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
    """
    Follow the approach used by the fast script above:
    text + image placeholder + text, followed by batched processing with processor(text=[...], images=[...]).
    """
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


def record_matches_placement(
    record: Dict[str, Any],
    px: int,
    py: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> bool:
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


def get_uitars_jsonl_paths(output_dir: str, icon_name: str, bg_index: int, rank: int) -> Tuple[str, str]:
    merged_jsonl_path = os.path.join(output_dir, f"{icon_name}_outputs_{bg_index}.jsonl")
    rank_jsonl_path = os.path.join(output_dir, f"{icon_name}_outputs_{bg_index}.rank{rank}.jsonl")
    return merged_jsonl_path, rank_jsonl_path


def get_background_png_path(output_dir: str, icon_name: str, bg_index: int) -> str:
    return os.path.join(output_dir, f"{icon_name}_blind_spot_relative_{bg_index}.png")


def load_all_uitars_progress(
    *,
    output_dir: str,
    rows: int,
    cols: int,
    bg_name: str,
    bg_index: int,
    icon_name: str,
    clean_malformed: bool = True,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    merged_jsonl_path = os.path.join(output_dir, f"{icon_name}_outputs_{bg_index}.jsonl")
    progress = load_existing_jsonl_progress(
        jsonl_path=merged_jsonl_path,
        rows=rows,
        cols=cols,
        bg_name=bg_name,
        bg_index=bg_index,
        icon_name=icon_name,
        clean_malformed=False,
    )

    rank_pattern = os.path.join(output_dir, f"{icon_name}_outputs_{bg_index}.rank*.jsonl")
    for rank_jsonl_path in sorted(glob.glob(rank_pattern)):
        rank_progress = load_existing_jsonl_progress(
            jsonl_path=rank_jsonl_path,
            rows=rows,
            cols=cols,
            bg_name=bg_name,
            bg_index=bg_index,
            icon_name=icon_name,
            clean_malformed=clean_malformed,
        )
        progress.update(rank_progress)

    return progress


def merged_outputs_complete(
    *,
    output_dir: str,
    rows: int,
    cols: int,
    bg_name: str,
    bg_index: int,
    icon_name: str,
    grid_w: float,
    grid_h: float,
    icon_w: int,
    icon_h: int,
    width: int,
    height: int,
) -> bool:
    merged_jsonl_path = os.path.join(output_dir, f"{icon_name}_outputs_{bg_index}.jsonl")
    npy_path = os.path.join(output_dir, f"{icon_name}_accuracy_relative_matrix_{bg_index}.npy")
    png_path = os.path.join(output_dir, f"{icon_name}_blind_spot_relative_{bg_index}.png")

    if not (os.path.exists(merged_jsonl_path) and os.path.exists(npy_path) and os.path.exists(png_path)):
        return False

    try:
        merged = np.load(npy_path)
    except Exception:
        return False
    if tuple(merged.shape) != (rows, cols):
        return False

    merged_records = load_existing_jsonl_progress(
        jsonl_path=merged_jsonl_path,
        rows=rows,
        cols=cols,
        bg_name=bg_name,
        bg_index=bg_index,
        icon_name=icon_name,
        clean_malformed=False,
    )
    if len(merged_records) < rows * cols:
        return False

    for cell_idx in range(rows * cols):
        r = cell_idx // cols
        c = cell_idx % cols
        record = merged_records.get((r, c))
        if record is None:
            return False
        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, width, height
        )
        if not record_matches_placement(record, px, py, x_min, x_max, y_min, y_max):
            return False

    return True


def count_missing_cells_for_rank(
    *,
    bg_path: str,
    bg_index: int,
    icon_info: Dict[str, Any],
    output_dir: str,
    rank: int,
    world_size: int,
) -> Tuple[int, int]:
    base_bg = Image.open(bg_path).convert("RGB")
    width, height = base_bg.size

    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])
    cols = math.ceil(width / icon_w)
    rows = height // icon_h
    if rows == 0:
        return 0, 0

    total_cells = rows * cols
    png_path = get_background_png_path(output_dir, icon_info["name"], bg_index)
    if os.path.exists(png_path):
        if rank == 0:
            print(f"✅ Background {bg_index} already has a PNG; treating it as complete and skipping: {png_path}")
        return 0, len([i for i in range(total_cells) if (i % world_size) == rank])

    grid_w = width / cols
    grid_h = height / rows
    my_cell_indices = [i for i in range(total_cells) if (i % world_size) == rank]
    existing_records = load_all_uitars_progress(
        output_dir=output_dir,
        rows=rows,
        cols=cols,
        bg_name=os.path.basename(bg_path),
        bg_index=bg_index,
        icon_name=icon_info["name"],
        clean_malformed=False,
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
        if record_matches_placement(record, px, py, x_min, x_max, y_min, y_max):
            completed += 1

    missing = len(my_cell_indices) - completed
    if missing == 0 and rank == 0:
        complete = merged_outputs_complete(
            output_dir=output_dir,
            rows=rows,
            cols=cols,
            bg_name=os.path.basename(bg_path),
            bg_index=bg_index,
            icon_name=icon_info["name"],
            grid_w=grid_w,
            grid_h=grid_h,
            icon_w=icon_w,
            icon_h=icon_h,
            width=width,
            height=height,
        )
        if not complete:
            missing = 1

    return missing, len(my_cell_indices)


# -------------------------
# Model: use the invocation pattern from the fast script above.
# -------------------------
def load_model(
    *,
    model_path: str,
    device: torch.device,
    rank: int,
    min_pixels: int,
    max_pixels: int,
    attn_impl: str,
) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    """
    Keep the configuration consistent with the fast script above:
    - AutoConfig checks model_type
    - torch_dtype="auto"
    - trust_remote_code=True
    - low_cpu_mem_usage=True
    - device_map={"": str(device)}
    - processor receives min_pixels / max_pixels directly
    """
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt != "qwen2_5_vl":
        raise RuntimeError(
            f"Expected model_type='qwen2_5_vl' but got {mt!r}. "
            f"Please check --model_path={model_path}"
        )

    model_kwargs = dict(
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},  # Place the entire model on this rank's GPU.
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    rank0_print(rank, f"[Rank {rank}] Loading model on {device} from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=True,
    )

    model.eval()
    rank0_print(rank, f"[Rank {rank}] Model loaded.")
    return model, processor


# -------------------------
# Batched inference
# -------------------------
def run_batch(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    batch_items: List[Dict[str, Any]],
    prompt_text: str,
    pad_token_id: int,
    max_new_tokens: int,
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """
    Each item in batch_items contains:
    r, c, image, bbox
    The image has already been resized to the dimensions seen by GTA1.
    """
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

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
            pad_token_id=pad_token_id,
        )

    # All prompts in the batch are identical, so remove the input using the padded prompt length.
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]

    out_texts = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results: List[Dict[str, Any]] = []
    for it, raw in zip(batch_items, out_texts):
        coords = extract_coordinates(raw)
        pred = None
        hit = 0

        if coords is not None:
            raw_x, raw_y = coords

            # GTA1 outputs resized-image coordinates; map them back to the original background coordinates.
            pred_x = raw_x * scale_x
            pred_y = raw_y * scale_y
            pred = (pred_x, pred_y)

            x_min, y_min, x_max, y_max = it["bbox"]
            if x_min <= pred_x <= x_max and y_min <= pred_y <= y_max:
                hit = 1

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
                "parsed_coord_resized": coords,
                "pred": pred,
            }
        )

    return results


# -------------------------
# Main background scan
# -------------------------
def run_one_background(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
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
) -> None:
    base_bg = Image.open(bg_path).convert("RGB")
    W, H = base_bg.size

    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])

    COLS = math.ceil(W / icon_w)
    ROWS = H // icon_h

    if ROWS == 0:
        rank0_print(rank, "⚠️ The background is shorter than the icon; cannot place the icon, skipping")
        return

    grid_w = W / COLS
    grid_h = H / ROWS

    # Do not read processor.image_processor.min_pixels/max_pixels.
    # Use args.min_pixels / args.max_pixels directly to match the fast script's settings.
    image_processor = processor.image_processor
    patch_size = getattr(image_processor, "patch_size", 14)
    merge_size = getattr(image_processor, "merge_size", 2)

    resized_height, resized_width = smart_resize(
        H,
        W,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    scale_x = W / resized_width
    scale_y = H / resized_height

    prompt_text = make_prompt_text(
        processor=processor,
        resized_width=resized_width,
        resized_height=resized_height,
        instruction=icon_info["prompt"],
    )

    icon = Image.open(icon_info["path"]).convert("RGBA").resize((icon_w, icon_h))
    pad_token_id = _get_pad_token_id(processor)

    rank0_print(rank, f"\n🖼️ Background [{bg_index}]: {os.path.basename(bg_path)} ({W}x{H})")
    rank0_print(rank, f"📐 Grid: {COLS} columns x {ROWS} rows; icon size {icon_w}x{icon_h}px")
    rank0_print(rank, f"🔁 GTA1 input size: {resized_width}x{resized_height}")
    rank0_print(rank, f"🔁 Coordinate scale: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    rank0_print(rank, f"🚀 batch inference: batch_size={batch_size}, world_size={world_size}")

    total_cells = ROWS * COLS
    my_cell_indices = [i for i in range(total_cells) if (i % world_size) == rank]

    # Fill rank_matrix only for cells assigned to this rank; leave all others as NaN.
    rank_matrix = np.full((ROWS, COLS), np.nan, dtype=np.float32)

    # Each rank writes its own JSONL shard to prevent interleaved output from multiple processes.
    # Rank 0 merges all shards into a complete JSONL file after all ranks finish.
    _, rank_jsonl_path = get_uitars_jsonl_paths(output_dir, icon_info["name"], bg_index, rank)
    existing_records = load_all_uitars_progress(
        output_dir=output_dir,
        rows=ROWS,
        cols=COLS,
        bg_name=os.path.basename(bg_path),
        bg_index=bg_index,
        icon_name=icon_info["name"],
    )

    valid_existing_records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    stale_records = 0
    for cell_idx in my_cell_indices:
        r = cell_idx // COLS
        c = cell_idx % COLS
        record = existing_records.get((r, c))
        if record is None:
            continue
        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, W, H
        )
        if record_matches_placement(record, px, py, x_min, x_max, y_min, y_max):
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

    pbar = tqdm(
        my_cell_indices,
        desc=f"BG{bg_index}-rank{rank}",
        disable=(rank != 0),
    )

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
            scale_x=scale_x,
            scale_y=scale_y,
        )
        jsonl_rows: List[Dict[str, Any]] = []
        for res in results:
            rank_matrix[res["r"], res["c"]] = float(res["hit"])
            x_min, y_min, x_max, y_max = res["bbox"]

            jsonl_rows.append(
                {
                    "background": os.path.basename(bg_path),
                    "background_index": bg_index,
                    "background_file": os.path.basename(bg_path),
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
                    "grid_rows": ROWS,
                    "grid_cols": COLS,
                    "original_width": W,
                    "original_height": H,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "scale_x": scale_x,
                    "scale_y": scale_y,
                    "icon_width": icon_w,
                    "icon_height": icon_h,
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
                        list(res["parsed_coord_resized"])
                        if res["parsed_coord_resized"] is not None
                        else None
                    ),
                    "pred_coord_original": (
                        list(res["pred"]) if res["pred"] is not None else None
                    ),
                    "hit": int(res["hit"]),
                }
            )

        with open(rank_jsonl_path, "a", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        new_cells_done += len(batch_items)
        batch_items = []

    for cell_idx in pbar:
        r = cell_idx // COLS
        c = cell_idx % COLS
        if (r, c) in existing_records:
            continue

        px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
            r, c, grid_w, grid_h, icon_w, icon_h, W, H
        )

        temp_img = base_bg.copy()
        temp_img.paste(icon, (px, py), icon)

        # Manually resize to the dimensions returned by smart_resize.
        # This keeps the resolution in the prompt consistent with the image dimensions seen by the model.
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

    # Wait for all ranks to finish writing their rank_matrix files.
    dist_barrier(dist_enabled)

    # Rank 0 merges all rank matrices and saves the final NPY file and heatmap.
    if rank == 0:
        merged = np.full((ROWS, COLS), np.nan, dtype=np.float32)

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
        plt.imshow(
            merged,
            cmap="RdYlGn",
            origin="upper",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(label="Accuracy")
        plt.title(
            f'GTA1-7B Blind Spot Heatmap: {icon_info["name"]}\n'
            f'Background: {os.path.basename(bg_path)} ({W}x{H})'
        )
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"🖼️ Saved heatmap: {png_path}")

        merged_jsonl_filename = f"{icon_info['name']}_outputs_{bg_index}.jsonl"
        merged_jsonl_path = os.path.join(output_dir, merged_jsonl_filename)
        merged_records_for_jsonl = load_all_uitars_progress(
            output_dir=output_dir,
            rows=ROWS,
            cols=COLS,
            bg_name=os.path.basename(bg_path),
            bg_index=bg_index,
            icon_name=icon_info["name"],
            clean_malformed=False,
        )
        with open(merged_jsonl_path, "w", encoding="utf-8") as fout:
            for cell_idx in range(ROWS * COLS):
                r = cell_idx // COLS
                c = cell_idx % COLS
                record = merged_records_for_jsonl.get((r, c))
                if record is None:
                    continue
                px, py, x_min, x_max, y_min, y_max = compute_icon_placement(
                    r, c, grid_w, grid_h, icon_w, icon_h, W, H
                )
                if not record_matches_placement(record, px, py, x_min, x_max, y_min, y_max):
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

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["", "flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument("--max_backgrounds", type=int, default=-1)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

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
        f"{args.icon_name}, size {icon_info['width']}x{icon_info['height']}"
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
            )
    finally:
        cleanup_dist(dist_enabled)

    rank0_print(rank, f"\n🎉 Complete! Results are saved in {args.output_dir}")


if __name__ == "__main__":
    main()
