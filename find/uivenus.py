#!/usr/bin/env python3
from __future__ import annotations

from common import (
    dist_init,
    rank0_print,
    dist_barrier,
    cleanup_dist,
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
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:
    raise ImportError(
        "缺少 qwen-vl-utils。请先运行：pip install qwen-vl-utils\n"
        "官方建议依赖：pip install transformers==4.49.0 qwen-vl-utils"
    ) from exc


# ========== 默认配置 ==========
DEFAULT_MODEL_PATH = "inclusionAI/UI-Venus-Ground-7B"
DEFAULT_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'background')
DEFAULT_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/uivenus')

MIN_PIXELS = 3136
MAX_PIXELS = 4096 * 2160
MODEL_DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = True


def get_result_paths(output_dir: str, icon_name: str, bg_index: int) -> Dict[str, str]:
    base = f"{icon_name}_accuracy_relative_matrix_{bg_index}"
    return {
        "npy": os.path.join(output_dir, f"{base}.npy"),
        "png": os.path.join(output_dir, f"{icon_name}_blind_spot_relative_{bg_index}.png"),
        "jsonl": os.path.join(output_dir, f"{icon_name}_results_bg_{bg_index}.jsonl"),
        "done": os.path.join(output_dir, f"{base}.done.json"),
    }


def get_rank_result_paths(output_dir: str, icon_name: str, bg_index: int, rank: int) -> Dict[str, str]:
    return {
        "npy": os.path.join(
            output_dir,
            f"{icon_name}_accuracy_relative_matrix_{bg_index}.rank{rank}.npy",
        ),
        "jsonl": os.path.join(output_dir, f"{icon_name}_results_bg_{bg_index}.rank{rank}.jsonl"),
    }


def get_background_grid_shape(bg_path: str, icon_w: int, icon_h: int) -> Tuple[int, int, int, int]:
    with Image.open(bg_path) as bg:
        width, height = bg.size
    cols = math.ceil(width / icon_w)
    rows = height // icon_h
    return rows, cols, width, height


def saved_result_is_complete(
    *,
    bg_path: str,
    icon_info: Dict[str, Any],
    bg_index: int,
    output_dir: str,
) -> bool:
    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])
    rows, cols, width, height = get_background_grid_shape(bg_path, icon_w, icon_h)
    if rows == 0:
        return False

    total_cells = rows * cols
    paths = get_result_paths(output_dir, icon_info["name"], bg_index)
    if not (os.path.exists(paths["npy"]) and os.path.exists(paths["png"]) and os.path.exists(paths["jsonl"])):
        return False

    if os.path.exists(paths["done"]):
        try:
            with open(paths["done"], "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Resume check ignored unreadable metadata {paths['done']}: {exc}")
            return False

        return (
            metadata.get("background_file") == os.path.basename(bg_path)
            and metadata.get("icon_name") == icon_info["name"]
            and metadata.get("width") == width
            and metadata.get("height") == height
            and metadata.get("rows") == rows
            and metadata.get("cols") == cols
            and metadata.get("completed_cells", 0) >= total_cells
            and metadata.get("total_cells") == total_cells
        )

    try:
        existing = np.load(paths["npy"], mmap_mode="r")
        if existing.shape != (rows, cols) or existing.size != total_cells:
            return False
    except (OSError, ValueError) as exc:
        print(f"Resume check ignored unreadable result {paths['npy']}: {exc}")
        return False

    completed_cells: set[int] = set()
    try:
        with open(paths["jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("background") != os.path.basename(bg_path):
                    continue
                if row.get("background_index") != bg_index:
                    continue
                if row.get("icon_name") != icon_info["name"]:
                    continue
                cell_idx = row.get("cell_idx")
                grid_row = row.get("grid_row")
                grid_col = row.get("grid_col")
                if not isinstance(cell_idx, int) or not isinstance(grid_row, int) or not isinstance(grid_col, int):
                    continue
                if 0 <= grid_row < rows and 0 <= grid_col < cols and cell_idx == grid_row * cols + grid_col:
                    completed_cells.add(cell_idx)
    except OSError as exc:
        print(f"Resume check ignored unreadable details {paths['jsonl']}: {exc}")
        return False

    return len(completed_cells) >= total_cells


def load_rank_progress(
    *,
    rank_jsonl_path: str,
    rank_matrix: np.ndarray,
    bg_path: str,
    bg_index: int,
    icon_info: Dict[str, Any],
    rank: int,
    world_size: int,
    rows: int,
    cols: int,
) -> Tuple[set[int], List[Dict[str, Any]]]:
    completed_cells: set[int] = set()
    latest_rows: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(rank_jsonl_path):
        return completed_cells, []

    with open(rank_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Resume ignored bad JSON at {rank_jsonl_path}:{line_no}: {exc}")
                continue

            if row.get("background") != os.path.basename(bg_path):
                continue
            if row.get("background_index") != bg_index:
                continue
            if row.get("icon_name") != icon_info["name"]:
                continue
            if row.get("rank") != rank or row.get("world_size") != world_size:
                continue

            cell_idx = row.get("cell_idx")
            grid_row = row.get("grid_row")
            grid_col = row.get("grid_col")
            if not isinstance(cell_idx, int) or not isinstance(grid_row, int) or not isinstance(grid_col, int):
                continue
            if cell_idx < 0 or cell_idx >= rows * cols:
                continue
            if not (0 <= grid_row < rows and 0 <= grid_col < cols):
                continue
            if cell_idx != grid_row * cols + grid_col:
                continue
            if (cell_idx % world_size) != rank:
                continue

            hit = row.get("hit")
            if hit not in (0, 1):
                continue

            rank_matrix[grid_row, grid_col] = float(hit)
            completed_cells.add(cell_idx)
            latest_rows[cell_idx] = row

    return completed_cells, [latest_rows[i] for i in sorted(latest_rows)]


def write_done_metadata(
    *,
    done_path: str,
    bg_path: str,
    bg_index: int,
    icon_info: Dict[str, Any],
    width: int,
    height: int,
    rows: int,
    cols: int,
    world_size: int,
    npy_path: str,
    png_path: str,
    jsonl_path: str,
) -> None:
    metadata = {
        "background_file": os.path.basename(bg_path),
        "background_index": bg_index,
        "background_path": bg_path,
        "icon_name": icon_info["name"],
        "width": width,
        "height": height,
        "rows": rows,
        "cols": cols,
        "completed_cells": rows * cols,
        "total_cells": rows * cols,
        "world_size": world_size,
        "npy_file": os.path.basename(npy_path),
        "png_file": os.path.basename(png_path),
        "jsonl_file": os.path.basename(jsonl_path),
    }
    tmp_path = f"{done_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, done_path)


def get_icons(icon_dir: str) -> Dict[str, Dict[str, Any]]:
    return {
        "gemini": {
            "path": os.path.join(icon_dir, "gemini_icon_40.png"),
            "width": 40,
            "height": 40,
            "prompt": "A red five-pointed star shape",
            "name": "gemini",
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
            "prompt": (
                "the clock icon, a white circle with black hands pointing to 3 o'clock "
                "and 12 o'clock, with the text 'clock' on the right"
            ),
            "name": "clock",
        },
    }


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _get_pad_token_id(processor: Any) -> Optional[int]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return None


def build_uivenus_prompt(element_description: str) -> str:
    return (
        "Outline the position corresponding to the instruction: "
        f"{element_description}. "
        "The output should be only [x1,y1,x2,y2]."
    )


def parse_bbox(output: str) -> Optional[List[float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", output)
    if len(nums) < 4:
        return None
    return [float(x) for x in nums[:4]]


def bbox_to_original_point(
    bbox: List[float],
    original_w: int,
    original_h: int,
    processed_w: float,
    processed_h: float,
) -> Tuple[float, float, List[float], str]:
    x1, y1, x2, y2 = bbox
    max_abs = max(abs(v) for v in bbox)

    if max_abs <= 1.5:
        norm_bbox = [x1, y1, x2, y2]
        coord_mode = "normalized_0_1"
    else:
        norm_bbox = [
            x1 / processed_w,
            y1 / processed_h,
            x2 / processed_w,
            y2 / processed_h,
        ]
        coord_mode = "processed_pixels"

    cx_norm = (norm_bbox[0] + norm_bbox[2]) / 2.0
    cy_norm = (norm_bbox[1] + norm_bbox[3]) / 2.0
    pred_x = cx_norm * original_w
    pred_y = cy_norm * original_h
    return pred_x, pred_y, norm_bbox, coord_mode


def load_model(
    *,
    model_path: str,
    device: torch.device,
    rank: int,
    min_pixels: int,
    max_pixels: int,
    attn_impl: str,
) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    rank0_print(rank, f"[Rank {rank}] Loading UI-Venus-Ground on {device} from {model_path} ...")

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    model_kwargs = dict(
        torch_dtype=MODEL_DTYPE,
        device_map={"": str(device)},
        trust_remote_code=TRUST_REMOTE_CODE,
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        )
    except Exception as exc:
        if attn_impl == "flash_attention_2":
            rank0_print(rank, f"⚠️ flash_attention_2 加载失败，改用默认 attention。原始错误: {exc}")
            model_kwargs.pop("attn_implementation", None)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs,
            )
        else:
            raise

    model.eval()
    rank0_print(rank, f"[Rank {rank}] Model loaded.")
    return model, processor


def run_batch(
    *,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    device: torch.device,
    batch_items: List[Dict[str, Any]],
    prompt: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    pad_token_id: Optional[int],
    original_w: int,
    original_h: int,
) -> List[Dict[str, Any]]:
    if not batch_items:
        return []

    messages_list = []
    texts = []
    image_inputs = []
    video_inputs = []
    for item in batch_items:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item["image"],
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        messages_list.append(messages)
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images, videos = process_vision_info(messages)
        image_inputs.extend(images or [])
        if videos:
            video_inputs.extend(videos)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    )
    inputs = move_to_device(inputs, device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": 0.0,
        "use_cache": True,
    }
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, prompt_len:]
        out_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise RuntimeError("processor 输出中缺少 image_grid_thw，无法把 UI-Venus 坐标转回原图。")

        processed_sizes = [
            (
                float(image_grid_thw[i][2].item() * 14),
                float(image_grid_thw[i][1].item() * 14),
            )
            for i in range(len(batch_items))
        ]
        errors: List[Optional[str]] = [None] * len(batch_items)
    except Exception as exc:
        out_texts = [""] * len(batch_items)
        processed_sizes = [(None, None)] * len(batch_items)
        errors = [str(exc)] * len(batch_items)

    results: List[Dict[str, Any]] = []
    for item, raw_output, (processed_w, processed_h), error_msg in zip(
        batch_items,
        out_texts,
        processed_sizes,
        errors,
    ):
        parsed_bbox = None
        norm_bbox = None
        coord_mode = None
        pred_x = None
        pred_y = None
        hit = 0

        if error_msg is None:
            parsed_bbox = parse_bbox(raw_output)
            if parsed_bbox is not None and processed_w is not None and processed_h is not None:
                pred_x, pred_y, norm_bbox, coord_mode = bbox_to_original_point(
                    bbox=parsed_bbox,
                    original_w=original_w,
                    original_h=original_h,
                    processed_w=processed_w,
                    processed_h=processed_h,
                )
                x_min, y_min, x_max, y_max = item["bbox"]
                if x_min <= pred_x <= x_max and y_min <= pred_y <= y_max:
                    hit = 1
            else:
                error_msg = f"Cannot parse bbox from output: {raw_output}"

        results.append(
            {
                "cell_idx": item["cell_idx"],
                "r": item["r"],
                "c": item["c"],
                "px": item["px"],
                "py": item["py"],
                "bbox": item["bbox"],
                "hit": hit,
                "raw_output": raw_output,
                "parsed_bbox": parsed_bbox,
                "norm_bbox": norm_bbox,
                "coord_mode": coord_mode,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "processed_w": processed_w,
                "processed_h": processed_h,
                "error": error_msg,
            }
        )

    return results


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
    width, height = base_bg.size

    icon_w = int(icon_info["width"])
    icon_h = int(icon_info["height"])
    cols = math.ceil(width / icon_w)
    rows = height // icon_h
    if rows == 0:
        rank0_print(rank, "⚠️ 背景高度小于图标高度，无法放置图标，跳过")
        return

    grid_w = width / cols
    grid_h = height / rows
    icon = Image.open(icon_info["path"]).convert("RGBA").resize((icon_w, icon_h))
    target_prompt = build_uivenus_prompt(icon_info["prompt"])
    pad_token_id = _get_pad_token_id(processor)

    rank0_print(rank, f"\n🖼️ 背景 [{bg_index}]: {os.path.basename(bg_path)} ({width}x{height})")
    rank0_print(rank, f"📐 网格划分: {cols} 列 x {rows} 行，图标尺寸 {icon_w}x{icon_h}px")
    rank0_print(rank, f"🚀 batch inference: batch_size={batch_size}, world_size={world_size}")

    total_cells = rows * cols
    rank_matrix = np.full((rows, cols), np.nan, dtype=np.float32)

    rank_paths = get_rank_result_paths(output_dir, icon_info["name"], bg_index, rank)
    completed_cells, _ = load_rank_progress(
        rank_jsonl_path=rank_paths["jsonl"],
        rank_matrix=rank_matrix,
        bg_path=bg_path,
        bg_index=bg_index,
        icon_info=icon_info,
        rank=rank,
        world_size=world_size,
        rows=rows,
        cols=cols,
    )
    my_cell_indices = [
        i for i in range(total_cells)
        if (i % world_size) == rank and i not in completed_cells
    ]
    if completed_cells:
        rank0_print(
            rank,
            f"♻️ 自动续跑: rank{rank} 已恢复 {len(completed_cells)} 个 cell，"
            f"剩余 {len(my_cell_indices)} 个 cell",
        )

    start_time = time.time()
    batch_items: List[Dict[str, Any]] = []
    pbar = tqdm(my_cell_indices, desc=f"BG{bg_index}-rank{rank}", disable=(rank != 0))

    def flush_batch() -> None:
        nonlocal batch_items
        if not batch_items:
            return

        results = run_batch(
            model=model,
            processor=processor,
            device=device,
            batch_items=batch_items,
            prompt=target_prompt,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            original_w=width,
            original_h=height,
        )

        jsonl_rows: List[Dict[str, Any]] = []
        for res in results:
            rank_matrix[res["r"], res["c"]] = float(res["hit"])
            x_min, y_min, x_max, y_max = res["bbox"]
            jsonl_rows.append(
                {
                    "background": os.path.basename(bg_path),
                    "background_index": bg_index,
                    "background_path": bg_path,
                    "icon_name": icon_info["name"],
                    "icon_prompt": icon_info["prompt"],
                    "rank": rank,
                    "world_size": world_size,
                    "cell_idx": res["cell_idx"],
                    "grid_row": res["r"],
                    "grid_col": res["c"],
                    "grid_rows": rows,
                    "grid_cols": cols,
                    "image_size": {"width": width, "height": height},
                    "processed_size": {
                        "width": res["processed_w"],
                        "height": res["processed_h"],
                    },
                    "icon_placement": {
                        "px": res["px"],
                        "py": res["py"],
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "center_x": res["px"] + icon_w / 2.0,
                        "center_y": res["py"] + icon_h / 2.0,
                    },
                    "prompt": target_prompt,
                    "model_raw_output": res["raw_output"],
                    "parsed_coords": {
                        "bbox_raw": res["parsed_bbox"],
                        "bbox_norm": res["norm_bbox"],
                        "coord_mode": res["coord_mode"],
                        "pred_x": res["pred_x"],
                        "pred_y": res["pred_y"],
                    },
                    "hit": int(res["hit"]),
                    "error": res["error"],
                }
            )

        with open(rank_paths["jsonl"], "a", encoding="utf-8") as f:
            for row in jsonl_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        batch_items = []

    for cell_idx in pbar:
        r = cell_idx // cols
        c = cell_idx % cols

        px = int(c * grid_w + (grid_w - icon_w) / 2)
        py = int(r * grid_h + (grid_h - icon_h) / 2)
        px = max(0, min(px, width - icon_w))
        py = max(0, min(py, height - icon_h))

        x_min, x_max = px, px + icon_w
        y_min, y_max = py, py + icon_h

        temp_img = base_bg.copy()
        temp_img.paste(icon, (px, py), icon)

        batch_items.append(
            {
                "cell_idx": cell_idx,
                "r": r,
                "c": c,
                "px": px,
                "py": py,
                "image": temp_img,
                "bbox": (x_min, y_min, x_max, y_max),
            }
        )

        if len(batch_items) >= batch_size:
            flush_batch()

    flush_batch()

    np.save(rank_paths["npy"], rank_matrix)

    elapsed = time.time() - start_time
    rank0_print(rank, f"⏱️ 背景 {bg_index} rank0 shard 用时: {elapsed/60:.2f} min")

    dist_barrier(dist_enabled)

    if rank == 0:
        merged = np.full((rows, cols), np.nan, dtype=np.float32)
        for rr in range(world_size):
            p = get_rank_result_paths(output_dir, icon_info["name"], bg_index, rr)["npy"]
            if not os.path.exists(p):
                print(f"⚠️ 缺少 rank 文件: {p}")
                continue
            m = np.load(p)
            mask = ~np.isnan(m)
            merged[mask] = m[mask]

        missing = int(np.isnan(merged).sum())
        if missing > 0:
            print(f"⚠️ 合并后仍有 {missing} 个 cell 没有结果，将其置为 0")
            merged = np.nan_to_num(merged, nan=0.0)

        final_paths = get_result_paths(output_dir, icon_info["name"], bg_index)
        np.save(final_paths["npy"], merged)
        print(f"💾 已保存合并矩阵: {final_paths['npy']}")

        plt.figure(figsize=(16, 9))
        plt.imshow(merged, cmap="RdYlGn", origin="upper", vmin=0, vmax=1)
        plt.colorbar(label="Accuracy")
        plt.title(
            f'UI-Venus-Ground Blind Spot Heatmap: {icon_info["name"]}\n'
            f'Background: {os.path.basename(bg_path)} ({width}x{height})'
        )
        plt.savefig(final_paths["png"], dpi=150)
        plt.close()
        print(f"🖼️ 已保存 heatmap: {final_paths['png']}")

        merged_rows_by_cell: Dict[int, Dict[str, Any]] = {}
        for rr in range(world_size):
            shard_paths = get_rank_result_paths(output_dir, icon_info["name"], bg_index, rr)
            shard_matrix = np.full((rows, cols), np.nan, dtype=np.float32)
            _, shard_rows = load_rank_progress(
                rank_jsonl_path=shard_paths["jsonl"],
                rank_matrix=shard_matrix,
                bg_path=bg_path,
                bg_index=bg_index,
                icon_info=icon_info,
                rank=rr,
                world_size=world_size,
                rows=rows,
                cols=cols,
            )
            if not os.path.exists(shard_paths["jsonl"]):
                print(f"⚠️ 缺少 rank jsonl 文件: {shard_paths['jsonl']}")
                continue
            for row in shard_rows:
                merged_rows_by_cell[int(row["cell_idx"])] = row

        with open(final_paths["jsonl"], "w", encoding="utf-8") as fout:
            for cell_idx in sorted(merged_rows_by_cell):
                fout.write(json.dumps(merged_rows_by_cell[cell_idx], ensure_ascii=False) + "\n")
        print(f"📝 已保存详细结果: {final_paths['jsonl']}")

        if missing == 0 and len(merged_rows_by_cell) >= total_cells:
            write_done_metadata(
                done_path=final_paths["done"],
                bg_path=bg_path,
                bg_index=bg_index,
                icon_info=icon_info,
                width=width,
                height=height,
                rows=rows,
                cols=cols,
                world_size=world_size,
                npy_path=final_paths["npy"],
                png_path=final_paths["png"],
                jsonl_path=final_paths["jsonl"],
            )
            print(f"✅ 已写入完成标记: {final_paths['done']}")
        elif os.path.exists(final_paths["done"]):
            os.remove(final_paths["done"])
            print(f"⚠️ 本次结果不完整，已移除完成标记: {final_paths['done']}")

    dist_barrier(dist_enabled)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist_enabled, rank, world_size, local_rank, device = dist_init()

    parser = argparse.ArgumentParser(description="UI-Venus-Ground-7B blind-zone probing")
    parser.add_argument("icon_name", type=str, choices=["gemini", "circle_ok", "clock"])
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
        raise FileNotFoundError(f"图标文件不存在: {icon_info['path']}")
    if not os.path.isdir(args.bg_dir):
        raise FileNotFoundError(f"背景目录不存在: {args.bg_dir}")

    bg_files = sorted([
        f for f in os.listdir(args.bg_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if args.max_backgrounds and args.max_backgrounds > 0:
        bg_files = bg_files[: args.max_backgrounds]
    if not bg_files:
        raise FileNotFoundError(f"未在 {args.bg_dir} 中找到背景图片")

    rank0_print(
        rank,
        f"\n📁 找到 {len(bg_files)} 张背景图片，将测试图标: "
        f"{args.icon_name}，尺寸 {icon_info['width']}x{icon_info['height']}",
    )
    rank0_print(rank, "=" * 60)

    pending_backgrounds: List[Tuple[int, str]] = []
    skipped_backgrounds = 0
    for idx, bg_file in enumerate(bg_files, start=1):
        bg_path = os.path.join(args.bg_dir, bg_file)
        if saved_result_is_complete(
            bg_path=bg_path,
            icon_info=icon_info,
            bg_index=idx,
            output_dir=args.output_dir,
        ):
            final_paths = get_result_paths(args.output_dir, icon_info["name"], idx)
            rank0_print(rank, f"♻️ 自动续跑跳过 background {idx}/{len(bg_files)}: {bg_file}")
            rank0_print(rank, f"   已存在: {final_paths['npy']}")
            skipped_backgrounds += 1
        else:
            pending_backgrounds.append((idx, bg_path))

    rank0_print(
        rank,
        f"Auto-resume summary: skipped={skipped_backgrounds}, "
        f"pending={len(pending_backgrounds)}, total={len(bg_files)}",
    )
    if not pending_backgrounds:
        rank0_print(rank, f"\n🎉 完成！所有请求的结果已存在于 {args.output_dir}")
        cleanup_dist(dist_enabled)
        return

    rank0_print(rank, f"[Rank {rank}] local_rank={local_rank}, device={device}, world_size={world_size}")
    model, processor = load_model(
        model_path=args.model_path,
        device=device,
        rank=rank,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_impl=args.attn_impl,
    )

    try:
        for idx, bg_path in pending_backgrounds:
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

    rank0_print(rank, f"\n🎉 完成！结果保存在 {args.output_dir}")


if __name__ == "__main__":
    main()
