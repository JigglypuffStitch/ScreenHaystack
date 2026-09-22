from __future__ import annotations

from common import (
    dist_init,
    rank0_print,
    dist_barrier,
    cleanup_dist,
    _get_pad_token_id,
)

import argparse
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from qwen_vl_utils import smart_resize


# ========== 默认配置 ==========
DEFAULT_MODEL_PATH = "HelloKKMe/GTA1-7B"

DEFAULT_BG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'background')
DEFAULT_ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/gta1')

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
        "gemini": {
            "path": os.path.join(icon_dir, "gemini_icon_40.png"),
            "width": 40,
            "height": 40,
            "prompt": "the Gemini icon, a red diamond shape inside a square yellow border",
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
            "prompt": "the clock icon, a white circle with black hour hand pointing to 3 o'clock and minute hand pointing to 12 o'clock, with text 'clock' on the right",
            "name": "clock",
        },
    }


def extract_coordinates(raw_string: str) -> Optional[Tuple[float, float]]:
    """
    优先解析 (x,y)，失败时退化为提取前两个数字。
    GTA1 这里按 pixel coordinate 处理，不再把 <=1005 的坐标默认视为 0-1000 归一化坐标。
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
    采用上面那个快脚本的写法：
    text + image placeholder + text，然后 processor(text=[...], images=[...]) 做 batched processing。
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


# -------------------------
# Model: 按上面那个快脚本的调用方式
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
    和上面快脚本保持一致：
    - AutoConfig 检查 model_type
    - torch_dtype="auto"
    - trust_remote_code=True
    - low_cpu_mem_usage=True
    - device_map={"": str(device)}
    - processor 直接接收 min_pixels / max_pixels
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
        device_map={"": str(device)},  # 整模型放到本 rank GPU
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
    batch_items 中每个 item 包含：
    r, c, image, bbox
    其中 image 已经 resize 到 GTA1 实际看到的尺寸。
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

    # 因为 batch 内 prompt 完全相同，所以直接用 padded prompt length 切掉输入部分。
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

            # GTA1 输出的是 resized image 坐标，映射回原始背景图坐标。
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
        rank0_print(rank, "⚠️ 背景高度小于图标高度，无法放置图标，跳过")
        return

    grid_w = W / COLS
    grid_h = H / ROWS

    # 不再读 processor.image_processor.min_pixels/max_pixels；
    # 直接用 args.min_pixels / args.max_pixels，和上面快脚本的参数设置保持一致。
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

    rank0_print(rank, f"\n🖼️ 背景 [{bg_index}]: {os.path.basename(bg_path)} ({W}x{H})")
    rank0_print(rank, f"📐 网格划分: {COLS} 列 x {ROWS} 行，图标尺寸 {icon_w}x{icon_h}px")
    rank0_print(rank, f"🔁 GTA1 输入尺寸: {resized_width}x{resized_height}")
    rank0_print(rank, f"🔁 坐标缩放比例: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
    rank0_print(rank, f"🚀 batch inference: batch_size={batch_size}, world_size={world_size}")

    total_cells = ROWS * COLS
    my_cell_indices = [i for i in range(total_cells) if (i % world_size) == rank]

    # rank_matrix 只填本 rank 负责的 cell，其余为 NaN。
    rank_matrix = np.full((ROWS, COLS), np.nan, dtype=np.float32)

    # 每个 rank 写自己的 jsonl shard，避免多进程同时写同一个文件造成内容交错。
    # rank0 会在所有 rank 完成后合并为一个完整 jsonl。
    rank_jsonl_path = os.path.join(
        output_dir,
        f"{icon_info['name']}_gta1_outputs_{bg_index}.rank{rank}.jsonl",
    )
    if os.path.exists(rank_jsonl_path):
        os.remove(rank_jsonl_path)

    start_time = time.time()
    batch_items: List[Dict[str, Any]] = []

    pbar = tqdm(
        my_cell_indices,
        desc=f"BG{bg_index}-rank{rank}",
        disable=(rank != 0),
    )

    def flush_batch() -> None:
        nonlocal batch_items
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

            jsonl_rows.append(
                {
                    "background_index": bg_index,
                    "background_file": os.path.basename(bg_path),
                    "background_path": bg_path,
                    "icon_name": icon_info["name"],
                    "icon_prompt": icon_info["prompt"],
                    "rank": rank,
                    "world_size": world_size,
                    "cell_idx": res["cell_idx"],
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
                    "bbox_original": list(res["bbox"]),
                    "model_output": res["raw"],
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

        batch_items = []

    for cell_idx in pbar:
        r = cell_idx // COLS
        c = cell_idx % COLS

        px = int(c * grid_w + (grid_w - icon_w) / 2)
        py = int(r * grid_h + (grid_h - icon_h) / 2)

        px = max(0, min(px, W - icon_w))
        py = max(0, min(py, H - icon_h))

        x_min, x_max = px, px + icon_w
        y_min, y_max = py, py + icon_h

        temp_img = base_bg.copy()
        temp_img.paste(icon, (px, py), icon)

        # 手动 resize 到 smart_resize 后的尺寸。
        # 这样 prompt 里的 resolution 和模型看到的图像尺寸一致。
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
        f"{icon_info['name']}_gta1_accuracy_relative_matrix_{bg_index}.rank{rank}.npy",
    )
    np.save(rank_path, rank_matrix)

    elapsed = time.time() - start_time
    rank0_print(rank, f"⏱️ 背景 {bg_index} rank0 shard 用时: {elapsed/60:.2f} min")

    # 等所有 rank 写完各自的 rank_matrix
    dist_barrier(dist_enabled)

    # rank0 合并所有 rank 的矩阵，并保存最终 npy + heatmap
    if rank == 0:
        merged = np.full((ROWS, COLS), np.nan, dtype=np.float32)

        for rr in range(world_size):
            p = os.path.join(
                output_dir,
                f"{icon_info['name']}_gta1_accuracy_relative_matrix_{bg_index}.rank{rr}.npy",
            )
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

        npy_filename = f"{icon_info['name']}_gta1_accuracy_relative_matrix_{bg_index}.npy"
        npy_path = os.path.join(output_dir, npy_filename)
        np.save(npy_path, merged)
        print(f"💾 已保存合并矩阵: {npy_path}")

        png_filename = f"{icon_info['name']}_gta1_blind_spot_relative_{bg_index}.png"
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

        print(f"🖼️ 已保存 heatmap: {png_path}")

        merged_jsonl_filename = f"{icon_info['name']}_gta1_outputs_{bg_index}.jsonl"
        merged_jsonl_path = os.path.join(output_dir, merged_jsonl_filename)
        with open(merged_jsonl_path, "w", encoding="utf-8") as fout:
            for rr in range(world_size):
                shard_path = os.path.join(
                    output_dir,
                    f"{icon_info['name']}_gta1_outputs_{bg_index}.rank{rr}.jsonl",
                )
                if not os.path.exists(shard_path):
                    print(f"⚠️ 缺少 rank jsonl 文件: {shard_path}")
                    continue
                with open(shard_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)

        print(f"📝 已保存 model outputs jsonl: {merged_jsonl_path}")

    dist_barrier(dist_enabled)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dist_enabled, rank, world_size, local_rank, device = dist_init()

    parser = argparse.ArgumentParser()
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

    rank0_print(rank, f"[Rank {rank}] local_rank={local_rank}, device={device}, world_size={world_size}")

    model, processor = load_model(
        model_path=args.model_path,
        device=device,
        rank=rank,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        attn_impl=args.attn_impl,
    )

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
        f"{args.icon_name}，尺寸 {icon_info['width']}x{icon_info['height']}"
    )
    rank0_print(rank, "=" * 60)

    for idx, bg_file in enumerate(bg_files, start=1):
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

    rank0_print(rank, f"\n🎉 完成！结果保存在 {args.output_dir}")
    cleanup_dist(dist_enabled)


if __name__ == "__main__":
    main()
