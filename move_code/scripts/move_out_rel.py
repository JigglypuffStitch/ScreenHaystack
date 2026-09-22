
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from PIL import Image, ImageChops, ImageDraw, ImageOps
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None


# ----------------------------
# Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

_COORD_RE = re.compile(
    r"\(?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?"
)

BBoxRel = Tuple[float, float, float, float]
BBoxPix = Tuple[float, float, float, float]


# -------------------------
# Distributed helpers
# -------------------------
def dist_init() -> Tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, world_size, local_rank, device

    rank = 0
    world_size = 1
    local_rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return False, rank, world_size, local_rank, device


def rank0_print(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs)


def dist_barrier(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.barrier()


def dist_cleanup(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------
# Basic helpers
# -------------------------
def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(v)))


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def ensure_trailing_newline(path: str) -> None:
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "ab") as f:
                f.write(b"\n")
    except Exception:
        pass


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def load_json_records(json_file_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    json_file_paths = [f for f in os.listdir(json_file_dir) if f.endswith((".json", ".jsonl"))]
    json_file_paths.sort()

    for fn in json_file_paths:
        full_path = os.path.join(json_file_dir, fn)
        if fn.endswith(".jsonl"):
            records.extend(iter_jsonl(full_path))
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend([x for x in data if isinstance(x, dict)])
            elif isinstance(data, dict):
                records.append(data)
    return records


def load_records_from_path(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if os.path.isdir(path):
        return load_json_records(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        return list(iter_jsonl(path))
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    raise ValueError(f"--subset_path must be a .json/.jsonl file or a directory, got: {path}")


def pick_instruction(sample: Dict[str, Any], language: str) -> str:
    if language == "cn":
        for k in ("prompt_to_evaluate_cn", "instruction_cn"):
            if sample.get(k):
                return str(sample[k])
    for k in ("prompt_to_evaluate", "instruction"):
        if sample.get(k) is not None:
            return str(sample[k])
    for k in ("query", "text", "prompt", "desc"):
        if sample.get(k) is not None:
            return str(sample[k])
    return ""


def resolve_image_path(sample: Dict[str, Any], base_image_dir: str) -> str:
    img_path = sample.get("img_path", None)
    img_fn = sample.get("img_filename", None)
    if img_path:
        img_path = str(img_path)
        if (not os.path.isabs(img_path)) and base_image_dir:
            return os.path.join(base_image_dir, img_path)
        return img_path
    if img_fn:
        return os.path.join(base_image_dir, str(img_fn))
    return ""


def stable_key_from_screenspot(sample: Dict[str, Any]) -> str:
    k0 = sample.get("key", None)
    if k0 is not None and str(k0) != "":
        return str(k0)
    rec_id = sample.get("id", None)
    if rec_id is not None and str(rec_id) != "":
        return f"id:{str(rec_id)}"

    payload = {
        "img_path": sample.get("img_path", None),
        "img_filename": sample.get("img_filename", None),
        "bbox": sample.get("bbox", None),
        "prompt_to_evaluate": sample.get("prompt_to_evaluate", None),
        "prompt_to_evaluate_cn": sample.get("prompt_to_evaluate_cn", None),
        "instruction": sample.get("instruction", None),
        "instruction_cn": sample.get("instruction_cn", None),
        "gt_type": sample.get("gt_type", None),
        "platform": sample.get("platform", None),
        "application": sample.get("application", None),
        "ui_type": sample.get("ui_type", None),
    }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return f"fp:{h}"


def load_resume_done_keys(out_jsonl_rank: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_jsonl_rank) or os.path.getsize(out_jsonl_rank) == 0:
        return done
    for obj in iter_jsonl(out_jsonl_rank):
        k = obj.get("key", None)
        if k:
            done.add(str(k))
    return done


def _get_pad_token_id(processor: Any) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return 0
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return 0


# -------------------------
# Output parsing
# -------------------------
def normalize_coord_output_mode(mode: str) -> str:
    """Normalize user-facing coordinate mode aliases.

    Supported modes:
      - auto:     legacy heuristic.
      - pixel:    absolute pixel coordinates in the current model-input image.
                  Use this for HelloKKMe/GTA1-7B / GTA1-style models.
      - norm1000: coordinates normalized to [0,1000].
      - rel01:    coordinates normalized to [0,1].
    """
    mode = str(mode or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "pixel": "pixel",
        "pixels": "pixel",
        "pix": "pixel",
        "abs": "pixel",
        "absolute": "pixel",
        "absolute_pixel": "pixel",
        "absolute_pixels": "pixel",
        "norm1000": "norm1000",
        "norm_1000": "norm1000",
        "normalized1000": "norm1000",
        "qwen": "norm1000",
        "rel01": "rel01",
        "relative": "rel01",
        "relative01": "rel01",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported --coord_output_mode={mode!r}. "
            "Use one of: auto, pixel, norm1000, rel01."
        )
    return aliases[mode]


def extract_coordinates(
    raw: str,
    W: int,
    H: int,
    *,
    coord_output_mode: str = "auto",
) -> Tuple[int, int, bool, Dict[str, Any]]:
    """
    Return (x_pixel, y_pixel, parsed_ok, debug_info).

    coord_output_mode controls how the parsed coordinate pair is interpreted:
      - pixel:    raw pair is already absolute pixels in the current model input.
                  This is the safe choice for HelloKKMe/GTA1-7B.
      - norm1000: raw pair is normalized to [0,1000].
      - rel01:    raw pair is normalized to [0,1].
      - auto:     legacy heuristic, kept for backward compatibility.

    Important: auto is ambiguous for large GUI screenshots. For example, a GTA1
    pixel output like (500,300) also falls in [0,1000], so auto may treat it as
    norm1000. Use --coord_output_mode pixel for absolute-coordinate models.
    """
    mode_req = normalize_coord_output_mode(coord_output_mode)

    m = _COORD_RE.search(raw or "")
    if not m:
        return 0, 0, False, {
            "ok": False,
            "reason": "regex_not_found",
            "raw": raw,
            "coord_output_mode": mode_req,
        }

    try:
        xr = float(m.group(1))
        yr = float(m.group(2))
    except Exception as e:
        return 0, 0, False, {
            "ok": False,
            "reason": "float_parse_failed",
            "error": str(e),
            "raw": raw,
            "coord_output_mode": mode_req,
        }

    mode = mode_req
    if mode_req == "pixel":
        x_pix = xr
        y_pix = yr
    elif mode_req == "norm1000":
        x_pix = xr / 1000.0 * float(W)
        y_pix = yr / 1000.0 * float(H)
    elif mode_req == "rel01":
        x_pix = xr * float(W)
        y_pix = yr * float(H)
    else:
        # Legacy heuristic. Keep for old runs, but prefer explicit modes.
        mode = "pixel"
        x_pix = xr
        y_pix = yr

        if 0.0 <= xr <= 1.0 and 0.0 <= yr <= 1.0:
            mode = "rel01"
            x_pix = xr * float(W)
            y_pix = yr * float(H)
        elif 0.0 <= xr <= 1000.0 and 0.0 <= yr <= 1000.0:
            if (W > 1000) or (H > 1000) or (xr > W) or (yr > H):
                mode = "norm1000"
                x_pix = xr / 1000.0 * float(W)
                y_pix = yr / 1000.0 * float(H)

    xi_before = int(round(x_pix))
    yi_before = int(round(y_pix))
    in_bounds_before_clamp = (0 <= xi_before < int(W)) and (0 <= yi_before < int(H))

    xi = clamp_int(xi_before, 0, max(0, W - 1))
    yi = clamp_int(yi_before, 0, max(0, H - 1))

    dbg = {
        "ok": True,
        "mode_requested": mode_req,
        "mode_used": mode,
        "raw_pair": [float(xr), float(yr)],
        "W": int(W),
        "H": int(H),
        "pixel_float": [float(x_pix), float(y_pix)],
        "pixel_int_before_clamp": [int(xi_before), int(yi_before)],
        "pixel_int_clamped": [int(xi), int(yi)],
        "in_bounds_before_clamp": bool(in_bounds_before_clamp),
    }
    if mode == "norm1000":
        dbg["norm1000_pair"] = [float(xr), float(yr)]
    if mode == "rel01":
        dbg["rel01_pair"] = [float(xr), float(yr)]
    return xi, yi, True, dbg


def extract_coordinates_auto(raw: str, W: int, H: int) -> Tuple[int, int, bool, Dict[str, Any]]:
    """Backward-compatible wrapper for old code paths."""
    return extract_coordinates(raw, W, H, coord_output_mode="auto")


# -------------------------
# Relative bbox / region helpers
# -------------------------
def normalize_rel_box(box: Any, *, name: str) -> BBoxRel:
    """
    Interpret every box as relative [0,1] xyxy.

    No absolute-coordinate detection is allowed here.
    """
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError(f"Invalid {name}: expected 4 values, got {box}")

    x1, y1, x2, y2 = map(float, box)
    vals = [x1, y1, x2, y2]
    if any((v < 0.0 or v > 1.0) for v in vals):
        raise ValueError(f"{name} must use relative coordinates in [0,1], got {box}")

    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    if x_hi <= x_lo or y_hi <= y_lo:
        raise ValueError(f"{name} has non-positive area, got {box}")

    return (x_lo, y_lo, x_hi, y_hi)


def bbox_rel_to_pixel_xyxy(b: BBoxRel, W: int, H: int) -> BBoxPix:
    x1, y1, x2, y2 = b
    return (x1 * W, y1 * H, x2 * W, y2 * H)


def normalize_sample_bbox_to_pixel_xyxy(box: Any, W: int, H: int) -> BBoxPix:
    """
    Convert the dataset sample bbox to pixel xyxy.

    Important:
    - Only --region_bbox_path is forced to use relative [0,1] coordinates.
    - sample["bbox"] is allowed to be the original dataset bbox, usually pixel xyxy.
    - For compatibility, if all four sample bbox values are in [0,1], it is treated
      as relative xyxy/xywh and converted to pixels.
    - If values are larger than 1, it is treated as pixel xyxy/xywh.

    Supported bbox forms:
      [x1, y1, x2, y2] when x2 >= x1 and y2 >= y1
      [x, y, w, h]     when x2 < x1 or y2 < y1
    """
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError(f"Invalid sample.bbox: expected 4 values, got {box}")

    a, b, c, d = map(float, box)

    is_rel = all(0.0 <= v <= 1.0 for v in (a, b, c, d))
    if is_rel:
        if c >= a and d >= b:
            x1, y1, x2, y2 = a * W, b * H, c * W, d * H
        else:
            x1, y1, x2, y2 = a * W, b * H, (a + c) * W, (b + d) * H
    else:
        if c >= a and d >= b:
            x1, y1, x2, y2 = a, b, c, d
        else:
            x1, y1, x2, y2 = a, b, a + c, b + d

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"sample.bbox has non-positive area after conversion, got {box}")

    return (float(x1), float(y1), float(x2), float(y2))


def bbox_pixel_to_rel_xyxy(b: BBoxPix, W: int, H: int) -> BBoxRel:
    x1, y1, x2, y2 = b
    return (
        float(x1) / float(W),
        float(y1) / float(H),
        float(x2) / float(W),
        float(y2) / float(H),
    )


def bbox_center_rel(b: BBoxRel) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_center_xy(b: BBoxPix) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def point_in_box_rel(x: float, y: float, box: BBoxRel) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def point_in_any_region_rect_rel(regions_rel: List[BBoxRel], x: float, y: float) -> bool:
    return any(point_in_box_rel(x, y, r) for r in regions_rel)


def load_region_boxes_rel01(path: str) -> List[BBoxRel]:
    """
    Load raw region boxes.

    Important:
    - All region coordinates are treated as relative [0,1].
    - There is no absolute-vs-relative detection.
    - There is no region merge.
    - Supported input rows:
        {"bbox": [x1, y1, x2, y2]}
        {"bounds": {"x0": ..., "y0": ..., "x1": ..., "y1": ...}}
    """
    boxes: List[BBoxRel] = []
    for obj in iter_jsonl(path):
        b = obj.get("bbox", None)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            boxes.append(normalize_rel_box(b, name="region.bbox"))
            continue

        bounds = obj.get("bounds", None)
        if isinstance(bounds, dict):
            try:
                b2 = [bounds["x0"], bounds["y0"], bounds["x1"], bounds["y1"]]
            except KeyError as e:
                raise ValueError(f"Missing key in region.bounds: {e}; obj={obj}") from e
            boxes.append(normalize_rel_box(b2, name="region.bounds"))
            continue

    return boxes


# -------------------------
# Pixel geometry used only for image shift/scoring/visualization
# -------------------------
def point_in_box(x: float, y: float, box_xyxy: BBoxPix) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def bbox_after_shift(gt_xyxy: BBoxPix, dx: int, dy: int) -> BBoxPix:
    x1, y1, x2, y2 = gt_xyxy
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _rect_rect_distance(a: BBoxPix, b: BBoxPix) -> float:
    """Euclidean distance between two axis-aligned rectangles; 0 if overlap/touch."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1

    dx = 0.0
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2

    dy = 0.0
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2

    return math.hypot(dx, dy)


def _bbox_intersects_any_region_rects(bbox_xyxy: BBoxPix, region_boxes: List[BBoxPix]) -> bool:
    return any(_rect_rect_distance(bbox_xyxy, r) <= 0.0 for r in region_boxes)


def _min_dist_rect_to_boxes(bbox_xyxy: BBoxPix, region_boxes: List[BBoxPix]) -> float:
    if not region_boxes:
        return float("inf")
    return min(_rect_rect_distance(bbox_xyxy, r) for r in region_boxes)


def _min_dist_rect_to_canvas_edges(bbox_xyxy: BBoxPix, W: int, H: int) -> float:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return float(min(x1, y1, float(W) - x2, float(H) - y2))


# -------------------------
# Aspect-ratio padding for model input only
# -------------------------
def pad_to_aspect_limit(
    img: Image.Image,
    *,
    max_ratio: float = 199.0,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Qwen smart_resize requires aspect ratio < 200.
    If the image is too skinny, pad the short side. This does not change the
    logical canvas used for scoring unless padding is actually applied to the
    model input; the returned pad is used to offset GT boxes and predictions.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return img, (0, 0, 0, 0)

    r = max(w, h) / float(min(w, h))
    if r < max_ratio:
        return img, (0, 0, 0, 0)

    max_dim = max(w, h)
    target_min = int(math.ceil(max_dim / max_ratio))

    if w >= h:
        target_h = max(h, target_min)
        pad_total = target_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=fill), (0, pad_top, 0, pad_bottom)

    target_w = max(w, target_min)
    pad_total = target_w - w
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=fill), (pad_left, 0, pad_right, 0)


def offset_bbox_for_padding(b: BBoxPix, pad: Tuple[int, int, int, int]) -> BBoxPix:
    pad_left, pad_top, _, _ = pad
    x1, y1, x2, y2 = b
    return (x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top)


# -------------------------
# Visualization
# -------------------------
def visualize_prediction_pixel(
    img: Image.Image,
    x: int,
    y: int,
    save_path: str,
    *,
    bbox_xyxy: Optional[BBoxPix] = None,
    region_boxes_xyxy: Optional[List[BBoxPix]] = None,
    parsed: bool = True,
) -> None:
    W, H = img.size
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    if region_boxes_xyxy:
        for rb in region_boxes_xyxy:
            rx1, ry1, rx2, ry2 = map(float, rb)
            rx1 = int(max(0, min(W - 1, round(rx1))))
            ry1 = int(max(0, min(H - 1, round(ry1))))
            rx2 = int(max(0, min(W - 1, round(rx2))))
            ry2 = int(max(0, min(H - 1, round(ry2))))
            if rx2 > rx1 and ry2 > ry1:
                draw.rectangle((rx1, ry1, rx2, ry2), outline="deepskyblue", width=3)
                draw.rectangle((rx1, ry1, rx2, ry2), fill=(0, 191, 255, 30))

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        x1 = int(max(0, min(W - 1, round(x1))))
        y1 = int(max(0, min(H - 1, round(y1))))
        x2 = int(max(0, min(W - 1, round(x2))))
        y2 = int(max(0, min(H - 1, round(y2))))
        if x2 > x1 and y2 > y1:
            draw.rectangle((x1, y1, x2, y2), outline="red", width=6)
            draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 40))

    if parsed:
        x = int(max(0, min(W - 1, x)))
        y = int(max(0, min(H - 1, y)))
        r = max(8, int(round(min(W, H) * 0.01)))
        draw.ellipse((x - r, y - r, x + r, y + r), outline="lime", fill=(0, 255, 0, 90), width=5)
        cross_len = r
        draw.line((x - cross_len, y, x + cross_len, y), fill="lime", width=5)
        draw.line((x, y - cross_len, x, y + cross_len), fill="lime", width=5)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    canvas.save(save_path)


# -------------------------
# Shift-only transform
# -------------------------
def shift_image_on_canvas(img: Image.Image, dx: int, dy: int) -> Image.Image:
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return img.copy()
    return ImageChops.offset(img, dx, dy)


def filled_pixels_for_shift(W: int, H: int, dx: int, dy: int) -> int:
    adx = abs(int(dx))
    ady = abs(int(dy))
    return int(adx * H + ady * W - adx * ady)


def compute_change_budget_pixels(W: int, H: int, max_change_pct: float) -> int:
    return int(math.floor(float(W) * float(H) * float(max_change_pct) / 100.0 + 1e-9))


def _ceil_int(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def _floor_int(x: float) -> int:
    return int(math.floor(x + 1e-9))


def _bbox_inside_canvas_shift_ranges(
    gt_xyxy: BBoxPix,
    W: int,
    H: int,
    edge_margin_px: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1, x2, y2 = map(float, gt_xyxy)
    m = float(edge_margin_px)
    dx_lo = _ceil_int(m - x1)
    dx_hi = _floor_int(float(W) - m - x2)
    dy_lo = _ceil_int(m - y1)
    dy_hi = _floor_int(float(H) - m - y2)
    return (dx_lo, dx_hi), (dy_lo, dy_hi)


def _rng_for_sample_run(sample_key: str, run_i: int) -> random.Random:
    h = hashlib.md5(f"{sample_key}::run{run_i}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:8], 16))


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(v)))


def _candidate_neighborhood(
    dx0: int,
    dy0: int,
    dx_rng: Tuple[int, int],
    dy_rng: Tuple[int, int],
) -> List[Tuple[int, int]]:
    dx_lo, dx_hi = dx_rng
    dy_lo, dy_hi = dy_rng
    steps = [0, 1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    out: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for s in steps:
        for ax in (-s, 0, s):
            for ay in (-s, 0, s):
                dx = _clamp(dx0 + ax, dx_lo, dx_hi)
                dy = _clamp(dy0 + ay, dy_lo, dy_hi)
                if (dx, dy) not in seen:
                    seen.add((dx, dy))
                    out.append((dx, dy))
    return out


def _anchors_for_style(style: str, W: int, H: int, cx: float, cy: float, edge_margin_px: float) -> List[Tuple[float, float]]:
    mx = max(float(edge_margin_px), 0.08 * float(W))
    my = max(float(edge_margin_px), 0.08 * float(H))
    mx = min(mx, max(0.0, (float(W) - 1.0) * 0.49))
    my = min(my, max(0.0, (float(H) - 1.0) * 0.49))

    left_x = mx
    right_x = float(W) - 1.0 - mx
    top_y = my
    bottom_y = float(H) - 1.0 - my

    cx_in = min(max(float(cx), left_x), right_x)
    cy_in = min(max(float(cy), top_y), bottom_y)

    if style == "left_far":
        return [(left_x, cy_in), (left_x, top_y), (left_x, bottom_y)]
    if style == "right_far":
        return [(right_x, cy_in), (right_x, top_y), (right_x, bottom_y)]
    if style == "up_far":
        return [(cx_in, top_y), (left_x, top_y), (right_x, top_y)]
    if style == "down_far":
        return [(cx_in, bottom_y), (left_x, bottom_y), (right_x, bottom_y)]
    return [(left_x, top_y), (right_x, top_y), (left_x, bottom_y), (right_x, bottom_y)]


def _pick_best_shift_move_out(
    *,
    sample_key: str,
    run_i: int,
    style: str,
    gt_xyxy: BBoxPix,
    region_boxes_px: List[BBoxPix],
    W: int,
    H: int,
    budget_px: int,
    away_margin_px: float,
    edge_margin_px: float,
    used_shifts: Set[Tuple[int, int]],
    random_trials: int,
    allow_budget_relax: bool,
    allow_margin_relax: bool,
    allow_edge_relax: bool,
) -> Optional[Dict[str, Any]]:
    """
    Find a shift so the shifted bbox is outside raw regions.

    Region boxes are raw boxes converted from rel01 to pixels only for geometric
    shift constraints. Region boxes are used without geometry transforms.
    """
    dx_rng, dy_rng = _bbox_inside_canvas_shift_ranges(gt_xyxy, W=W, H=H, edge_margin_px=edge_margin_px)
    dx_lo, dx_hi = dx_rng
    dy_lo, dy_hi = dy_rng
    if dx_hi < dx_lo or dy_hi < dy_lo:
        return None

    cx, cy = bbox_center_xy(gt_xyxy)
    rng = _rng_for_sample_run(sample_key, run_i)
    cand_pairs: List[Tuple[int, int]] = []

    for tx, ty in _anchors_for_style(style, W, H, cx, cy, edge_margin_px=edge_margin_px):
        dx0 = _clamp(int(round(tx - cx)), dx_lo, dx_hi)
        dy0 = _clamp(int(round(ty - cy)), dy_lo, dy_hi)
        cand_pairs.extend(_candidate_neighborhood(dx0, dy0, dx_rng, dy_rng))

    extremes = [
        (dx_lo, 0),
        (dx_hi, 0),
        (0, dy_lo),
        (0, dy_hi),
        (dx_lo, dy_lo),
        (dx_lo, dy_hi),
        (dx_hi, dy_lo),
        (dx_hi, dy_hi),
    ]
    for dx0, dy0 in extremes:
        cand_pairs.extend(_candidate_neighborhood(_clamp(dx0, dx_lo, dx_hi), _clamp(dy0, dy_lo, dy_hi), dx_rng, dy_rng))

    for _ in range(int(max(0, random_trials))):
        cand_pairs.append((rng.randint(dx_lo, dx_hi), rng.randint(dy_lo, dy_hi)))

    dedup: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for p in cand_pairs:
        if p not in seen:
            seen.add(p)
            dedup.append(p)

    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[Any, ...]] = None

    for dx, dy in dedup:
        dx = int(dx)
        dy = int(dy)
        if (dx, dy) in used_shifts:
            continue
        if dx == 0 and dy == 0:
            continue

        affected = filled_pixels_for_shift(W, H, dx, dy)
        if (not allow_budget_relax) and affected > budget_px:
            continue

        bbox_t = bbox_after_shift(gt_xyxy, dx, dy)
        edge_d = _min_dist_rect_to_canvas_edges(bbox_t, W=W, H=H)
        if (not allow_edge_relax) and edge_d < float(edge_margin_px):
            continue

        if _bbox_intersects_any_region_rects(bbox_t, region_boxes_px):
            continue

        mind = _min_dist_rect_to_boxes(bbox_t, region_boxes_px)
        if (not allow_margin_relax) and mind < float(away_margin_px):
            continue

        move = abs(dx) + abs(dy)
        if allow_margin_relax:
            deficit = max(0.0, float(away_margin_px) - float(mind))
            key = (float(deficit), -float(mind), -float(edge_d), int(affected), int(move), abs(dx), abs(dy))
        else:
            key = (-float(mind), -float(edge_d), int(affected), int(move), abs(dx), abs(dy))

        if best is None or key < best_key:
            best = {
                "dx": dx,
                "dy": dy,
                "filled_px": int(affected),
                "min_dist_to_regions": float(mind),
                "min_dist_to_canvas_edge": float(edge_d),
            }
            best_key = key

    return best


def build_multi_run_shift_plans_move_out_of_regions(
    *,
    sample_key: str,
    gt_xyxy: BBoxPix,
    region_boxes_px: List[BBoxPix],
    W: int,
    H: int,
    budget_px: int,
    away_margin_px: float,
    edge_margin_px: float,
    num_runs: int,
    random_trials: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    styles_base = ["left_far", "right_far", "up_far", "down_far", "diag_far"]
    styles = [styles_base[i % len(styles_base)] for i in range(num_runs)]

    min_dist0 = _min_dist_rect_to_boxes(gt_xyxy, region_boxes_px)
    edge_dist0 = _min_dist_rect_to_canvas_edges(gt_xyxy, W=W, H=H)

    debug: Dict[str, Any] = {
        "budget_px": int(budget_px),
        "bbox_intersects_any_raw_region_at_dxdy0": bool(_bbox_intersects_any_region_rects(gt_xyxy, region_boxes_px)),
        "min_dist_to_raw_regions_at_dxdy0": float(min_dist0),
        "min_dist_to_canvas_edge_at_dxdy0": float(edge_dist0),
        "num_raw_regions": int(len(region_boxes_px)),
        "away_margin_px": float(away_margin_px),
        "edge_margin_px": float(edge_margin_px),
        "styles": list(styles),
        "rule": (
            "shift-only fixed canvas; raw regions only; "
            "membership is selected by bbox center in relative [0,1] coordinates."
        ),
    }

    plans: List[Dict[str, Any]] = []
    used_shifts: Set[Tuple[int, int]] = set()

    for run_i, style in enumerate(styles, 1):
        relax_margin = False
        relax_budget = False
        relax_edge = False

        best = _pick_best_shift_move_out(
            sample_key=sample_key,
            run_i=run_i,
            style=style,
            gt_xyxy=gt_xyxy,
            region_boxes_px=region_boxes_px,
            W=W,
            H=H,
            budget_px=budget_px,
            away_margin_px=away_margin_px,
            edge_margin_px=edge_margin_px,
            used_shifts=used_shifts,
            random_trials=random_trials,
            allow_budget_relax=False,
            allow_margin_relax=False,
            allow_edge_relax=False,
        )

        if best is None:
            best = _pick_best_shift_move_out(
                sample_key=sample_key,
                run_i=run_i,
                style=style,
                gt_xyxy=gt_xyxy,
                region_boxes_px=region_boxes_px,
                W=W,
                H=H,
                budget_px=budget_px,
                away_margin_px=away_margin_px,
                edge_margin_px=edge_margin_px,
                used_shifts=used_shifts,
                random_trials=random_trials * 2,
                allow_budget_relax=False,
                allow_margin_relax=True,
                allow_edge_relax=False,
            )
            relax_margin = True

        if best is None:
            best = _pick_best_shift_move_out(
                sample_key=sample_key,
                run_i=run_i,
                style=style,
                gt_xyxy=gt_xyxy,
                region_boxes_px=region_boxes_px,
                W=W,
                H=H,
                budget_px=budget_px,
                away_margin_px=away_margin_px,
                edge_margin_px=edge_margin_px,
                used_shifts=used_shifts,
                random_trials=random_trials * 3,
                allow_budget_relax=True,
                allow_margin_relax=False,
                allow_edge_relax=False,
            )
            relax_budget = True
            relax_margin = False

        if best is None:
            best = _pick_best_shift_move_out(
                sample_key=sample_key,
                run_i=run_i,
                style=style,
                gt_xyxy=gt_xyxy,
                region_boxes_px=region_boxes_px,
                W=W,
                H=H,
                budget_px=budget_px,
                away_margin_px=away_margin_px,
                edge_margin_px=edge_margin_px,
                used_shifts=used_shifts,
                random_trials=random_trials * 4,
                allow_budget_relax=True,
                allow_margin_relax=True,
                allow_edge_relax=False,
            )
            relax_budget = True
            relax_margin = True

        if best is None:
            best = _pick_best_shift_move_out(
                sample_key=sample_key,
                run_i=run_i,
                style=style,
                gt_xyxy=gt_xyxy,
                region_boxes_px=region_boxes_px,
                W=W,
                H=H,
                budget_px=budget_px,
                away_margin_px=away_margin_px,
                edge_margin_px=edge_margin_px,
                used_shifts=used_shifts,
                random_trials=random_trials * 5,
                allow_budget_relax=True,
                allow_margin_relax=True,
                allow_edge_relax=True,
            )
            relax_budget = True
            relax_margin = True
            relax_edge = True

        if best is None:
            plans.append(
                {
                    "run": int(run_i),
                    "style": str(style),
                    "dx": 0,
                    "dy": 0,
                    "filled_px": 0,
                    "filled_ratio": 0.0,
                    "move_out_failed": True,
                    "reason": "no_feasible_shift_to_satisfy_constraints",
                    "relax_margin": True,
                    "relax_budget": True,
                    "relax_edge": True,
                    "min_dist_to_regions": float(min_dist0),
                    "min_dist_to_canvas_edge": float(edge_dist0),
                }
            )
            used_shifts.add((0, 0))
            continue

        dx = int(best["dx"])
        dy = int(best["dy"])
        used_shifts.add((dx, dy))
        plans.append(
            {
                "run": int(run_i),
                "style": str(style),
                "dx": dx,
                "dy": dy,
                "filled_px": int(best["filled_px"]),
                "filled_ratio": 0.0 if (W * H) == 0 else float(best["filled_px"]) / float(W * H),
                "move_out_failed": False,
                "relax_margin": bool(relax_margin),
                "relax_budget": bool(relax_budget),
                "relax_edge": bool(relax_edge),
                "min_dist_to_regions": float(best["min_dist_to_regions"]),
                "min_dist_to_canvas_edge": float(best["min_dist_to_canvas_edge"]),
            }
        )

    debug["used_shifts_count"] = int(len(used_shifts))
    return plans, debug


# -------------------------
# Stats / merge
# -------------------------
_VALID_CORRECTNESS = {"correct", "wrong", "wrong_format", "skipped"}


def normalize_correctness(v: Any) -> str:
    s = str(v) if v is not None else "wrong_format"
    return s if s in _VALID_CORRECTNESS else "wrong_format"


def majority_vote_threshold(num_runs: int) -> int:
    num_runs = int(num_runs)
    return 0 if num_runs <= 0 else (num_runs // 2) + 1


def summarize_vote_correctness(correctnesses: List[str]) -> Dict[str, Any]:
    normalized = [normalize_correctness(c) for c in correctnesses]
    num_runs = len(normalized)
    threshold = majority_vote_threshold(num_runs)
    counts = Counter(normalized)
    num_correct = int(counts["correct"])

    if num_runs == 0 or counts["skipped"] == num_runs:
        overall = "skipped"
    elif num_correct >= threshold:
        overall = "correct"
    elif num_correct == 0 and counts["wrong"] == 0 and counts["wrong_format"] > 0:
        overall = "wrong_format"
    else:
        overall = "wrong"

    return {
        "correctness": overall,
        "num_runs": int(num_runs),
        "vote_threshold": int(threshold),
        "num_correct": num_correct,
        "num_wrong": int(counts["wrong"]),
        "num_wrong_format": int(counts["wrong_format"]),
        "num_skipped": int(counts["skipped"]),
    }


@dataclass
class Stats:
    total: int = 0
    scored: int = 0
    correct: int = 0
    wrong: int = 0
    wrong_format: int = 0
    skipped: int = 0

    def add(self, correctness: str) -> None:
        correctness = normalize_correctness(correctness)
        self.total += 1
        if correctness == "skipped":
            self.skipped += 1
            return
        self.scored += 1
        if correctness == "correct":
            self.correct += 1
        elif correctness == "wrong":
            self.wrong += 1
        else:
            self.wrong_format += 1

    @property
    def acc(self) -> float:
        return 0.0 if self.scored == 0 else self.correct / self.scored


def stats_to_metrics(s: Stats) -> Dict[str, Any]:
    return {
        "num_total": s.total,
        "num_scored": s.scored,
        "num_skipped": s.skipped,
        "num_correct": s.correct,
        "num_wrong": s.wrong,
        "num_wrong_format": s.wrong_format,
        "acc": s.acc,
    }


def mean_and_variance(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = float(sum(vals) / len(vals))
    var = float(sum((x - mean) * (x - mean) for x in vals) / len(vals))
    return mean, var


def is_transformed_record(obj: Dict[str, Any]) -> bool:
    trials = obj.get("multi_run_trials", None)
    if isinstance(trials, dict):
        for run_v in trials.values():
            if not isinstance(run_v, dict):
                continue
            t = run_v.get("shift", None)
            if isinstance(t, dict):
                dx = safe_int(t.get("dx", 0), 0)
                dy = safe_int(t.get("dy", 0), 0)
                if dx != 0 or dy != 0:
                    return True
    return False


def merge_rank_outputs_single(
    *,
    output_path_final: str,
    output_path_rank_pattern: str,
    world_size: int,
    summary_path_final: str,
    summary_path_rank_pattern: str,
    transformed_output_path_final: str = "",
) -> Dict[str, Any]:
    merged_stats = Stats()
    transformed_stats = Stats()
    multi_run_merged_stats: List[Stats] = []
    multi_run_transformed_stats: List[Stats] = []
    transformed_count = 0
    lr_mode_counter = Counter()
    seen: Set[str] = set()

    os.makedirs(os.path.dirname(output_path_final) or ".", exist_ok=True)
    transformed_fout = None
    if transformed_output_path_final:
        os.makedirs(os.path.dirname(transformed_output_path_final) or ".", exist_ok=True)
        transformed_fout = open(transformed_output_path_final, "w", encoding="utf-8")

    with open(output_path_final, "w", encoding="utf-8") as fout:
        for r in range(world_size):
            p = output_path_rank_pattern.format(rank=r)
            if not os.path.exists(p):
                continue
            for obj in iter_jsonl(p):
                k = obj.get("key", None)
                if not k:
                    k = stable_key_from_screenspot(obj)
                    obj["key"] = k
                if k in seen:
                    continue
                seen.add(str(k))

                c = normalize_correctness(obj.get("correctness", "wrong_format"))
                merged_stats.add(c)

                multi_corr = obj.get("multi_run_correctnesses", None)
                if isinstance(multi_corr, list) and multi_corr:
                    while len(multi_run_merged_stats) < len(multi_corr):
                        multi_run_merged_stats.append(Stats())
                    for i, ci in enumerate(multi_corr):
                        multi_run_merged_stats[i].add(normalize_correctness(ci))

                if is_transformed_record(obj):
                    transformed_stats.add(c)
                    transformed_count += 1
                    if isinstance(multi_corr, list) and multi_corr:
                        while len(multi_run_transformed_stats) < len(multi_corr):
                            multi_run_transformed_stats.append(Stats())
                        for i, ci in enumerate(multi_corr):
                            multi_run_transformed_stats[i].add(normalize_correctness(ci))
                    if transformed_fout is not None:
                        transformed_fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                mode = obj.get("lr_mode", None)
                if mode:
                    lr_mode_counter[str(mode)] += 1

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if transformed_fout is not None:
        transformed_fout.close()

    merged_summary: Dict[str, Any] = {
        "metrics": {
            "overall": stats_to_metrics(merged_stats),
            "transformed_selected": stats_to_metrics(transformed_stats),
            "transformed_selected_baseline_before_transform": perfect_correct_metrics(transformed_count),
            "transformed_selected_count": int(transformed_count),
            "lr_mode_counts": dict(lr_mode_counter),
            "dedup_keys": len(seen),
        },
        "merge": {
            "world_size": world_size,
            "rank_files": [output_path_rank_pattern.format(rank=r) for r in range(world_size)],
        },
        "rank_summaries": [],
    }

    if multi_run_merged_stats:
        acc_runs = [s.acc for s in multi_run_merged_stats]
        acc_mean, acc_var = mean_and_variance(acc_runs)
        merged_summary["metrics"]["multi_run_overall"] = {
            "num_runs": len(multi_run_merged_stats),
            "vote_threshold": majority_vote_threshold(len(multi_run_merged_stats)),
            "per_run": [stats_to_metrics(s) for s in multi_run_merged_stats],
            "acc_runs": acc_runs,
            "acc_mean": acc_mean,
            "acc_variance": acc_var,
        }

    if multi_run_transformed_stats:
        acc_runs_t = [s.acc for s in multi_run_transformed_stats]
        acc_mean_t, acc_var_t = mean_and_variance(acc_runs_t)
        merged_summary["metrics"]["multi_run_transformed_selected"] = {
            "num_runs": len(multi_run_transformed_stats),
            "vote_threshold": majority_vote_threshold(len(multi_run_transformed_stats)),
            "per_run": [stats_to_metrics(s) for s in multi_run_transformed_stats],
            "acc_runs": acc_runs_t,
            "acc_mean": acc_mean_t,
            "acc_variance": acc_var_t,
        }

    for r in range(world_size):
        sp = summary_path_rank_pattern.format(rank=r)
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    merged_summary["rank_summaries"].append(json.load(f))
            except Exception:
                pass

    # Compute overall including skipped items using a baseline accuracy if available
    # Baseline accuracy is looked up from rank summaries' "transformed_selected_baseline_before_transform" if present.
    try:
        baseline_accs: List[float] = []
        for rs in merged_summary.get("rank_summaries", []):
            ma = rs.get("metrics", {}).get("transformed_selected_baseline_before_transform", None)
            if isinstance(ma, dict) and ("acc" in ma):
                try:
                    baseline_accs.append(float(ma["acc"]))
                except Exception:
                    pass
        if baseline_accs:
            baseline_acc = float(sum(baseline_accs) / len(baseline_accs))
        else:
            baseline_acc = float(merged_summary.get("metrics", {}).get("transformed_selected_baseline_before_transform", {}).get("acc", 1.0))

        denom = merged_stats.scored + merged_stats.skipped
        if denom > 0:
            adjusted_correct = float(merged_stats.correct) + float(merged_stats.skipped) * baseline_acc
            acc_including_skipped = adjusted_correct / float(denom)
        else:
            adjusted_correct = 0.0
            acc_including_skipped = 0.0

        merged_summary["metrics"]["overall_including_skipped_baseline"] = {
            "num_total": merged_stats.total,
            "num_scored_including_skipped": int(denom),
            "num_skipped": int(merged_stats.skipped),
            "num_correct_adjusted": adjusted_correct,
            "acc": acc_including_skipped,
            "baseline_acc_used": baseline_acc,
        }
    except Exception:
        pass

    os.makedirs(os.path.dirname(summary_path_final) or ".", exist_ok=True)
    with open(summary_path_final, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, indent=2, ensure_ascii=False)

    return merged_summary


# -------------------------
# Model helpers
# -------------------------
def pick_model_cls(model_type: str):
    if model_type == "qwen2_5_vl":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError("Qwen2_5_VLForConditionalGeneration is not available in this transformers install.")
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("Qwen3VLForConditionalGeneration is not available. Please upgrade transformers.")
        return Qwen3VLForConditionalGeneration
    raise RuntimeError(f"Unsupported model_type={model_type!r}. Expected qwen2_5_vl or qwen3_vl.")


def processor_from_pretrained_flexible(model_path: str, min_pixels: int, max_pixels: int):
    try:
        return AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
    except TypeError:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def run_vlm_once(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    instruction: str,
    max_new_tokens: int,
    pad_token_id: int,
) -> str:
    W, H = image.size
    text_prompt = SYSTEM_PROMPT.format(height=H, width=W) + "\n\nUser element description:\n" + instruction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    except TypeError:
        # Some processors expect positional images instead of keyword images.
        inputs = processor([prompt_text], [image], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    input_ids = inputs.get("input_ids", None)
    if input_ids is not None:
        generated_ids = generated_ids[:, input_ids.shape[1]:]

    out = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out.strip()


# -------------------------
# Main
# -------------------------
def main() -> None:
    enabled_dist, rank, world_size, local_rank, device = dist_init()

    ap = argparse.ArgumentParser()

    # ---- model ----
    ap.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--attn_impl", type=str, default="", choices=["", "flash_attention_2", "sdpa", "eager"])

    # ---- data ----
    ap.add_argument("--json_file_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/annotations")
    ap.add_argument("--base_image_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/images")
    ap.add_argument("--subset_path", type=str, default="")

    # ---- region ----
    ap.add_argument(
        "--region_bbox_path",
        type=str,
        required=True,
        help=(
            "Region bbox jsonl. All region coordinates are relative values in [0,1]. "
            "Supported rows: {'bbox':[x1,y1,x2,y2]} or "
            "{'bounds':{'x0':..., 'y0':..., 'x1':..., 'y1':...}}. "
            "No absolute-coordinate detection or region geometry transforms."
        ),
    )
    ap.add_argument(
        "--away_margin_px",
        type=float,
        default=100.0,
        help="After move-out, require shifted bbox to be at least this many pixels away from raw regions.",
    )
    ap.add_argument(
        "--canvas_edge_margin_px",
        type=float,
        default=400.0,
        help="After move-out, require shifted bbox to stay at least this many pixels away from image edges.",
    )
    ap.add_argument(
        "--random_trials",
        type=int,
        default=1024,
        help="Random shift candidates per run when searching for a feasible move-out shift.",
    )

    # ---- change budget ----
    ap.add_argument(
        "--max_change_pct",
        type=float,
        default=60.0,
        help="Max affected pixels percent of original canvas. affected=|dx|H+|dy|W-|dx||dy|.",
    )
    ap.add_argument("--num_crop_runs", type=int, default=5, help="Number of shift runs. Kept name for compatibility.")

    # ---- run control ----
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--summary_path", type=str, default="")
    ap.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    ap.add_argument("--gt_type", type=str, default="all", choices=["positive", "negative", "all"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument(
        "--coord_output_mode",
        type=str,
        default="pixel",
        help=(
            "Coordinate format of model output. "
            "Use 'pixel' for absolute-pixel models such as HelloKKMe/GTA1-7B; "
            "use 'norm1000' for Qwen-style [0,1000] outputs; "
            "use 'rel01' for [0,1] relative outputs; "
            "use 'auto' for the legacy heuristic."
        ),
    )

    # processor image scaling
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--max_pixels", type=int, default=4096 * 2160)

    # padding aspect-ratio fix
    ap.add_argument(
        "--aspect_ratio_limit",
        type=float,
        default=99.0,
        help="Pad short side if image is too skinny. Must be < 200 for Qwen smart_resize constraint.",
    )

    # visualization
    ap.add_argument("--vis_dir", type=str, default="")
    ap.add_argument("--vis_every", type=int, default=1)

    # streaming flush
    ap.add_argument("--flush_every", type=int, default=1)

    # resume + merge
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--no_merge", action="store_true")

    args = ap.parse_args()
    args.coord_output_mode = normalize_coord_output_mode(args.coord_output_mode)

    if not args.summary_path:
        args.summary_path = args.output_path + ".summary.json"
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.aspect_ratio_limit >= 200.0:
        raise ValueError("--aspect_ratio_limit must be < 200")
    if args.max_change_pct < 0.0 or args.max_change_pct > 100.0:
        raise ValueError("--max_change_pct must be in [0, 100]")
    if args.num_crop_runs < 1:
        raise ValueError("--num_crop_runs must be >= 1")
    if args.away_margin_px < 0:
        raise ValueError("--away_margin_px must be >= 0")
    if args.canvas_edge_margin_px < 0:
        raise ValueError("--canvas_edge_margin_px must be >= 0")

    output_path_rank = f"{args.output_path}.rank{rank}.jsonl"
    summary_path_rank = f"{args.summary_path}.rank{rank}.json"
    transformed_output_path = f"{args.output_path}.transformed.jsonl"
    transformed_output_path_rank = f"{args.output_path}.transformed.rank{rank}.jsonl"

    os.makedirs(os.path.dirname(output_path_rank) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_path_rank) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(transformed_output_path_rank) or ".", exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    resume = args.resume and (not args.no_resume)
    if (not args.resume) and (not args.no_resume):
        resume = True

    rank0_print(rank, "[Mode] center-only raw-region membership; region coords are always rel01; no abs detection.")
    rank0_print(rank, "[Region] --region_bbox_path is rel01 only; sample bbox may be pixel; raw region geometry is unchanged; bbox center decides membership.")
    rank0_print(rank, f"[MoveOut] away_margin_px={args.away_margin_px}; canvas_edge_margin_px={args.canvas_edge_margin_px}")
    rank0_print(rank, f"[Budget] max_change_pct={args.max_change_pct:.2f}%")
    rank0_print(rank, f"[MultiRun] num_runs={args.num_crop_runs}")
    rank0_print(rank, f"[Parse] coord_output_mode={args.coord_output_mode}")

    # ----------------------------
    # Load model / processor
    # ----------------------------
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt is None:
        raise RuntimeError("Missing model_type in config. Please check --model_path.")

    ModelCls = pick_model_cls(str(mt))
    model_kwargs = dict(torch_dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    model_kwargs["device_map"] = {"": str(device)} if device.type == "cuda" else {"": "cpu"}

    rank0_print(rank, f"[Rank {rank}] Loading model_type={mt} on {device} from {args.model_path} ...")
    model = ModelCls.from_pretrained(args.model_path, **model_kwargs)
    processor = processor_from_pretrained_flexible(args.model_path, args.min_pixels, args.max_pixels)
    model.eval()
    pad_token_id = _get_pad_token_id(processor)

    # ----------------------------
    # Load records / regions
    # ----------------------------
    if args.subset_path:
        records = load_records_from_path(args.subset_path)
        rank0_print(rank, f"[Data] subset_path mode: loaded {len(records)} records from {args.subset_path}")
    else:
        records = load_json_records(args.json_file_dir)
        rank0_print(rank, f"[Data] annotation dir mode: loaded {len(records)} records from {args.json_file_dir}")

    if args.gt_type != "all":
        records = [r for r in records if r.get("gt_type", "positive") == args.gt_type]
    if args.stride > 1:
        records = records[:: args.stride]
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    region_boxes_rel = load_region_boxes_rel01(args.region_bbox_path)
    if not region_boxes_rel:
        raise RuntimeError(f"No valid relative region boxes found in: {args.region_bbox_path}")

    my_indices = [i for i in range(len(records)) if (i % world_size) == rank]
    rank0_print(rank, f"[Data] kept records={len(records)}; shard={len(my_indices)}/{len(records)} world_size={world_size}")
    rank0_print(rank, f"[Region] loaded raw region boxes={len(region_boxes_rel)}")

    num_runs = int(args.num_crop_runs)
    stats_overall_runs: List[Stats] = [Stats() for _ in range(num_runs)]
    stats_transformed_selected_runs: List[Stats] = [Stats() for _ in range(num_runs)]
    stats_overall = Stats()
    stats_transformed_selected = Stats()
    lr_mode_counter = Counter()

    n_written = 0
    n_transformed_written = 0
    done_keys: Set[str] = set()
    out_mode = "w"
    if resume and os.path.exists(output_path_rank) and os.path.getsize(output_path_rank) > 0:
        done_keys = load_resume_done_keys(output_path_rank)
        ensure_trailing_newline(output_path_rank)
        out_mode = "a"
        rank0_print(rank, f"[Rank {rank}] Resume enabled: done_keys={len(done_keys)}")
    if out_mode == "a" and os.path.exists(transformed_output_path_rank) and os.path.getsize(transformed_output_path_rank) > 0:
        ensure_trailing_newline(transformed_output_path_rank)

    def write_one(fout, obj: Dict[str, Any]) -> None:
        nonlocal n_written
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n_written += 1
        if args.flush_every > 0 and (n_written % args.flush_every == 0):
            fout.flush()

    def write_one_transformed(fout, obj: Dict[str, Any]) -> None:
        nonlocal n_transformed_written
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n_transformed_written += 1
        if args.flush_every > 0 and (n_transformed_written % args.flush_every == 0):
            fout.flush()

    # ----------------------------
    # Main loop
    # ----------------------------
    pbar = tqdm(my_indices, desc=f"Eval(rank{rank})", disable=(rank != 0))
    fout_transformed = open(transformed_output_path_rank, out_mode, encoding="utf-8")

    try:
        with open(output_path_rank, out_mode, encoding="utf-8") as fout:
            for list_idx in pbar:
                sample = records[list_idx]
                key = stable_key_from_screenspot(sample)
                if resume and key in done_keys:
                    continue

                src_idx = safe_int(sample.get("idx", list_idx), list_idx)
                sample_gt_type = sample.get("gt_type", "positive")
                instruction = pick_instruction(sample, args.language)
                img_path = resolve_image_path(sample, args.base_image_dir)

                if (not img_path) or (not os.path.exists(img_path)):
                    out = {
                        "idx": src_idx,
                        "key": key,
                        "img_path": img_path if img_path else None,
                        "gt_type": sample_gt_type,
                        "prompt_to_evaluate": instruction,
                        "bbox_orig": sample.get("bbox", None),
                        "lr_mode": "skipped",
                        "correctness": "wrong_format",
                        "_debug": {"rank": rank, "reason": "missing_image", "list_idx": list_idx},
                    }
                    for s in stats_overall_runs:
                        s.add("wrong_format")
                    stats_overall.add("wrong_format")
                    write_one(fout, out)
                    if resume:
                        done_keys.add(key)
                    continue

                try:
                    img_rgb = Image.open(img_path).convert("RGB")
                except Exception as e:
                    out = {
                        "idx": src_idx,
                        "key": key,
                        "img_path": img_path,
                        "gt_type": sample_gt_type,
                        "prompt_to_evaluate": instruction,
                        "bbox_orig": sample.get("bbox", None),
                        "lr_mode": "skipped",
                        "correctness": "wrong_format",
                        "_debug": {"rank": rank, "reason": "image_open_failed", "error": str(e), "list_idx": list_idx},
                    }
                    for s in stats_overall_runs:
                        s.add("wrong_format")
                    stats_overall.add("wrong_format")
                    write_one(fout, out)
                    if resume:
                        done_keys.add(key)
                    continue

                W, H = img_rgb.size
                try:
                    gt_pix = normalize_sample_bbox_to_pixel_xyxy(sample.get("bbox", None), W, H)
                    gt_rel = bbox_pixel_to_rel_xyxy(gt_pix, W, H)
                except Exception as e:
                    out = {
                        "idx": src_idx,
                        "key": key,
                        "img_path": img_path,
                        "gt_type": sample_gt_type,
                        "prompt_to_evaluate": instruction,
                        "bbox_orig": sample.get("bbox", None),
                        "lr_mode": "skipped",
                        "correctness": "wrong_format",
                        "_debug": {
                            "rank": rank,
                            "reason": "invalid_sample_bbox",
                            "error": str(e),
                            "list_idx": list_idx,
                        },
                    }
                    for s in stats_overall_runs:
                        s.add("wrong_format")
                    stats_overall.add("wrong_format")
                    write_one(fout, out)
                    if resume:
                        done_keys.add(key)
                    continue

                cx_rel, cy_rel = bbox_center_rel(gt_rel)
                center_in_region = point_in_any_region_rect_rel(region_boxes_rel, cx_rel, cy_rel)

                region_boxes_px = [bbox_rel_to_pixel_xyxy(r, W, H) for r in region_boxes_rel]
                budget_px = compute_change_budget_pixels(W, H, args.max_change_pct)

                if center_in_region:
                    plans, shift_debug = build_multi_run_shift_plans_move_out_of_regions(
                        sample_key=key,
                        gt_xyxy=gt_pix,
                        region_boxes_px=region_boxes_px,
                        W=W,
                        H=H,
                        budget_px=budget_px,
                        away_margin_px=float(args.away_margin_px),
                        edge_margin_px=float(args.canvas_edge_margin_px),
                        num_runs=num_runs,
                        random_trials=int(args.random_trials),
                    )
                    lr_mode = "center_in_region_move_out"
                else:
                    plans = [
                        {
                            "run": i + 1,
                            "style": "no_shift_center_outside_region",
                            "dx": 0,
                            "dy": 0,
                            "filled_px": 0,
                            "filled_ratio": 0.0,
                            "move_out_failed": False,
                            "reason": "bbox_center_not_in_any_raw_region",
                            "min_dist_to_regions": float(_min_dist_rect_to_boxes(gt_pix, region_boxes_px)),
                            "min_dist_to_canvas_edge": float(_min_dist_rect_to_canvas_edges(gt_pix, W=W, H=H)),
                        }
                        for i in range(num_runs)
                    ]
                    shift_debug = {
                        "rule": "bbox center is outside all raw regions, so no shift is applied.",
                        "num_raw_regions": len(region_boxes_rel),
                        "budget_px": int(budget_px),
                    }
                    lr_mode = "center_outside_region_no_shift"

                multi_run_trials: Dict[str, Dict[str, Any]] = {}
                multi_run_correctnesses: List[str] = []
                any_transformed = False

                for run_idx, plan in enumerate(plans, 1):
                    dx = safe_int(plan.get("dx", 0), 0)
                    dy = safe_int(plan.get("dy", 0), 0)
                    any_transformed = any_transformed or (dx != 0 or dy != 0)

                    shifted_img = shift_image_on_canvas(img_rgb, dx, dy)
                    shifted_gt_pix = bbox_after_shift(gt_pix, dx, dy)

                    model_img, pad = pad_to_aspect_limit(shifted_img, max_ratio=float(args.aspect_ratio_limit))
                    shifted_gt_for_model = offset_bbox_for_padding(shifted_gt_pix, pad)
                    model_region_boxes = [offset_bbox_for_padding(rb, pad) for rb in region_boxes_px]

                    try:
                        raw = run_vlm_once(
                            model=model,
                            processor=processor,
                            image=model_img,
                            instruction=instruction,
                            max_new_tokens=int(args.max_new_tokens),
                            pad_token_id=int(pad_token_id),
                        )
                        pred_x, pred_y, parsed_ok, parse_dbg = extract_coordinates(
                            raw,
                            model_img.size[0],
                            model_img.size[1],
                            coord_output_mode=args.coord_output_mode,
                        )
                    except Exception as e:
                        raw = ""
                        pred_x, pred_y, parsed_ok = 0, 0, False
                        parse_dbg = {"ok": False, "reason": "model_or_parse_exception", "error": str(e)}

                    if not parsed_ok:
                        correctness = "wrong_format"
                    else:
                        correctness = "correct" if point_in_box(pred_x, pred_y, shifted_gt_for_model) else "wrong"

                    multi_run_correctnesses.append(correctness)
                    stats_overall_runs[run_idx - 1].add(correctness)
                    if any_transformed:
                        stats_transformed_selected_runs[run_idx - 1].add(correctness)

                    trial = {
                        "run": int(run_idx),
                        "raw_response": raw,
                        "parsed_ok": bool(parsed_ok),
                        "pred_pixel": [int(pred_x), int(pred_y)] if parsed_ok else None,
                        "parse_debug": parse_dbg,
                        "correctness": correctness,
                        "shift": {
                            **plan,
                            "transform": {"kind": "shift", "dx": dx, "dy": dy},
                        },
                        "bbox_shifted_pixel": list(map(float, shifted_gt_for_model)),
                        "bbox_shifted_rel_original_canvas": list(map(float, bbox_pixel_to_rel_xyxy(shifted_gt_pix, W, H))),
                        "pad_for_model_input": list(map(int, pad)),
                    }
                    multi_run_trials[f"run_{run_idx}"] = trial

                    if args.vis_dir and args.vis_every > 0 and (src_idx % args.vis_every == 0):
                        safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(key))[:120]
                        vis_path = os.path.join(args.vis_dir, f"idx{src_idx}_{safe_key}_run{run_idx}.png")
                        visualize_prediction_pixel(
                            model_img,
                            pred_x,
                            pred_y,
                            vis_path,
                            bbox_xyxy=shifted_gt_for_model,
                            region_boxes_xyxy=model_region_boxes,
                            parsed=parsed_ok,
                        )
                        trial["vis_path"] = vis_path

                vote = summarize_vote_correctness(multi_run_correctnesses)
                correctness = vote["correctness"]
                stats_overall.add(correctness)
                if any_transformed:
                    stats_transformed_selected.add(correctness)

                lr_mode_counter[lr_mode] += 1

                out = {
                    "idx": src_idx,
                    "key": key,
                    "img_path": img_path,
                    "gt_type": sample_gt_type,
                    "prompt_to_evaluate": instruction,
                    "bbox_orig": sample.get("bbox", None),
                    "bbox_rel": list(map(float, gt_rel)),
                    "bbox_center_rel": [float(cx_rel), float(cy_rel)],
                    "center_in_raw_region": bool(center_in_region),
                    "region_rule": "center-only over raw rel01 region boxes from --region_bbox_path; sample bbox converted to pixel then rel center; region geometry unchanged",
                    "num_raw_regions": int(len(region_boxes_rel)),
                    "lr_mode": lr_mode,
                    "multi_run_correctnesses": multi_run_correctnesses,
                    "multi_run_vote": vote,
                    "multi_run_trials": multi_run_trials,
                    "correctness": correctness,
                    "_debug": {
                        "rank": rank,
                        "list_idx": list_idx,
                        "image_size": [int(W), int(H)],
                        "shift_debug": shift_debug,
                    },
                }

                write_one(fout, out)
                if any_transformed:
                    write_one_transformed(fout_transformed, out)
                if resume:
                    done_keys.add(key)

    finally:
        fout_transformed.close()

    # ----------------------------
    # Rank summary
    # ----------------------------
    acc_runs = [s.acc for s in stats_overall_runs]
    acc_mean, acc_var = mean_and_variance(acc_runs)
    acc_runs_t = [s.acc for s in stats_transformed_selected_runs]
    acc_mean_t, acc_var_t = mean_and_variance(acc_runs_t)

    rank_summary = {
        "rank": rank,
        "world_size": world_size,
        "num_written": n_written,
        "num_transformed_written": n_transformed_written,
        "metrics": {
            "overall": stats_to_metrics(stats_overall),
            "transformed_selected": stats_to_metrics(stats_transformed_selected),
            "multi_run_overall": {
                "num_runs": num_runs,
                "vote_threshold": majority_vote_threshold(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_overall_runs],
                "acc_runs": acc_runs,
                "acc_mean": acc_mean,
                "acc_variance": acc_var,
            },
            "multi_run_transformed_selected": {
                "num_runs": num_runs,
                "vote_threshold": majority_vote_threshold(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_transformed_selected_runs],
                "acc_runs": acc_runs_t,
                "acc_mean": acc_mean_t,
                "acc_variance": acc_var_t,
            },
            "lr_mode_counts": dict(lr_mode_counter),
        },
        "config": {
            "model_path": args.model_path,
            "coord_output_mode": args.coord_output_mode,
            "region_bbox_path": args.region_bbox_path,
            "region_coordinate_mode": "relative_0_1_only",
            "membership_rule": "sample_bbox_center_in_any_raw_rel01_region",
            "merge_region": False,
            "dilate_region": False,
            "away_margin_px": args.away_margin_px,
            "canvas_edge_margin_px": args.canvas_edge_margin_px,
            "max_change_pct": args.max_change_pct,
            "num_runs": num_runs,
        },
    }

    with open(summary_path_rank, "w", encoding="utf-8") as f:
        json.dump(rank_summary, f, indent=2, ensure_ascii=False)

    dist_barrier(enabled_dist)

    if rank == 0 and not args.no_merge:
        merged_summary = merge_rank_outputs_single(
            output_path_final=args.output_path,
            output_path_rank_pattern=f"{args.output_path}.rank{{rank}}.jsonl",
            world_size=world_size,
            summary_path_final=args.summary_path,
            summary_path_rank_pattern=f"{args.summary_path}.rank{{rank}}.json",
            transformed_output_path_final=transformed_output_path,
        )
        print(json.dumps(merged_summary["metrics"], indent=2, ensure_ascii=False))

    dist_barrier(enabled_dist)
    dist_cleanup(enabled_dist)


if __name__ == "__main__":
    main()
