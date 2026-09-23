from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import Counter

import torch
import torch.distributed as dist
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps, ImageChops

from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# Qwen3-VL
try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


# ----------------------------
# Prompt (pixel output)
# ----------------------------
SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

_COORD_RE = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")

DEFAULT_MODEL_PATH = "ByteDance-Seed/UI-TARS-1.5-7B"
DEFAULT_REGION_BBOX_PATH = "/pasteur/u/andy0207/chenyue/uitars_region_bounds_only.jsonl"


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


def rank0_print(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def dist_barrier(enabled: bool):
    if enabled and dist.is_initialized():
        dist.barrier()


# -------------------------
# Basic helpers
# -------------------------
def clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


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


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_json_records(json_file_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    json_file_paths = [f for f in os.listdir(json_file_dir) if f.endswith((".json", ".jsonl"))]
    json_file_paths.sort()

    for fn in json_file_paths:
        full_path = os.path.join(json_file_dir, fn)
        with open(full_path, "r", encoding="utf-8") as f:
            if fn.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        continue
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
    return records


def load_records_from_path(path: str) -> List[Dict[str, Any]]:
    if not path:
        return []
    if os.path.isdir(path):
        return load_json_records(path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    records: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
        return records

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return list(data)
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
        if (not os.path.isabs(str(img_path))) and base_image_dir:
            return os.path.join(base_image_dir, str(img_path))
        return str(img_path)
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
            done.add(k)
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
    """Normalize user-facing coordinate mode aliases."""
    mode = str(mode or "auto").strip().lower()
    aliases = {
        "auto": "auto",
        "pixel": "pixel",
        "pixels": "pixel",
        "abs": "pixel",
        "absolute": "pixel",
        "absolute_pixel": "pixel",
        "absolute_pixels": "pixel",
        "norm1000": "norm1000",
        "norm_1000": "norm1000",
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
    Return (x_pix, y_pix, parsed_ok, debug_info).

    coord_output_mode:
      - pixel:    model outputs absolute pixel coordinates in the current model-input image space.
                  Use this for HelloKKMe/GTA1-7B and other absolute-coordinate models.
      - norm1000: model outputs coordinates normalized to [0,1000].
      - rel01:    model outputs coordinates normalized to [0,1].
      - auto:     legacy heuristic, kept for backward compatibility.

    Important: auto is ambiguous for absolute-pixel models, because a real pixel
    output like (500,300) overlaps with the [0,1000] norm1000 range.  For GTA1,
    explicitly use --coord_output_mode pixel.
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
        # Legacy auto heuristic.  Kept to avoid changing old runs, but do not
        # use it for GTA1/absolute-pixel models.
        mode = "pixel"
        x_pix = xr
        y_pix = yr

        if (0.0 <= xr <= 1.0) and (0.0 <= yr <= 1.0):
            mode = "rel01"
            x_pix = xr * float(W)
            y_pix = yr * float(H)
        elif (0.0 <= xr <= 1000.0) and (0.0 <= yr <= 1000.0):
            if (W > 1000) or (H > 1000):
                mode = "norm1000"
                x_pix = xr / 1000.0 * float(W)
                y_pix = yr / 1000.0 * float(H)
            else:
                if (xr > W) or (yr > H):
                    mode = "norm1000"
                    x_pix = xr / 1000.0 * float(W)
                    y_pix = yr / 1000.0 * float(H)
                else:
                    mode = "pixel"
                    x_pix = xr
                    y_pix = yr

    xi_raw = int(round(x_pix))
    yi_raw = int(round(y_pix))
    in_bounds_before_clamp = (0 <= xi_raw < int(W)) and (0 <= yi_raw < int(H))

    xi = clamp_int(xi_raw, 0, max(0, int(W) - 1))
    yi = clamp_int(yi_raw, 0, max(0, int(H) - 1))

    dbg = {
        "ok": True,
        "mode_requested": mode_req,
        "mode_used": mode,
        "raw_pair": [float(xr), float(yr)],
        "W": int(W),
        "H": int(H),
        "pixel_float": [float(x_pix), float(y_pix)],
        "pixel_int_before_clamp": [int(xi_raw), int(yi_raw)],
        "pixel_int_clamped": [int(xi), int(yi)],
        "in_bounds_before_clamp": bool(in_bounds_before_clamp),
    }
    if mode == "norm1000":
        dbg["norm1000_pair"] = [float(xr), float(yr)]
    if mode == "rel01":
        dbg["rel01_pair"] = [float(xr), float(yr)]
    return xi, yi, True, dbg


def extract_coordinates_auto(raw: str, W: int, H: int) -> Tuple[int, int, bool, Dict[str, Any]]:
    """Backward-compatible wrapper for old call sites."""
    return extract_coordinates(raw, W, H, coord_output_mode="auto")


# -------------------------
# BBox helpers
# -------------------------
def coerce_bbox_to_xyxy_pixels(bbox: Any, W: int, H: int) -> Tuple[float, float, float, float]:
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox: {bbox}")

    a, b, c, d = map(float, bbox)

    # normalized (0~1)
    if max(a, b, c, d) <= 1.0:
        if c >= a and d >= b:
            x1, y1, x2, y2 = a * W, b * H, c * W, d * H
        else:
            x1, y1, x2, y2 = a * W, b * H, (a + c) * W, (b + d) * H
        return x1, y1, x2, y2

    # pixel
    if c >= a and d >= b:
        return a, b, c, d
    return a, b, a + c, b + d


def point_in_box(x: float, y: float, box_xyxy: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box_xyxy
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def bbox_center_xy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def bbox_after_shift(gt_xyxy: Tuple[float, float, float, float], dx: int, dy: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = gt_xyxy
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def point_in_any_box(x: float, y: float, boxes: List[Tuple[float, float, float, float]]) -> bool:
    for b in boxes:
        if point_in_box(x, y, b):
            return True
    return False


def bbox_intersects_box(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def bbox_intersects_any_region(
    bbox_xyxy: Tuple[float, float, float, float],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    dilate_x: float,
    dilate_y: float,
) -> bool:
    for (rx1, ry1, rx2, ry2) in region_boxes_xyxy:
        rb = (
            float(rx1) - float(dilate_x),
            float(ry1) - float(dilate_y),
            float(rx2) + float(dilate_x),
            float(ry2) + float(dilate_y),
        )
        if bbox_intersects_box(bbox_xyxy, rb):
            return True
    return False


def bbox_fully_within_canvas(
    bbox_xyxy: Tuple[float, float, float, float],
    canvas_w: int,
    canvas_h: int,
    edge_margin: int = 0,
) -> bool:
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    m = max(0.0, float(edge_margin))
    return (
        x2 > x1
        and y2 > y1
        and x1 >= m
        and y1 >= m
        and x2 <= float(canvas_w) - m
        and y2 <= float(canvas_h) - m
    )


def bbox_rect_distance(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """
    Return the minimum Euclidean distance between two axis-aligned bounding boxes; return 0 if they overlap or touch.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(math.hypot(dx, dy))


def min_distance_bbox_parts_to_regions(
    bbox_parts: List[Tuple[float, float, float, float]],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    dilate_x: float,
    dilate_y: float,
) -> float:
    """
    Compute the minimum distance from the bounding-box parts to all dilated regions.
    A larger value means the parts are farther from the regions.
    """
    if not bbox_parts:
        return 0.0
    if not region_boxes_xyxy:
        return float("inf")

    best = float("inf")
    for part in bbox_parts:
        for (rx1, ry1, rx2, ry2) in region_boxes_xyxy:
            rb = (
                float(rx1) - float(dilate_x),
                float(ry1) - float(dilate_y),
                float(rx2) + float(dilate_x),
                float(ry2) + float(dilate_y),
            )
            best = min(best, bbox_rect_distance(part, rb))
            if best <= 0.0:
                return 0.0
    return float(best)


# -------------------------
# Aspect-ratio padding (model-input legalize, NOT shift)
# -------------------------
def pad_to_aspect_limit(
    img: Image.Image,
    *,
    max_ratio: float = 199.0,  # strict < 200
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Qwen smart_resize requires aspect ratio < 200.
    If too "skinny", pad the short side (black border) to satisfy the constraint.
    Return padded image and (pad_left, pad_top, pad_right, pad_bottom).
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
        out = ImageOps.expand(img, border=(0, pad_top, 0, pad_bottom), fill=fill)
        return out, (0, pad_top, 0, pad_bottom)
    else:
        target_w = max(w, target_min)
        pad_total = target_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        out = ImageOps.expand(img, border=(pad_left, 0, pad_right, 0), fill=fill)
        return out, (pad_left, 0, pad_right, 0)


# -------------------------
# Region loading
# 1) {"bbox":[x1,y1,x2,y2]} abs pixels or relative [0,1]
# 2) {"bounds":{"x0":..,"x1":..,"y0":..,"y1":..}} relative
# 3) {"bounds":{"x0":..,"x1":..,"y0":..,"y1":..}} abs pixels
# -------------------------
RegionSpec = Dict[str, Any]


def _is_probably_rel_bounds(bounds: Dict[str, Any]) -> bool:
    try:
        x0 = float(bounds["x0"])
        x1 = float(bounds["x1"])
        y0 = float(bounds["y0"])
        y1 = float(bounds["y1"])
    except Exception:
        return False
    return (max(x0, x1, y0, y1) <= 1.5) and (min(x0, x1, y0, y1) >= -0.5)


def load_region_specs_jsonl_flexible(path: str) -> List[RegionSpec]:
    specs: List[RegionSpec] = []
    for obj in iter_jsonl(path):
        b = obj.get("bbox", None)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                x1, y1, x2, y2 = map(float, b)
                x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
                y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
                bbox_bounds = {"x0": x_lo, "x1": x_hi, "y0": y_lo, "y1": y_hi}
                if _is_probably_rel_bounds(bbox_bounds):
                    specs.append({"kind": "rel_bounds", "x0": x_lo, "x1": x_hi, "y0": y_lo, "y1": y_hi})
                else:
                    specs.append({"kind": "abs_bbox_xyxy", "xyxy": (x_lo, y_lo, x_hi, y_hi)})
                continue
            except Exception:
                pass

        bounds = obj.get("bounds", None)
        if isinstance(bounds, dict):
            try:
                x0 = float(bounds["x0"])
                x1 = float(bounds["x1"])
                y0 = float(bounds["y0"])
                y1 = float(bounds["y1"])

                x_lo, x_hi = (x0, x1) if x0 <= x1 else (x1, x0)
                y_lo, y_hi = (y0, y1) if y0 <= y1 else (y1, y0)

                if _is_probably_rel_bounds(bounds):
                    specs.append({"kind": "rel_bounds", "x0": x_lo, "x1": x_hi, "y0": y_lo, "y1": y_hi})
                else:
                    specs.append({"kind": "abs_bbox_xyxy", "xyxy": (x_lo, y_lo, x_hi, y_hi)})
                continue
            except Exception:
                pass

    return specs


def region_specs_to_abs_xyxy(specs: List[RegionSpec], W: int, H: int) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for s in specs:
        if s.get("kind") == "abs_bbox_xyxy":
            x1, y1, x2, y2 = s["xyxy"]
            out.append((float(x1), float(y1), float(x2), float(y2)))
        elif s.get("kind") == "rel_bounds":
            x0 = float(s["x0"])
            x1 = float(s["x1"])
            y0 = float(s["y0"])
            y1 = float(s["y1"])
            x_lo, x_hi = (x0, x1) if x0 <= x1 else (x1, x0)
            y_lo, y_hi = (y0, y1) if y0 <= y1 else (y1, y0)
            out.append((x_lo * W, y_lo * H, x_hi * W, y_hi * H))
    return out


# -----------------------------
# BBoxUnion (NO MERGE)
# -----------------------------
class BBoxUnion:
    def __init__(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        eps: float = 1e-9,
        shift_dx: float = 0.0,
        shift_dy: float = 0.0,
        drop_collapsed: bool = True,
    ):
        self.eps = float(eps)
        self.shift_dx = float(shift_dx)
        self.shift_dy = float(shift_dy)
        self.drop_collapsed = bool(drop_collapsed)

        dx = self.shift_dx
        dy = self.shift_dy

        processed: List[Tuple[float, float, float, float]] = []
        for (x1, y1, x2, y2) in bboxes:
            rx1 = float(x1) + dx
            ry1 = float(y1) + dy
            rx2 = float(x2) + dx
            ry2 = float(y2) + dy
            if rx2 <= rx1 or ry2 <= ry1:
                if not self.drop_collapsed:
                    raise ValueError(f"Invalid region bbox: orig={x1,y1,x2,y2} -> {rx1,ry1,rx2,ry2}")
                continue
            processed.append((rx1, ry1, rx2, ry2))

        if not processed:
            raise RuntimeError("All region bboxes are invalid.")

        self.boxes: List[Tuple[float, float, float, float]] = processed
        self.region_bbox_count_after_merge = int(len(self.boxes))


# -------------------------
# Visualization
# -------------------------
def visualize_prediction_pixel(
    img: Image.Image,
    x: int,
    y: int,
    save_path: str,
    *,
    bbox_xyxy: Optional[Any] = None,
    region_boxes_xyxy: Optional[List[Tuple[float, float, float, float]]] = None,
    parsed: bool = True,
) -> None:
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    if region_boxes_xyxy:
        for rb in region_boxes_xyxy:
            try:
                rx1, ry1, rx2, ry2 = map(float, rb)
            except Exception:
                continue
            rx1 = int(max(0, min(W - 1, round(rx1))))
            ry1 = int(max(0, min(H - 1, round(ry1))))
            rx2 = int(max(0, min(W - 1, round(rx2))))
            ry2 = int(max(0, min(H - 1, round(ry2))))
            if rx2 > rx1 and ry2 > ry1:
                draw.rectangle((rx1, ry1, rx2, ry2), outline="deepskyblue", width=3)
                draw.rectangle((rx1, ry1, rx2, ry2), fill=(0, 191, 255, 30))

    if bbox_xyxy is not None:
        bboxes_to_draw: List[Tuple[float, float, float, float]] = []
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) == 4 and not isinstance(bbox_xyxy[0], (list, tuple)):
            try:
                bboxes_to_draw = [tuple(map(float, bbox_xyxy))]
            except Exception:
                bboxes_to_draw = []
        elif isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) > 0 and isinstance(bbox_xyxy[0], (list, tuple)) and len(bbox_xyxy[0]) == 4:
            for bb in bbox_xyxy:
                try:
                    bboxes_to_draw.append(tuple(map(float, bb)))
                except Exception:
                    pass

        for (x1, y1, x2, y2) in bboxes_to_draw:
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
        r = 30
        draw.ellipse((x - r, y - r, x + r, y + r), outline="lime", fill=(0, 255, 0, 90), width=5)
        cross_len = 15
        draw.line((x - cross_len, y, x + cross_len, y), fill="lime", width=5)
        draw.line((x, y - cross_len, x, y + cross_len), fill="lime", width=5)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img.save(save_path)


# -------------------------
# Shift-only transform (fixed canvas, wrap-around translation, no cropping)
# -------------------------
def _canonical_shift_delta(d: int, size: int) -> int:
    """
    Wrap-around shift: only d mod size matters.
    Canonicalize to a representative with small |d|.
    """
    size = int(size)
    if size <= 1:
        return 0
    d = int(d) % size
    hi = (size - 1) // 2
    if d > hi:
        d -= size
    return int(d)


def _wrap_interval_1d(lo: float, hi: float, size: int) -> List[Tuple[float, float]]:
    """
    Map an interval [lo,hi] onto a wrap-around axis mod size.
    Returns 1~2 segments in [0,size].
    Assumes hi>lo and (hi-lo) <= size.
    """
    size = int(size)
    if size <= 1:
        return [(0.0, float(size))]

    length = float(hi - lo)
    if length <= 0:
        return []

    start = float(lo % size)
    end = start + length
    if end <= size:
        return [(start, end)]
    return [(0.0, end - size), (start, float(size))]


def bbox_after_shift_wrap(
    gt_xyxy: Tuple[float, float, float, float],
    dx: int,
    dy: int,
    W: int,
    H: int,
) -> List[Tuple[float, float, float, float]]:
    """
    Wrap-around shifted bbox, possibly splits into up to 4 rectangles.
    """
    x1, y1, x2, y2 = map(float, gt_xyxy)
    W = int(W)
    H = int(H)
    dx = _canonical_shift_delta(dx, W)
    dy = _canonical_shift_delta(dy, H)

    xs = _wrap_interval_1d(x1 + dx, x2 + dx, W)
    ys = _wrap_interval_1d(y1 + dy, y2 + dy, H)

    parts: List[Tuple[float, float, float, float]] = []
    for (xa1, xa2) in xs:
        for (ya1, ya2) in ys:
            if xa2 > xa1 and ya2 > ya1:
                parts.append((xa1, ya1, xa2, ya2))
    return parts


def bbox_is_complete_after_wrap(
    gt_xyxy: Tuple[float, float, float, float],
    dx: int,
    dy: int,
    W: int,
    H: int,
) -> bool:
    parts = bbox_after_shift_wrap(gt_xyxy, dx=dx, dy=dy, W=W, H=H)
    return len(parts) == 1


def bbox_parts_intersect_any_region(
    bbox_parts: List[Tuple[float, float, float, float]],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    dilate_x: float,
    dilate_y: float,
) -> bool:
    if not bbox_parts:
        return False
    for part in bbox_parts:
        for (rx1, ry1, rx2, ry2) in region_boxes_xyxy:
            rb = (
                float(rx1) - float(dilate_x),
                float(ry1) - float(dilate_y),
                float(rx2) + float(dilate_x),
                float(ry2) + float(dilate_y),
            )
            if bbox_intersects_box(part, rb):
                return True
    return False


def shift_image_on_canvas(img: Image.Image, dx: int, dy: int, fill=(0, 0, 0)) -> Image.Image:
    """
    Keep the canvas at the original image size (W, H) and shift all image content with wrap-around:
      - dx > 0: shift content to the right
      - dy > 0: shift content downward
    This produces no black borders; overflow wraps in from the opposite side like a tiled image.

    The fill parameter is retained only for backward compatibility and is unused in wrap mode.
    """
    W, H = img.size
    dx = _canonical_shift_delta(dx, W)
    dy = _canonical_shift_delta(dy, H)
    return ImageChops.offset(img, dx, dy)


def filled_pixels_for_shift(W: int, H: int, dx: int, dy: int) -> int:
    """
    A wrap-around shift produces no black borders.
    Retain an approximate change budget:
      seam = |dx|*H + |dy|*W - |dx|*|dy|
    It must be computed from canonical dx/dy values.
    """
    W = int(W)
    H = int(H)
    dx = _canonical_shift_delta(dx, W)
    dy = _canonical_shift_delta(dy, H)
    adx = abs(int(dx))
    ady = abs(int(dy))
    return int(adx * H + ady * W - adx * ady)


def compute_change_budget_pixels(W: int, H: int, max_change_pct: float) -> int:
    return int(math.floor((float(W) * float(H) * float(max_change_pct) / 100.0) + 1e-9))


def _clamp_to_range(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _nearest_to_zero_in_range(lo: int, hi: int, *, forbid_zero: bool) -> Optional[int]:
    if hi < lo:
        return None
    if (not forbid_zero) and (lo <= 0 <= hi):
        return 0

    cand = []
    if lo <= -1 <= hi:
        cand.append(-1)
    if lo <= 1 <= hi:
        cand.append(1)
    if cand:
        return cand[0]

    if hi < 0:
        return hi
    if lo > 0:
        return lo
    return None


def _candidate_values_for_range(lo: int, hi: int) -> List[int]:
    if hi < lo:
        return []

    vals = set()
    vals.add(lo)
    vals.add(hi)

    for t in [0, -1, 1, -2, 2, -4, 4, -8, 8]:
        vals.add(_clamp_to_range(t, lo, hi))

    for frac in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
        vals.add(_clamp_to_range(int(round(lo + (hi - lo) * frac)), lo, hi))

    nz = _nearest_to_zero_in_range(lo, hi, forbid_zero=True)
    if nz is not None:
        vals.add(int(nz))

    if hi >= 1:
        vals.add(max(1, lo))
    if lo <= -1:
        vals.add(min(-1, hi))

    return sorted(vals)


def _build_wrap_axis_candidates(size: int, prefer_positive: bool = False) -> List[int]:
    """
    Construct a richer set of candidates for wrapped axes to allow larger shifts.
    """
    size = int(size)
    lo = -size // 2
    hi = (size - 1) // 2
    vals = set(_candidate_values_for_range(lo, hi))

    mags = [
        1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256,
        320, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
        2560, 3072, 3584, 4096
    ]
    max_abs = max(abs(lo), abs(hi))
    for m in mags:
        if m > max_abs:
            continue
        if lo <= -m <= hi:
            vals.add(-m)
        if lo <= m <= hi:
            vals.add(m)

    vals.add(lo)
    vals.add(hi)
    vals.add(_clamp_to_range(int(round(lo * 0.75)), lo, hi))
    vals.add(_clamp_to_range(int(round(hi * 0.75)), lo, hi))
    vals.add(_clamp_to_range(int(round(lo * 0.5)), lo, hi))
    vals.add(_clamp_to_range(int(round(hi * 0.5)), lo, hi))

    if prefer_positive:
        vals.add(hi)
        vals.add(_clamp_to_range(hi - 1, lo, hi))
        vals.add(_clamp_to_range(int(round(hi * 0.9)), lo, hi))

    return sorted(vals)


def _style_score_penalty(style: str, dx: int, dy: int, W: int, H: int) -> float:
    s = 0.0
    if style == "horizontal":
        if dy != 0:
            s += 10.0 * abs(dy) * W
    elif style == "vertical":
        if dx != 0:
            s += 10.0 * abs(dx) * H
    elif style == "diagonal":
        if dx == 0 or dy == 0:
            s += 1e12
    elif style == "prefer_right":
        if dx <= 0:
            s += 1e9 + 1000.0 * abs(dx) * H
        s += 1.0 * abs(dy) * W
    elif style == "prefer_down":
        if dy <= 0:
            s += 1e9 + 1000.0 * abs(dy) * W
        s += 1.0 * abs(dx) * H
    return float(s)


def bbox_fully_within_box(
    inner: Tuple[float, float, float, float],
    outer: Tuple[float, float, float, float],
    *,
    eps: float = 1e-6,
) -> bool:
    ix1, iy1, ix2, iy2 = map(float, inner)
    ox1, oy1, ox2, oy2 = map(float, outer)
    return (
        ix2 > ix1 and iy2 > iy1
        and ix1 >= ox1 - eps and iy1 >= oy1 - eps
        and ix2 <= ox2 + eps and iy2 <= oy2 + eps
    )


def bbox_fully_within_any_region(
    bbox_xyxy: Tuple[float, float, float, float],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
) -> bool:
    return any(bbox_fully_within_box(bbox_xyxy, rb) for rb in region_boxes_xyxy)


def bbox_parts_fully_within_any_region(
    bbox_parts: List[Tuple[float, float, float, float]],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
) -> bool:
    return len(bbox_parts) == 1 and bbox_fully_within_any_region(bbox_parts[0], region_boxes_xyxy)


def _clip_region_to_canvas(
    rb: Tuple[float, float, float, float],
    W: int,
    H: int,
) -> Optional[Tuple[float, float, float, float]]:
    rx1, ry1, rx2, ry2 = map(float, rb)
    rx1 = max(0.0, min(float(W), rx1))
    ry1 = max(0.0, min(float(H), ry1))
    rx2 = max(0.0, min(float(W), rx2))
    ry2 = max(0.0, min(float(H), ry2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return (rx1, ry1, rx2, ry2)


def _candidate_regions_for_inside_move(
    gt_xyxy: Tuple[float, float, float, float],
    region_boxes_proc: List[Tuple[float, float, float, float]],
    W: int,
    H: int,
) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    cx, cy = bbox_center_xy(gt_xyxy)
    containing: List[Tuple[int, Tuple[float, float, float, float]]] = []
    all_regions: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for rid, rb0 in enumerate(region_boxes_proc):
        rb = _clip_region_to_canvas(rb0, W=W, H=H)
        if rb is None:
            continue
        item = (int(rid), rb)
        all_regions.append(item)
        if point_in_box(cx, cy, rb):
            containing.append(item)
    return containing if containing else all_regions


def pick_random_shift_inside_region(
    *,
    gt_xyxy: Tuple[float, float, float, float],
    W: int,
    H: int,
    region_boxes_proc: List[Tuple[float, float, float, float]],
    rng: random.Random,
    used_pairs: Optional[Set[Tuple[int, int]]] = None,
    forbid_zero: bool = True,
    max_tries: int = 500,
) -> Optional[Dict[str, Any]]:
    W, H = int(W), int(H)
    x1, y1, x2, y2 = map(float, gt_xyxy)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    candidate_regions = _candidate_regions_for_inside_move(gt_xyxy, region_boxes_proc, W, H)
    if not candidate_regions:
        return None
    weights = []
    for _, (rx1, ry1, rx2, ry2) in candidate_regions:
        weights.append(max(1e-6, (rx2 - rx1 + 1.0) * (ry2 - ry1 + 1.0)))
    best_zero: Optional[Dict[str, Any]] = None
    for _ in range(max(1, int(max_tries))):
        rid, rb = rng.choices(candidate_regions, weights=weights, k=1)[0]
        rx1, ry1, rx2, ry2 = rb
        new_cx = rng.uniform(rx1, rx2)
        new_cy = rng.uniform(ry1, ry2)
        old_cx = x1 + bw / 2.0
        old_cy = y1 + bh / 2.0
        dx = _canonical_shift_delta(int(round(new_cx - old_cx)), W)
        dy = _canonical_shift_delta(int(round(new_cy - old_cy)), H)
        if used_pairs is not None and (dx, dy) in used_pairs:
            continue
        gt_parts = bbox_after_shift_wrap(gt_xyxy, dx=dx, dy=dy, W=W, H=H)
        if len(gt_parts) != 1:
            continue
        shifted_bbox = gt_parts[0]
        if not bbox_fully_within_canvas(shifted_bbox, canvas_w=W, canvas_h=H, edge_margin=0):
            continue
        shifted_cx, shifted_cy = bbox_center_xy(shifted_bbox)
        if not point_in_box(shifted_cx, shifted_cy, rb):
            continue
        filled = filled_pixels_for_shift(W, H, dx, dy)
        bbox_inside_target_region = bbox_fully_within_box(shifted_bbox, rb)
        out = {
            "dx": int(dx), "dy": int(dy), "filled_px": int(filled),
            "region_id": int(rid),
            "region_xyxy_proc": [float(rx1), float(ry1), float(rx2), float(ry2)],
            "bbox_after_shift_parts": [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in gt_parts],
            "bbox_center_in_target_region": True,
            "bbox_center_after_shift": [float(shifted_cx), float(shifted_cy)],
            "bbox_inside_target_region": bool(bbox_inside_target_region),
            "bbox_inside_any_region": bbox_fully_within_any_region(shifted_bbox, region_boxes_proc),
            "shift_mode": "inside_region_random",
        }
        if dx == 0 and dy == 0:
            best_zero = out
            if forbid_zero:
                continue
        return out
    if (not forbid_zero) and best_zero is not None:
        return best_zero
    return None
def pick_any_nonzero_wrap_shift_style(
    *,
    W: int,
    H: int,
    budget_px: int,
    style: str,
    used_pairs: Optional[Set[Tuple[int, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    For negative samples without a ground-truth bounding box: require a nonzero wrapped shift within budget and maximize movement.
    """
    W = int(W)
    H = int(H)

    dx_vals = set(_build_wrap_axis_candidates(W, prefer_positive=(style == "prefer_right")))
    dy_vals = set(_build_wrap_axis_candidates(H, prefer_positive=(style == "prefer_down")))

    best = None
    target_filled = max(1, int(round(0.35 * float(budget_px))))

    for dx0 in sorted(dx_vals):
        for dy0 in sorted(dy_vals):
            dx = _canonical_shift_delta(int(dx0), W)
            dy = _canonical_shift_delta(int(dy0), H)

            if dx == 0 and dy == 0:
                continue
            if used_pairs is not None and (dx, dy) in used_pairs:
                continue

            filled = filled_pixels_for_shift(W, H, dx, dy)
            if filled > budget_px:
                continue

            s = abs(float(filled) - float(target_filled))
            s += _style_score_penalty(style, dx, dy, W, H)

            key = (s, -filled, -(abs(dx) + abs(dy)), -abs(dx), -abs(dy))
            if best is None or key < best["key"]:
                best = {
                    "dx": int(dx),
                    "dy": int(dy),
                    "filled_px": int(filled),
                    "score": float(s),
                    "key": key,
                }

    if best is None:
        return None
    best.pop("key", None)
    return best


def build_negative_wrap_shift_plans(
    *,
    W: int,
    H: int,
    budget_px: int,
    num_runs: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    styles_base = ["horizontal", "vertical", "diagonal", "prefer_right", "prefer_down"]
    styles: List[str] = [styles_base[i % len(styles_base)] for i in range(num_runs)]

    debug: Dict[str, Any] = {
        "budget_px": int(budget_px),
        "shift_mode": "wrap",
        "rule": "negative sample: must move with non-zero wrap-around shift under seam-area budget",
    }

    plans: List[Dict[str, Any]] = []
    used_pairs: Set[Tuple[int, int]] = set()

    for run_i, style in enumerate(styles, 1):
        cand = pick_any_nonzero_wrap_shift_style(
            W=W,
            H=H,
            budget_px=budget_px,
            style=str(style),
            used_pairs=used_pairs,
        )
        if cand is None:
            plans.append({
                "run": int(run_i),
                "skipped": True,
                "reason": "no_feasible_nonzero_wrap_shift_for_negative",
                "style": str(style),
            })
            continue

        used_pairs.add((int(cand["dx"]), int(cand["dy"])))
        plans.append({
            "run": int(run_i),
            "skipped": False,
            "style": str(style),
            "region_id": None,
            "region_xyxy_proc": None,
            "dx": int(cand["dx"]),
            "dy": int(cand["dy"]),
            "dx_full": int(cand["dx"]),
            "dy_full": int(cand["dy"]),
            "dx_target_1_3": int(cand["dx"]),
            "dy_target_1_3": int(cand["dy"]),
            "applied_fraction": "full",
            "applied_fraction_human": "full",
            "fraction_mode": "direct_wrap_shift_negative",
            "filled_px": int(cand["filled_px"]),
            "filled_ratio": (0.0 if (W * H) == 0 else float(cand["filled_px"]) / float(W * H)),
            "bbox_after_shift_parts": None,
        })

    debug["used_shift_count"] = int(len(used_pairs))
    return plans, debug


def build_multi_run_shift_plans(
    *,
    gt_xyxy: Tuple[float, float, float, float],
    region_boxes_proc: List[Tuple[float, float, float, float]],
    W: int,
    H: int,
    num_runs: int,
    rng: random.Random,
    max_tries: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "num_regions_proc": int(len(region_boxes_proc)),
        "shift_mode": "inside_region_random",
        "rule": (
            "randomly sample a new target position inside the region; "
            "the shifted bbox must remain complete and its center must be inside the selected region"
        ),
        "max_tries_per_run": int(max_tries),
        "placement_constraints_used": True,
    }
    plans: List[Dict[str, Any]] = []
    used_pairs: Set[Tuple[int, int]] = set()
    for run_i in range(1, int(num_runs) + 1):
        cand = pick_random_shift_inside_region(
            gt_xyxy=gt_xyxy, W=W, H=H, region_boxes_proc=region_boxes_proc,
            rng=rng, used_pairs=used_pairs, forbid_zero=True, max_tries=max_tries,
        )
        if cand is None:
            plans.append({
                "run": int(run_i), "skipped": True,
                "reason": "no_feasible_nonzero_random_shift_inside_region",
                "style": "inside_region_random",
            })
            continue
        used_pairs.add((int(cand["dx"]), int(cand["dy"])))
        plans.append({
            "run": int(run_i), "skipped": False, "style": "inside_region_random",
            "region_id": cand.get("region_id", None),
            "region_xyxy_proc": cand.get("region_xyxy_proc", None),
            "dx": int(cand["dx"]), "dy": int(cand["dy"]),
            "dx_full": int(cand["dx"]), "dy_full": int(cand["dy"]),
            "dx_target_1_3": int(cand["dx"]), "dy_target_1_3": int(cand["dy"]),
            "applied_fraction": "full", "applied_fraction_human": "full",
            "fraction_mode": "random_inside_region",
            "filled_px": int(cand["filled_px"]),
            "filled_ratio": (0.0 if (W * H) == 0 else float(cand["filled_px"]) / float(W * H)),
            "bbox_after_shift_parts": cand.get("bbox_after_shift_parts", None),
            "bbox_center_in_target_region": bool(cand.get("bbox_center_in_target_region", False)),
            "bbox_center_after_shift": cand.get("bbox_center_after_shift", None),
            "bbox_inside_target_region": bool(cand.get("bbox_inside_target_region", False)),
            "bbox_inside_any_region": bool(cand.get("bbox_inside_any_region", False)),
            "shift_mode": "inside_region_random",
        })
    debug["used_shift_count"] = int(len(used_pairs))
    return plans, debug


# -------------------------
# Stats / merge
# -------------------------
_VALID_CORRECTNESS = {"correct", "wrong", "wrong_format", "skipped"}


def normalize_correctness(v: Any) -> str:
    s = str(v) if v is not None else "wrong_format"
    return s if s in _VALID_CORRECTNESS else "wrong_format"


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


def perfect_correct_metrics(count: int) -> Dict[str, Any]:
    n = int(max(0, count))
    return {
        "num_total": n,
        "num_scored": n,
        "num_skipped": 0,
        "num_correct": n,
        "num_wrong": 0,
        "num_wrong_format": 0,
        "acc": (1.0 if n > 0 else 0.0),
    }


def mean_and_variance(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    mean = float(sum(vals) / len(vals))
    var = float(sum((x - mean) * (x - mean) for x in vals) / len(vals))
    return mean, var


def vote_correctness(correctnesses: List[Any]) -> str:
    if not correctnesses:
        return "skipped"

    norm = [normalize_correctness(c) for c in correctnesses]
    num_correct = sum(1 for c in norm if c == "correct")
    if num_correct * 2 > len(norm):
        return "correct"

    non_correct = [c for c in norm if c != "correct"]
    if not non_correct:
        return "wrong"

    cnt = Counter(non_correct)
    max_votes = max(cnt.values())
    winners = {k for k, v in cnt.items() if v == max_votes}
    for c in non_correct:
        if c in winners:
            return c
    return "wrong"


def is_transformed_record(obj: Dict[str, Any]) -> bool:
    trials = obj.get("multi_run_trials", None)
    if isinstance(trials, dict):
        for _, run_v in trials.items():
            if not isinstance(run_v, dict):
                continue
            t = run_v.get("shift", None)
            if isinstance(t, dict):
                tr = t.get("transform", None)
                if isinstance(tr, dict) and str(tr.get("kind", "")) == "shift":
                    dx = safe_int(tr.get("dx", 0), 0)
                    dy = safe_int(tr.get("dy", 0), 0)
                    if dx != 0 or dy != 0:
                        return True
    lr_trials = obj.get("lr_trials", None)
    if isinstance(lr_trials, dict):
        for v in lr_trials.values():
            if isinstance(v, dict):
                tr = v.get("transform", None)
                if isinstance(tr, dict) and str(tr.get("kind", "")) == "shift":
                    dx = safe_int(tr.get("dx", 0), 0)
                    dy = safe_int(tr.get("dy", 0), 0)
                    if dx != 0 or dy != 0:
                        return True
    return False


def extract_transformed_ops(obj: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    trials = obj.get("multi_run_trials", None)
    if isinstance(trials, dict):
        for _, run_v in trials.items():
            if not isinstance(run_v, dict):
                continue
            t = run_v.get("shift", None)
            if isinstance(t, dict):
                style = str(t.get("style", "shift"))
                tr = t.get("transform", None)
                if isinstance(tr, dict) and str(tr.get("kind", "")) == "shift":
                    dx = safe_int(tr.get("dx", 0), 0)
                    dy = safe_int(tr.get("dy", 0), 0)
                    if dx != 0 or dy != 0:
                        out.append(f"shift_{style}")
    if out:
        return out
    return ["shift"] if is_transformed_record(obj) else []


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
    transformed_op_counter = Counter()
    seen: Set[str] = set()
    lr_mode_counter = Counter()

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
                seen.add(k)

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

                    ops = extract_transformed_ops(obj)
                    if not ops:
                        transformed_op_counter["unknown"] += 1
                    else:
                        for op in ops:
                            transformed_op_counter[str(op)] += 1
                    if transformed_fout is not None:
                        transformed_fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

                mode = obj.get("lr_mode", None)
                if mode:
                    lr_mode_counter[str(mode)] += 1

                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if transformed_fout is not None:
        transformed_fout.close()

    merged_summary = {
        "metrics": {
            "overall": stats_to_metrics(merged_stats),
            "transformed_selected": stats_to_metrics(transformed_stats),
            "transformed_selected_count": int(transformed_count),
            "transformed_selected_baseline_before_transform": perfect_correct_metrics(transformed_count),
            "transformed_selected_ops": dict(transformed_op_counter),
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

    os.makedirs(os.path.dirname(summary_path_final) or ".", exist_ok=True)
    with open(summary_path_final, "w", encoding="utf-8") as f:
        json.dump(merged_summary, f, indent=2, ensure_ascii=False)

    return merged_summary


# -------------------------
# Model selection
# -------------------------
def pick_model_cls(model_type: str):
    if model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("Qwen3VLForConditionalGeneration not available. Please upgrade transformers.")
        return Qwen3VLForConditionalGeneration
    raise RuntimeError(f"Unsupported model_type={model_type!r} (expected qwen2_5_vl or qwen3_vl)")


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


# -------------------------
# Main
# -------------------------
def main():
    enabled_dist, rank, world_size, local_rank, device = dist_init()

    ap = argparse.ArgumentParser(
        description=(
            "Standalone rewrite of move_random_new.py for relative-region jsonl "
            "(for example worst_cells_32b.jsonl) and Qwen3-VL-32B outputs that "
            "often use norm1000 coordinates."
        )
    )

    # ---- model ----
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--attn_impl", type=str, default="", choices=["", "flash_attention_2", "sdpa", "eager"])

    # ---- data ----
    ap.add_argument("--json_file_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/annotations")
    ap.add_argument("--base_image_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/images")
    ap.add_argument("--subset_path", type=str, default="/pasteur/u/andy0207/chenyue/uitars_new_subset.jsonl")

    # ---- region ----
    ap.add_argument(
        "--region_bbox_path",
        type=str,
        default=DEFAULT_REGION_BBOX_PATH,
        help=(
            "Region bbox jsonl. Supports {'bbox':[x1,y1,x2,y2]} abs pixels OR "
            "{'bounds':{x0,x1,y0,y1}} with either relative or absolute coordinates. "
            f"Default: {DEFAULT_REGION_BBOX_PATH}"
        ),
    )

    # ---- budget / runs ----
    ap.add_argument(
        "--max_change_pct",
        type=float,
        default=99.0,
        help="Max seam-area percent for negative-sample fallback shifts; positive in-region moves do not use this budget.",
    )
    ap.add_argument("--num_crop_runs", type=int, default=5, help="Number of random in-region moves.")
    ap.add_argument("--random_seed", type=int, default=42, help="Base seed for reproducible random in-region movement.")
    ap.add_argument("--random_search_tries", type=int, default=500, help="Max rejection-sampling attempts per run.")

    # ---- run control ----
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--summary_path", type=str, default="")
    ap.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    ap.add_argument("--inst_style", type=str, default="instruction")
    ap.add_argument("--gt_type", type=str, default="all", choices=["positive", "negative", "all"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of shifted runs to generate together per sample.",
    )
    ap.add_argument(
        "--coord_output_mode",
        type=str,
        default="pixel",
        help=(
            "Coordinate format of model output. "
            "Use 'pixel' for absolute-pixel models such as HelloKKMe/GTA1-7B; "
            "use 'norm1000' for Qwen-style normalized outputs; "
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
        default=199.0,
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
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

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

    rank0_print(rank, "[Mode] ✅ RANDOM IN-REGION shift: bbox is moved randomly inside region and never outside region.")
    rank0_print(rank, f"[Preset] model_path={args.model_path}")
    rank0_print(rank, f"[Preset] region_bbox_path={args.region_bbox_path}")
    rank0_print(rank, "[Constraint] moved bbox center must be inside the selected region")
    rank0_print(rank, f"[MultiRun] num_runs={args.num_crop_runs}; random_seed={args.random_seed}")
    rank0_print(rank, "[Shift] non-zero WRAP shift; bbox stays complete and its center remains inside the selected region")
    rank0_print(rank, f"[AspectFix] aspect_ratio_limit={args.aspect_ratio_limit} (<200), will pad if too skinny")
    rank0_print(rank, f"[Parse] coord_output_mode={args.coord_output_mode} (auto / pixel / norm1000 / rel01)")
    rank0_print(rank, "[Parse] region jsonl auto mode: bbox(abs) / bounds(rel01) / bounds(abs)")
    rank0_print(rank, f"[Infer] batch_size={args.batch_size} shifted runs per generate call")

    # ----------------------------
    # Load model/processor
    # ----------------------------
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt is None:
        raise RuntimeError("Missing model_type in config. Please check --model_path.")

    ModelCls = pick_model_cls(str(mt))

    model_kwargs = dict(
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    if device.type == "cuda":
        model_kwargs["device_map"] = {"": str(device)}
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    rank0_print(rank, f"[Rank {rank}] Loading model_type={mt} on {device} from {args.model_path} ...")
    model = ModelCls.from_pretrained(args.model_path, **model_kwargs)

    processor = processor_from_pretrained_flexible(args.model_path, args.min_pixels, args.max_pixels)
    model.eval()
    pad_token_id = _get_pad_token_id(processor)

    # ----------------------------
    # Load records & region
    # ----------------------------
    if args.subset_path:
        records = load_records_from_path(args.subset_path)
        rank0_print(rank, f"[Data] subset_path mode ON: loaded {len(records)} records from {args.subset_path}")
    else:
        records = load_json_records(args.json_file_dir)
        rank0_print(rank, f"[Data] annotation dir mode: loaded {len(records)} records from {args.json_file_dir}")

    if args.gt_type != "all":
        records = [r for r in records if r.get("gt_type", "positive") == args.gt_type]

    if args.stride and args.stride > 1:
        records = records[:: args.stride]

    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    rank0_print(rank, f"[Data] keep all records (no correctness filter): {len(records)}")

    region_specs = load_region_specs_jsonl_flexible(args.region_bbox_path)
    if not region_specs:
        raise RuntimeError(f"No valid region specs found in: {args.region_bbox_path}")

    my_indices = [i for i in range(len(records)) if (i % world_size) == rank]
    rank0_print(rank, f"[Data] Loaded {len(records)} records; shard={len(my_indices)}/{len(records)} world_size={world_size}")

    # ----------------------------
    # Resume
    # ----------------------------
    num_runs = int(args.num_crop_runs)
    stats_overall_runs: List[Stats] = [Stats() for _ in range(num_runs)]
    stats_transformed_selected_runs: List[Stats] = [Stats() for _ in range(num_runs)]
    transformed_op_counter_runs: List[Counter] = [Counter() for _ in range(num_runs)]
    stats_overall = stats_overall_runs[0]
    stats_transformed_selected = stats_transformed_selected_runs[0]
    transformed_op_counter = transformed_op_counter_runs[0]
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

    union_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def get_union_for_original_image(W: int, H: int) -> Tuple[BBoxUnion, List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
        key = (int(W), int(H))
        if key in union_cache:
            c = union_cache[key]
            return c["u0"], c["region_bboxes_abs_orig"], c["region_boxes_proc"]

        region_bboxes_abs_orig = region_specs_to_abs_xyxy(region_specs, W=W, H=H)
        if not region_bboxes_abs_orig:
            raise RuntimeError(f"No valid region boxes after converting specs at W,H=({W},{H}).")

        u0 = BBoxUnion(
            region_bboxes_abs_orig,
            shift_dx=0.0,
            shift_dy=0.0,
            drop_collapsed=True,
        )
        region_boxes_proc = list(u0.boxes)

        union_cache[key] = {
            "u0": u0,
            "region_bboxes_abs_orig": region_bboxes_abs_orig,
            "region_boxes_proc": region_boxes_proc,
        }
        return u0, region_bboxes_abs_orig, region_boxes_proc

    # ----------------------------
    # Main loop
    # ----------------------------
    pbar = tqdm(my_indices, desc=f"Eval(rank{rank})", disable=(rank != 0))
    fout_transformed = open(transformed_output_path_rank, out_mode, encoding="utf-8")
    with open(output_path_rank, out_mode, encoding="utf-8") as fout:
        for list_idx in pbar:
            sample = records[list_idx]
            key = stable_key_from_screenspot(sample)
            if resume and key in done_keys:
                continue

            src_idx = safe_int(sample.get("idx", list_idx), list_idx)
            sample_gt_type = sample.get("gt_type", "positive")

            img_path = resolve_image_path(sample, args.base_image_dir)
            if (not img_path) or (not os.path.exists(img_path)):
                out = {
                    "idx": src_idx,
                    "key": key,
                    "img_path": img_path if img_path else None,
                    "gt_type": sample_gt_type,
                    "prompt_to_evaluate": pick_instruction(sample, args.language),
                    "bbox_orig": sample.get("bbox", None),
                    "lr_mode": "skipped",
                    "lr_crop_search": {},
                    "lr_trials": {},
                    "correctness": "wrong_format",
                    "_debug": {"rank": rank, "reason": "missing_image", "list_idx": list_idx},
                }
                for _s in stats_overall_runs:
                    _s.add("wrong_format")
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
                    "prompt_to_evaluate": pick_instruction(sample, args.language),
                    "bbox_orig": sample.get("bbox", None),
                    "lr_mode": "skipped",
                    "lr_crop_search": {},
                    "lr_trials": {},
                    "correctness": "wrong_format",
                    "_debug": {"rank": rank, "reason": "image_open_failed", "error": str(e), "list_idx": list_idx},
                }
                for _s in stats_overall_runs:
                    _s.add("wrong_format")
                write_one(fout, out)
                if resume:
                    done_keys.add(key)
                continue

            W, H = img_rgb.size
            instruction = pick_instruction(sample, args.language)
            bbox_raw = sample.get("bbox", None)

            gt_xyxy0: Optional[Tuple[float, float, float, float]] = None
            if sample_gt_type != "negative" and (bbox_raw is not None):
                try:
                    gt_xyxy0 = coerce_bbox_to_xyxy_pixels(bbox_raw, W=W, H=H)
                except Exception:
                    gt_xyxy0 = None

            if sample_gt_type != "negative" and gt_xyxy0 is None:
                out = {
                    "idx": src_idx,
                    "key": key,
                    "img_path": img_path,
                    "platform": sample.get("platform", "unknown"),
                    "application": sample.get("application", "unknown"),
                    "group": sample.get("group", sample.get("ui_type", "unknown")),
                    "lang": sample.get("lang", args.language),
                    "instruction_style": sample.get("instruction_style", args.inst_style),
                    "gt_type": sample_gt_type,
                    "ui_type": sample.get("ui_type", "unknown"),
                    "prompt_to_evaluate": instruction,
                    "bbox_orig": bbox_raw,
                    "lr_mode": "skipped",
                    "lr_crop_search": {},
                    "lr_trials": {},
                    "correctness": "wrong_format",
                    "_debug": {"rank": rank, "reason": "gt_bbox_parse_failed", "list_idx": list_idx},
                }
                for _s in stats_overall_runs:
                    _s.add("wrong_format")
                write_one(fout, out)
                if resume:
                    done_keys.add(key)
                continue

            _u0, region_bboxes_abs_orig, region_boxes_proc = get_union_for_original_image(W, H)
            lr_crop_search: Dict[str, Any] = {
                "_mode": "random_inside_region",
            }
            selected_run_plans: List[Dict[str, Any]] = []

            if sample_gt_type == "negative":
                budget_px = compute_change_budget_pixels(W, H, float(args.max_change_pct))
                lr_crop_search["_budget"] = {
                    "budget_px": int(budget_px),
                    "max_change_pct": float(args.max_change_pct),
                }
                plans, dbg = build_negative_wrap_shift_plans(
                    W=W,
                    H=H,
                    budget_px=budget_px,
                    num_runs=num_runs,
                )
                selected_run_plans = plans
                lr_crop_search["_search_debug"] = dbg
            else:
                assert gt_xyxy0 is not None
                plans, dbg = build_multi_run_shift_plans(
                    gt_xyxy=gt_xyxy0,
                    region_boxes_proc=region_boxes_proc,
                    W=W,
                    H=H,
                    num_runs=num_runs,
                    rng=random.Random(int(args.random_seed) + int(src_idx) * 1000003 + int(rank) * 9176),
                    max_tries=int(args.random_search_tries),
                )
                selected_run_plans = plans
                lr_crop_search["_search_debug"] = dbg

            any_non_skipped = any((not p.get("skipped", True)) for p in selected_run_plans)
            any_nonzero_shift = any(
                (not p.get("skipped", True)) and (int(p.get("dx", 0)) != 0 or int(p.get("dy", 0)) != 0) for p in selected_run_plans
            )

            if not any_non_skipped:
                lr_mode = "skipped_no_feasible_inside_region_shift"
            elif not any_nonzero_shift:
                lr_mode = "none_or_zero_shift_only"
            else:
                lr_mode = "inside_region_shift_selected"
            lr_mode_counter[lr_mode] += 1

            def _eval_one_shift(
                plan: Dict[str, Any],
                run_tag: str,
                infer_result: Optional[Dict[str, Any]] = None,
                defer_infer: bool = False,
            ) -> Dict[str, Any]:
                trial: Dict[str, Any] = {
                    "style": str(plan.get("style", "shift")),
                    "run_tag": str(run_tag),
                }

                if plan.get("skipped", False):
                    trial.update({
                        "transform": {"kind": "shift", "mode": "wrap", "dx": 0, "dy": 0},
                        "shift_dx": 0,
                        "shift_dy": 0,
                        "target_region_id": plan.get("region_id", None),
                        "target_region_xyxy_proc": plan.get("region_xyxy_proc", None),
                        "filled_px": None,
                        "raw_response": "",
                        "pred_pixel": None,
                        "pred_pixel_padded": None,
                        "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                        "img_size_padded": [int(W), int(H)],
                        "bbox_transformed": None,
                        "bbox_transformed_parts": None,
                        "correctness": "skipped",
                        "_debug": {"reason": plan.get("reason", "skipped")},
                    })
                    return trial

                dx = _canonical_shift_delta(int(plan.get("dx", 0)), W)
                dy = _canonical_shift_delta(int(plan.get("dy", 0)), H)
                filled = int(plan.get("filled_px", filled_pixels_for_shift(W, H, dx, dy)))

                trial["transform"] = {"kind": "shift", "mode": str(plan.get("shift_mode", "inside_region_random")), "dx": int(dx), "dy": int(dy)}
                trial["shift_dx"] = int(dx)
                trial["shift_dy"] = int(dy)
                trial["filled_px"] = int(filled)
                trial["budget_px"] = int(budget_px)
                trial["target_region_id"] = plan.get("region_id", None)
                trial["target_region_xyxy_proc"] = plan.get("region_xyxy_proc", None)
                trial["dx_full"] = int(plan.get("dx_full", 0))
                trial["dy_full"] = int(plan.get("dy_full", 0))
                trial["dx_target_1_3"] = int(plan.get("dx_target_1_3", dx))
                trial["dy_target_1_3"] = int(plan.get("dy_target_1_3", dy))
                trial["applied_fraction"] = str(plan.get("applied_fraction", ""))
                trial["applied_fraction_human"] = str(plan.get("applied_fraction_human", ""))
                trial["fraction_mode"] = str(plan.get("fraction_mode", ""))
                trial["bbox_after_shift_parts_search"] = plan.get("bbox_after_shift_parts", None)
                trial["min_region_distance"] = float(plan.get("min_region_distance", 0.0))
                trial["bbox_center_in_target_region"] = bool(plan.get("bbox_center_in_target_region", False))
                trial["bbox_center_after_shift_search"] = plan.get("bbox_center_after_shift", None)
                trial["bbox_inside_target_region"] = bool(plan.get("bbox_inside_target_region", False))
                trial["bbox_inside_any_region"] = bool(plan.get("bbox_inside_any_region", False))

                img_shift = shift_image_on_canvas(img_rgb, dx=dx, dy=dy, fill=(0, 0, 0))
                trial["img_size_transformed"] = [int(W), int(H)]

                gt_parts: Optional[List[Tuple[float, float, float, float]]] = None
                if (sample_gt_type != "negative" and gt_xyxy0 is not None):
                    gt_parts = bbox_after_shift_wrap(gt_xyxy0, dx=dx, dy=dy, W=W, H=H)

                shifted_bbox_within_canvas_runtime = None
                if gt_parts is None or len(gt_parts) == 0:
                    trial["bbox_transformed"] = None
                    trial["bbox_transformed_parts"] = None
                else:
                    if len(gt_parts) == 1:
                        g0 = gt_parts[0]
                        trial["bbox_transformed"] = [float(g0[0]), float(g0[1]), float(g0[2]), float(g0[3])]
                        trial["bbox_transformed_parts"] = None
                        shifted_bbox_within_canvas_runtime = bbox_fully_within_canvas(
                            g0,
                            canvas_w=int(W),
                            canvas_h=int(H),
                            edge_margin=0,
                        )
                    else:
                        trial["bbox_transformed"] = None
                        trial["bbox_transformed_parts"] = [[float(a), float(b), float(c), float(d)] for (a, b, c, d) in gt_parts]
                        shifted_bbox_within_canvas_runtime = False

                img_for_model, pad4 = pad_to_aspect_limit(img_shift, max_ratio=float(args.aspect_ratio_limit), fill=(0, 0, 0))
                pad_left, pad_top, pad_right, pad_bottom = pad4
                Wp, Hp = img_for_model.size
                trial["pad"] = {"left": int(pad_left), "top": int(pad_top), "right": int(pad_right), "bottom": int(pad_bottom)}
                trial["img_size_padded"] = [int(Wp), int(Hp)]

                region_boxes_for_model = []
                for (rx1, ry1, rx2, ry2) in region_boxes_proc:
                    region_boxes_for_model.append((rx1 + pad_left, ry1 + pad_top, rx2 + pad_left, ry2 + pad_top))

                user_prompt = SYSTEM_PROMPT.format(height=Hp, width=Wp).strip()

                raw = ""
                prompt_len = 0
                if defer_infer:
                    trial["_deferred_infer"] = {
                        "user_prompt": user_prompt,
                        "instruction": instruction,
                        "image": img_for_model,
                    }
                    return trial

                if infer_result is not None:
                    err = str(infer_result.get("error", ""))
                    if err:
                        trial.update({
                            "raw_response": "",
                            "pred_pixel": None,
                            "pred_pixel_padded": None,
                            "correctness": "wrong_format",
                            "_debug": {"reason": "infer_failed", "error": err},
                        })
                        return trial
                    raw = str(infer_result.get("raw", ""))
                    prompt_len = int(infer_result.get("prompt_len", 0))
                else:
                    try:
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt + "\n\n"},
                                {"type": "image", "image": img_for_model},
                                {"type": "text", "text": "\n" + instruction},
                            ],
                        }]
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        inputs.pop("token_type_ids", None)
                    except Exception:
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt + "\n\n"},
                                {"type": "image"},
                                {"type": "text", "text": "\n" + instruction},
                            ],
                        }]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(
                            text=[text],
                            images=[img_for_model],
                            return_tensors="pt",
                            padding=False,
                        )
                        inputs.pop("token_type_ids", None)

                    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                    try:
                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=0.0,
                                use_cache=True,
                                pad_token_id=_get_pad_token_id(processor),
                            )

                        if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
                            prompt_len = int(inputs["attention_mask"][0].sum().item())
                        else:
                            prompt_len = int(inputs["input_ids"].shape[-1])

                        gen_trim = generated_ids[0, prompt_len:].detach().cpu()
                        out_texts = processor.batch_decode([gen_trim], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        raw = out_texts[0] if out_texts else ""
                    except Exception as e:
                        trial.update({
                            "raw_response": "",
                            "pred_pixel": None,
                            "pred_pixel_padded": None,
                            "correctness": "wrong_format",
                            "_debug": {"reason": "infer_failed", "error": str(e)},
                        })
                        return trial

                trial["raw_response"] = raw

                xpad, ypad, parsed, coord_dbg = extract_coordinates(
                    raw,
                    W=Wp,
                    H=Hp,
                    coord_output_mode=args.coord_output_mode,
                )
                trial["pred_pixel_padded"] = [int(xpad), int(ypad)] if parsed else None

                if parsed:
                    x0 = int(xpad - pad_left)
                    y0 = int(ypad - pad_top)
                    in_bounds_unpad = (0 <= x0 < W) and (0 <= y0 < H)
                else:
                    x0, y0 = 0, 0
                    in_bounds_unpad = False

                trial["pred_pixel"] = [int(x0), int(y0)] if (parsed and in_bounds_unpad) else None

                if sample_gt_type == "negative":
                    correctness = "correct" if (not parsed) else "wrong"
                else:
                    if (not parsed) or (gt_parts is None) or (len(gt_parts) == 0):
                        correctness = "wrong_format"
                    else:
                        correctness = "wrong" if (not in_bounds_unpad) else ("correct" if point_in_any_box(float(x0), float(y0), gt_parts) else "wrong")
                correctness = normalize_correctness(correctness)
                trial["correctness"] = correctness

                if gt_xyxy0 is not None:
                    cx0, cy0 = bbox_center_xy(gt_xyxy0)
                    cx2 = (float(cx0) + float(dx)) % float(W)
                    cy2 = (float(cy0) + float(dy)) % float(H)

                    gt_parts_dbg = bbox_after_shift_wrap(gt_xyxy0, dx=dx, dy=dy, W=W, H=H)
                    bbox_intersects_region_after = bbox_parts_intersect_any_region(
                        gt_parts_dbg,
                        region_boxes_proc,
                        dilate_x=0.0,
                        dilate_y=0.0,
                    )
                    min_region_dist_after = min_distance_bbox_parts_to_regions(
                        gt_parts_dbg,
                        region_boxes_proc,
                        dilate_x=0.0,
                        dilate_y=0.0,
                    )
                    bbox_inside_any_region_runtime = bbox_parts_fully_within_any_region(
                        gt_parts_dbg,
                        region_boxes_proc,
                    )
                    target_region_xyxy = plan.get("region_xyxy_proc", None)
                    bbox_center_in_any_region_runtime = point_in_any_box(float(cx2), float(cy2), region_boxes_proc)
                    if target_region_xyxy is not None and len(gt_parts_dbg) == 1:
                        target_region_tuple = tuple(map(float, target_region_xyxy))
                        bbox_inside_target_region_runtime = bbox_fully_within_box(
                            gt_parts_dbg[0], target_region_tuple
                        )
                        bbox_center_in_target_region_runtime = point_in_box(float(cx2), float(cy2), target_region_tuple)
                    else:
                        bbox_inside_target_region_runtime = None
                        bbox_center_in_target_region_runtime = None

                    center_xy_before = [float(cx0), float(cy0)]
                    center_xy_after = [float(cx2), float(cy2)]
                else:
                    bbox_intersects_region_after = None
                    min_region_dist_after = None
                    bbox_inside_any_region_runtime = None
                    bbox_inside_target_region_runtime = None
                    bbox_center_in_any_region_runtime = None
                    bbox_center_in_target_region_runtime = None
                    center_xy_before = None
                    center_xy_after = None

                trial["_debug"] = {
                    "parsed": bool(parsed),
                    "in_bounds_unpad": bool(in_bounds_unpad),
                    "prompt_len": int(prompt_len),
                    "coord_parse": coord_dbg,
                    "bbox_intersects_region_after_shift": bbox_intersects_region_after,
                    "min_region_distance_after_shift": min_region_dist_after,
                    "bbox_center_in_any_region_runtime": bbox_center_in_any_region_runtime,
                    "bbox_center_in_target_region_runtime": bbox_center_in_target_region_runtime,
                    "bbox_inside_any_region_runtime": bbox_inside_any_region_runtime,
                    "bbox_inside_target_region_runtime": bbox_inside_target_region_runtime,
                    "center_xy_before": center_xy_before,
                    "center_xy_after_wrap": center_xy_after,
                    "canvas_wh": [int(W), int(H)],
                    "bbox_within_canvas_runtime": shifted_bbox_within_canvas_runtime,
                    "region_vis_count_model": int(len(region_boxes_for_model)),
                    "region_proc_count": int(len(region_boxes_proc)),
                    "shift_mode": "inside_region_random",
                }

                if args.vis_dir and args.vis_every and (src_idx % args.vis_every == 0):
                    vis_path = os.path.join(args.vis_dir, f"idx_{src_idx:06d}.shift.{run_tag}.rank{rank}.png")

                    bbox_for_vis_model = None
                    if gt_parts is not None and len(gt_parts) > 0:
                        bbox_for_vis_model = []
                        for (a, b, c, d) in gt_parts:
                            bbox_for_vis_model.append(
                                (float(a) + float(pad_left), float(b) + float(pad_top), float(c) + float(pad_left), float(d) + float(pad_top))
                            )

                    visualize_prediction_pixel(
                        img_for_model.copy(),
                        int(xpad) if parsed else 0,
                        int(ypad) if parsed else 0,
                        vis_path,
                        bbox_xyxy=bbox_for_vis_model,
                        region_boxes_xyxy=region_boxes_for_model,
                        parsed=bool(parsed),
                    )

                    vis_path_region = os.path.join(args.vis_dir, f"idx_{src_idx:06d}.shift.{run_tag}.rank{rank}.region_orig.png")
                    visualize_prediction_pixel(
                        img_rgb.copy(),
                        0,
                        0,
                        vis_path_region,
                        bbox_xyxy=gt_xyxy0 if gt_xyxy0 is not None else None,
                        region_boxes_xyxy=region_bboxes_abs_orig,
                        parsed=False,
                    )

                return trial

            def _infer_deferred_shift_trials(prepared_trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                results: List[Dict[str, Any]] = []
                batch_size = int(args.batch_size)

                def _infer_one(payload: Dict[str, Any]) -> Dict[str, Any]:
                    try:
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": str(payload["user_prompt"]) + "\n\n"},
                                {"type": "image", "image": payload["image"]},
                                {"type": "text", "text": "\n" + str(payload["instruction"])},
                            ],
                        }]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(
                            text=[text],
                            images=[payload["image"]],
                            return_tensors="pt",
                            padding=False,
                        )
                        inputs.pop("token_type_ids", None)
                        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=0.0,
                                use_cache=True,
                                pad_token_id=_get_pad_token_id(processor),
                            )

                        if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
                            prompt_len = int(inputs["attention_mask"][0].sum().item())
                        else:
                            prompt_len = int(inputs["input_ids"].shape[-1])

                        gen_trim = generated_ids[0, prompt_len:].detach().cpu()
                        out_texts = processor.batch_decode([gen_trim], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        return {"raw": out_texts[0] if out_texts else "", "prompt_len": prompt_len, "error": ""}
                    except Exception as e:
                        return {"raw": "", "prompt_len": 0, "error": str(e)}

                for start in range(0, len(prepared_trials), batch_size):
                    chunk = prepared_trials[start:start + batch_size]
                    payloads = [t["_deferred_infer"] for t in chunk]
                    try:
                        texts: List[str] = []
                        images: List[Any] = []
                        for payload in payloads:
                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": str(payload["user_prompt"]) + "\n\n"},
                                    {"type": "image", "image": payload["image"]},
                                    {"type": "text", "text": "\n" + str(payload["instruction"])},
                                ],
                            }]
                            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                            images.append(payload["image"])

                        inputs = processor(
                            text=texts,
                            images=images,
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs.pop("token_type_ids", None)
                        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=0.0,
                                use_cache=True,
                                pad_token_id=_get_pad_token_id(processor),
                            )

                        trim_start = int(inputs["input_ids"].shape[-1])
                        if "attention_mask" in inputs and torch.is_tensor(inputs["attention_mask"]):
                            prompt_lens = [int(inputs["attention_mask"][i].sum().item()) for i in range(len(chunk))]
                        else:
                            prompt_lens = [trim_start for _ in range(len(chunk))]

                        gen_trims = [generated_ids[i, trim_start:].detach().cpu() for i in range(len(chunk))]
                        out_texts = processor.batch_decode(gen_trims, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        for i, raw_text in enumerate(out_texts):
                            results.append({"raw": raw_text, "prompt_len": prompt_lens[i], "error": ""})
                    except Exception:
                        for payload in payloads:
                            results.append(_infer_one(payload))

                return results

            lr_trials: Dict[str, Any] = {}
            multi_run_trials: Dict[str, Dict[str, Any]] = {}
            multi_run_choices: List[Dict[str, Any]] = []
            multi_run_correctnesses: List[str] = []
            run_items: List[Tuple[int, Dict[str, Any], str]] = []

            for ri, plan in enumerate(selected_run_plans):
                run_tag = (
                    f"run{ri + 1}"
                    f".dx{int(plan.get('dx', 0))}.dy{int(plan.get('dy', 0))}"
                    f".full{int(plan.get('dx_full', 0))},{int(plan.get('dy_full', 0))}"
                    f".style{plan.get('style', 'shift')}"
                )
                run_items.append((ri, plan, run_tag))

            trials_by_ri: Dict[int, Dict[str, Any]] = {}
            if args.batch_size <= 1:
                for ri, plan, run_tag in run_items:
                    trials_by_ri[ri] = _eval_one_shift(plan, run_tag=run_tag)
            else:
                deferred_items: List[Tuple[int, Dict[str, Any], str, Dict[str, Any]]] = []
                for ri, plan, run_tag in run_items:
                    prepared_trial = _eval_one_shift(plan, run_tag=run_tag, defer_infer=True)
                    if "_deferred_infer" in prepared_trial:
                        deferred_items.append((ri, plan, run_tag, prepared_trial))
                    else:
                        trials_by_ri[ri] = prepared_trial

                infer_results = _infer_deferred_shift_trials([item[3] for item in deferred_items])
                for (ri, plan, run_tag, _prepared_trial), infer_result in zip(deferred_items, infer_results):
                    trials_by_ri[ri] = _eval_one_shift(plan, run_tag=run_tag, infer_result=infer_result)

            for ri, plan, run_tag in run_items:
                trial = trials_by_ri[ri]

                run_key = f"run_{ri + 1}"
                multi_run_trials[run_key] = {"shift": trial}

                multi_run_choices.append({
                    "run": int(ri + 1),
                    "style": str(plan.get("style", "shift")),
                    "dx": int(plan.get("dx", 0)),
                    "dy": int(plan.get("dy", 0)),
                    "dx_full": int(plan.get("dx_full", 0)),
                    "dy_full": int(plan.get("dy_full", 0)),
                    "dx_target_1_3": int(plan.get("dx_target_1_3", 0)),
                    "dy_target_1_3": int(plan.get("dy_target_1_3", 0)),
                    "applied_fraction": str(plan.get("applied_fraction", "")),
                    "applied_fraction_human": str(plan.get("applied_fraction_human", "")),
                    "fraction_mode": str(plan.get("fraction_mode", "")),
                    "region_id": plan.get("region_id", None),
                    "skipped": bool(plan.get("skipped", False)),
                    "reason": str(plan.get("reason", "")),
                    "bbox_after_shift_parts": plan.get("bbox_after_shift_parts", None),
                    "bbox_center_in_target_region": bool(plan.get("bbox_center_in_target_region", False)),
                    "bbox_center_after_shift": plan.get("bbox_center_after_shift", None),
                    "bbox_inside_target_region": bool(plan.get("bbox_inside_target_region", False)),
                    "bbox_inside_any_region": bool(plan.get("bbox_inside_any_region", False)),
                })

                c_i = normalize_correctness(trial.get("correctness", "wrong_format"))
                multi_run_correctnesses.append(c_i)
                stats_overall_runs[ri].add(c_i)

            trial0 = multi_run_trials["run_1"]["shift"]
            lr_trials["shift"] = trial0
            overall_correctness = vote_correctness(multi_run_correctnesses)
            stats_overall.add(overall_correctness)

            selected_for_transform = bool(
                any((not p.get("skipped", True)) and (int(p.get("dx", 0)) != 0 or int(p.get("dy", 0)) != 0) for p in selected_run_plans)
            )

            out = {
                "idx": src_idx,
                "key": key,
                "img_path": img_path,
                "platform": sample.get("platform", "unknown"),
                "application": sample.get("application", "unknown"),
                "group": sample.get("group", sample.get("ui_type", "unknown")),
                "lang": sample.get("lang", args.language),
                "instruction_style": sample.get("instruction_style", args.inst_style),
                "gt_type": sample_gt_type,
                "ui_type": sample.get("ui_type", "unknown"),
                "prompt_to_evaluate": instruction,
                "bbox_orig": bbox_raw,
                "lr_mode": lr_mode,
                "runnable_dirs": ["shift"],
                "lr_crop_search": lr_crop_search,
                "lr_trials": lr_trials,
                "selected_for_transform": selected_for_transform,
                "correctness": overall_correctness,
                "multi_run_num": num_runs,
                "multi_run_choices": multi_run_choices,
                "multi_run_correctnesses": multi_run_correctnesses,
                "multi_run_trials": multi_run_trials,
                "_debug": {
                    "rank": rank,
                    "world_size": world_size,
                    "device": str(device),
                    "pad_token_id": pad_token_id,
                    "attn_impl": args.attn_impl,
                    "min_pixels": args.min_pixels,
                    "max_pixels": args.max_pixels,
                    "aspect_ratio_limit": float(args.aspect_ratio_limit),
                    "max_change_pct": float(args.max_change_pct),
                    "num_runs": int(num_runs),
                    "batch_size": int(args.batch_size),
                    "list_idx": list_idx,
                    "subset_path": args.subset_path if args.subset_path else None,
                    "region_bbox_path": args.region_bbox_path,
                    "canvas_w": int(W),
                    "canvas_h": int(H),
                    "movement_rule": (
                        "random in-region shift; the original bbox is already inside a region; "
                        "we preserve bbox size and randomly sample a new location such that the shifted bbox "
                        "remains complete and its center is inside the selected region"
                    ),
                    "model_type": str(mt),
                },
            }

            write_one(fout, out)
            if selected_for_transform:
                stats_transformed_selected.add(overall_correctness)
                for ri, c_i in enumerate(multi_run_correctnesses):
                    stats_transformed_selected_runs[ri].add(c_i)
                    style_i = str(selected_run_plans[ri].get("style", "shift"))
                    transformed_op_counter_runs[ri][f"shift_{style_i}"] += 1
                write_one_transformed(fout_transformed, out)

            if resume:
                done_keys.add(key)

    fout_transformed.close()

    # ----------------------------
    # Per-rank summary
    # ----------------------------
    overall_acc_runs = [s.acc for s in stats_overall_runs]
    overall_acc_mean, overall_acc_var = mean_and_variance(overall_acc_runs)
    transformed_acc_runs = [s.acc for s in stats_transformed_selected_runs]
    transformed_acc_mean, transformed_acc_var = mean_and_variance(transformed_acc_runs)

    summary_rank = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "metrics": {
            "overall": stats_to_metrics(stats_overall),
            "transformed_selected": stats_to_metrics(stats_transformed_selected),
            "transformed_selected_count": int(n_transformed_written),
            "transformed_selected_baseline_before_transform": perfect_correct_metrics(n_transformed_written),
            "transformed_selected_ops": dict(transformed_op_counter),
            "lr_mode_counts": dict(lr_mode_counter),
            "multi_run_overall": {
                "num_runs": int(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_overall_runs],
                "acc_runs": overall_acc_runs,
                "acc_mean": overall_acc_mean,
                "acc_variance": overall_acc_var,
            },
            "multi_run_transformed_selected": {
                "num_runs": int(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_transformed_selected_runs],
                "acc_runs": transformed_acc_runs,
                "acc_mean": transformed_acc_mean,
                "acc_variance": transformed_acc_var,
            },
        },
        "config": {
            "model_path": args.model_path,
            "subset_path": args.subset_path,
            "json_file_dir": args.json_file_dir,
            "base_image_dir": args.base_image_dir,
            "language": args.language,
            "gt_type": args.gt_type,
            "max_samples": args.max_samples,
            "stride": args.stride,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": int(args.batch_size),
            "coord_output_mode": args.coord_output_mode,
            "num_runs": int(num_runs),
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "aspect_ratio_limit": float(args.aspect_ratio_limit),
            "max_change_pct": float(args.max_change_pct),
            "region_bbox_path": args.region_bbox_path,
            "vis_dir": args.vis_dir,
            "vis_every": args.vis_every,
            "flush_every": args.flush_every,
            "attn_impl": args.attn_impl,
            "output_jsonl_rank": output_path_rank,
            "output_transformed_jsonl_rank": transformed_output_path_rank,
            "output_transformed_jsonl_final": transformed_output_path,
            "mode": "shift_only_fixed_canvas_wrap",
        },
    }
    with open(summary_path_rank, "w", encoding="utf-8") as f:
        json.dump(summary_rank, f, indent=2, ensure_ascii=False)

    rank0_print(rank, f"[Rank {rank}] Done. acc={stats_overall.acc:.6f} (scored={stats_overall.scored}, skipped={stats_overall.skipped})")
    rank0_print(
        rank,
        f"[Rank {rank}] transformed_selected_acc={stats_transformed_selected.acc:.6f} "
        f"(count={n_transformed_written}, scored={stats_transformed_selected.scored}, skipped={stats_transformed_selected.skipped})",
    )
    rank0_print(rank, f"[Rank {rank}] multi_run_overall_accs={overall_acc_runs} mean={overall_acc_mean:.6f} var={overall_acc_var:.6f}")
    rank0_print(rank, f"[Rank {rank}] multi_run_transformed_accs={transformed_acc_runs} mean={transformed_acc_mean:.6f} var={transformed_acc_var:.6f}")
    rank0_print(rank, f"[Rank {rank}] Rank output: {output_path_rank}")
    rank0_print(rank, f"[Rank {rank}] Transformed-selected output: {transformed_output_path_rank}")
    rank0_print(rank, f"[Rank {rank}] Rank summary: {summary_path_rank}")

    # ----------------------------
    # Merge (rank0 only)
    # ----------------------------
    dist_barrier(enabled_dist)
    if (world_size > 1) and (not args.no_merge) and (rank == 0):
        output_rank_pattern = f"{args.output_path}.rank{{rank}}.jsonl"
        summary_rank_pattern = f"{args.summary_path}.rank{{rank}}.json"
        rank0_print(rank, f"[Merge] Merging rank outputs -> {args.output_path}")
        merged = merge_rank_outputs_single(
            output_path_final=args.output_path,
            output_path_rank_pattern=output_rank_pattern,
            world_size=world_size,
            summary_path_final=args.summary_path,
            summary_path_rank_pattern=summary_rank_pattern,
            transformed_output_path_final=transformed_output_path,
        )
        rank0_print(rank, f"[Merge] Done. merged_acc={merged['metrics']['overall']['acc']:.6f}")
        rank0_print(
            rank,
            f"[Merge] transformed_selected_acc={merged['metrics']['transformed_selected']['acc']:.6f} "
            f"(count={merged['metrics']['transformed_selected_count']})",
        )
        rank0_print(rank, f"[Merge] transformed_selected_ops={merged['metrics'].get('transformed_selected_ops', {})}")
        mr_overall = merged["metrics"].get("multi_run_overall", None)
        if isinstance(mr_overall, dict):
            rank0_print(
                rank,
                f"[Merge] multi_run_overall_accs={mr_overall.get('acc_runs', [])} "
                f"mean={float(mr_overall.get('acc_mean', 0.0)):.6f} "
                f"var={float(mr_overall.get('acc_variance', 0.0)):.6f}",
            )
        mr_trans = merged["metrics"].get("multi_run_transformed_selected", None)
        if isinstance(mr_trans, dict):
            rank0_print(
                rank,
                f"[Merge] multi_run_transformed_accs={mr_trans.get('acc_runs', [])} "
                f"mean={float(mr_trans.get('acc_mean', 0.0)):.6f} "
                f"var={float(mr_trans.get('acc_variance', 0.0)):.6f}",
            )
        baseline_transformed_acc_merged = merged["metrics"]["transformed_selected_baseline_before_transform"]["acc"]
        rank0_print(
            rank,
            f"[Compare-Merged-Transformed] baseline_acc={baseline_transformed_acc_merged:.6f} "
            f"-> transformed_selected_acc={merged['metrics']['transformed_selected']['acc']:.6f}  "
            f"delta={(merged['metrics']['transformed_selected']['acc']-baseline_transformed_acc_merged):+.6f}",
        )
        rank0_print(rank, f"[Merge] Final out: {args.output_path}")
        rank0_print(rank, f"[Merge] Final transformed-selected out: {transformed_output_path}")
        rank0_print(rank, f"[Merge] Final summary: {args.summary_path}")

    dist_barrier(enabled_dist)
    if enabled_dist and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
