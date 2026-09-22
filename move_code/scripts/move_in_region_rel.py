from __future__ import annotations

import argparse
import json
import os
import re
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

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
# Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
""".strip()

_COORD_RE = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")
_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)

DEFAULT_MODEL_PATH = "inclusionAI/UI-Venus-Ground-7B"
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


def str2bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "t", "yes", "y", "on")


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
    """
    Normalize user-facing coordinate-output mode names.

    Supported modes:
      - auto:     legacy heuristic, compatible with the original script behavior.
      - pixel:    model output is absolute pixel coordinate in the current model-input image.
                  Use this for models such as HelloKKMe/GTA1-7B.
      - norm1000: model output is normalized to [0, 1000].
      - rel01:    model output is normalized to [0, 1].
      - venus_bbox: UI-Venus bbox [x1,y1,x2,y2] in processed image coordinates.
    """
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
        "venus": "venus_bbox",
        "venus_bbox": "venus_bbox",
        "bbox": "venus_bbox",
        "x1y1x2y2": "venus_bbox",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unsupported --coord_output_mode={mode!r}. "
            "Use one of: auto, pixel, norm1000, rel01, venus_bbox."
        )
    return aliases[mode]


def extract_coordinates(
    raw: str,
    W: int,
    H: int,
    *,
    coord_output_mode: str = "auto",
    source_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, bool, Dict[str, Any]]:
    """
    Return (x_pix, y_pix, parsed_ok, debug_info).

    coord_output_mode:
      - pixel:    absolute pixel coordinates in the current model-input image.
                  This is required for HelloKKMe/GTA1-7B-like models.
      - norm1000: coordinates normalized to [0, 1000].
      - rel01:    coordinates normalized to [0, 1].
      - venus_bbox: bbox [x1,y1,x2,y2] in UI-Venus processed image coordinates;
                    returns the bbox center mapped to the current W/H image.
      - auto:     legacy heuristic from the original script.

    Important:
      The old auto heuristic is ambiguous for absolute-pixel models, because a
      valid pixel output like (500,300) also lies in [0,1000].  For GTA1-style
      models, explicitly pass --coord_output_mode pixel.
    """
    mode_req = normalize_coord_output_mode(coord_output_mode)

    if mode_req == "venus_bbox":
        m_box = _BBOX_RE.search(raw or "")
        if not m_box:
            return 0, 0, False, {
                "ok": False,
                "reason": "bbox_regex_not_found",
                "raw": raw,
                "coord_output_mode": mode_req,
            }
        try:
            x1, y1, x2, y2 = [float(m_box.group(i)) for i in range(1, 5)]
        except Exception as e:
            return 0, 0, False, {
                "ok": False,
                "reason": "bbox_float_parse_failed",
                "error": str(e),
                "raw": raw,
                "coord_output_mode": mode_req,
            }

        src_w, src_h = source_size if source_size is not None else (W, H)
        src_w = max(1, int(src_w))
        src_h = max(1, int(src_h))
        cx_src = (x1 + x2) * 0.5
        cy_src = (y1 + y2) * 0.5
        x_pix = cx_src / float(src_w) * float(W)
        y_pix = cy_src / float(src_h) * float(H)

        xi_raw = int(round(x_pix))
        yi_raw = int(round(y_pix))
        in_bounds_before_clamp = (0 <= xi_raw < int(W)) and (0 <= yi_raw < int(H))
        xi = clamp_int(xi_raw, 0, max(0, int(W) - 1))
        yi = clamp_int(yi_raw, 0, max(0, int(H) - 1))

        return xi, yi, True, {
            "ok": True,
            "mode_requested": mode_req,
            "mode_used": mode_req,
            "raw_box": [float(x1), float(y1), float(x2), float(y2)],
            "raw_pair": [float(cx_src), float(cy_src)],
            "source_size": [int(src_w), int(src_h)],
            "W": int(W),
            "H": int(H),
            "pixel_float": [float(x_pix), float(y_pix)],
            "pixel_int_before_clamp": [int(xi_raw), int(yi_raw)],
            "pixel_int_clamped": [int(xi), int(yi)],
            "in_bounds_before_clamp": bool(in_bounds_before_clamp),
        }

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

    mode_used = mode_req

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
        # Legacy auto heuristic.  Kept for backward compatibility.
        mode_used = "pixel"
        x_pix = xr
        y_pix = yr

        if (0.0 <= xr <= 1.0) and (0.0 <= yr <= 1.0):
            mode_used = "rel01"
            x_pix = xr * float(W)
            y_pix = yr * float(H)
        elif (0.0 <= xr <= 1000.0) and (0.0 <= yr <= 1000.0):
            if (W > 1000) or (H > 1000):
                mode_used = "norm1000"
                x_pix = xr / 1000.0 * float(W)
                y_pix = yr / 1000.0 * float(H)
            else:
                if (xr > W) or (yr > H):
                    mode_used = "norm1000"
                    x_pix = xr / 1000.0 * float(W)
                    y_pix = yr / 1000.0 * float(H)
                else:
                    mode_used = "pixel"
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
        "mode_used": mode_used,
        "raw_pair": [float(xr), float(yr)],
        "W": int(W),
        "H": int(H),
        "pixel_float": [float(x_pix), float(y_pix)],
        "pixel_int_before_clamp": [int(xi_raw), int(yi_raw)],
        "pixel_int_clamped": [int(xi), int(yi)],
        "in_bounds_before_clamp": bool(in_bounds_before_clamp),
    }
    if mode_used == "norm1000":
        dbg["norm1000_pair"] = [float(xr), float(yr)]
    if mode_used == "rel01":
        dbg["rel01_pair"] = [float(xr), float(yr)]
    return xi, yi, True, dbg


# Backward-compatible wrapper for older calls.
def extract_coordinates_auto(raw: str, W: int, H: int) -> Tuple[int, int, bool, Dict[str, Any]]:
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


# --------- NEW: wrap-around bbox helpers ---------
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


def bbox_after_shift_wrap_single(
    gt_xyxy: Tuple[float, float, float, float],
    dx: int,
    dy: int,
    W: int,
    H: int,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return the shifted bbox only when wrap-around keeps it as ONE intact rectangle.
    If wrap would split it into multiple parts, return None.
    """
    parts = bbox_after_shift_wrap(gt_xyxy, dx=dx, dy=dy, W=W, H=H)
    if len(parts) != 1:
        return None
    return parts[0]


def point_in_any_box(x: float, y: float, boxes: List[Tuple[float, float, float, float]]) -> bool:
    for b in boxes:
        if point_in_box(x, y, b):
            return True
    return False


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
    target_min = int(math.ceil(max_dim / max_ratio))  # ensure max_dim/target_min <= max_ratio

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
# Region loading (兼容三种格式)
# 1) {"bbox":[x1,y1,x2,y2]} abs pixels
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
        # format 1: {"bbox":[x1,y1,x2,y2]} absolute pixels
        b = obj.get("bbox", None)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                x1, y1, x2, y2 = map(float, b)
                x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
                y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
                specs.append({"kind": "abs_bbox_xyxy", "xyxy": (x_lo, y_lo, x_hi, y_hi)})
                continue
            except Exception:
                pass

        # format 2/3: {"bounds":{"x0":..,"x1":..,"y0":..,"y1":..}}
        #   - relative bounds -> kind="rel_bounds"
        #   - absolute bounds -> kind="abs_bbox_xyxy"
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
    """
    映射到“原图 (W,H)”绝对像素坐标。
    这里是 shift-only 固定画布，因此 region 坐标系始终是原图坐标系，不随内容移动。
    """
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
# BBoxUnion (raw box preprocessing; component merge happens later)
# -----------------------------
class BBoxUnion:
    """
    仅对每个 raw region bbox 做 shift 预处理（shift 默认 0）。
    相邻 region 的 connected-component merge 会在 make_region_components() 中完成，
    这样 center_inset_px / bbox_edge_margin_px 可以基于 merged component union 判断。
    """

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
            raise RuntimeError("All region bboxes are invalid after shift.")

        self.boxes: List[Tuple[float, float, float, float]] = processed
        self.region_bbox_count_after_merge = int(len(self.boxes))


# -----------------------------
# Region components
# -----------------------------
@dataclass
class RegionComponent:
    """
    A connected blind-zone component.

    boxes: the original region boxes in this component.  We keep the
    full union of these boxes, instead of replacing it by only the outer bbox,
    because a component may be non-rectangular, e.g. an L-shape.
    outer_xyxy: coarse outer bounding box, used only to produce candidate shift
    ranges.  Final validation is always against boxes union.
    """
    component_id: int
    boxes: List[Tuple[float, float, float, float]]
    outer_xyxy: Tuple[float, float, float, float]


def _normalize_xyxy_box(box: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
    try:
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return None
    xlo, xhi = (x1, x2) if x1 <= x2 else (x2, x1)
    ylo, yhi = (y1, y2) if y1 <= y2 else (y2, y1)
    if xhi <= xlo or yhi <= ylo:
        return None
    return (float(xlo), float(ylo), float(xhi), float(yhi))


def _overlap_len_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return float(min(a2, b2) - max(a1, b1))


def boxes_adjacent_or_overlap_4conn(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    *,
    tol: float = 1.0,
) -> bool:
    """
    4-connected adjacency for axis-aligned region boxes.

    Merge boxes that overlap in area or share/touch an edge within tol pixels.
    Pure diagonal corner touching is NOT treated as connected, because there is
    no continuous blind area between them.
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    tol = max(float(tol), 0.0)

    x_overlap = _overlap_len_1d(ax1, ax2, bx1, bx2)
    y_overlap = _overlap_len_1d(ay1, ay2, by1, by2)

    # Area overlap.
    if x_overlap > 0 and y_overlap > 0:
        return True

    # Vertical edge contact: x edges touch, y ranges overlap with positive length.
    x_edge_touch = (abs(ax2 - bx1) <= tol) or (abs(bx2 - ax1) <= tol)
    if x_edge_touch and y_overlap > 0:
        return True

    # Horizontal edge contact: y edges touch, x ranges overlap with positive length.
    y_edge_touch = (abs(ay2 - by1) <= tol) or (abs(by2 - ay1) <= tol)
    if y_edge_touch and x_overlap > 0:
        return True

    return False


def make_region_components(
    boxes: List[Tuple[float, float, float, float]],
    *,
    merge_connected: bool = True,
    touch_eps: float = 1.0,
) -> List[RegionComponent]:
    """
    Convert raw/shrunk region boxes into connected components.

    If merge_connected=True, adjacent/overlapping boxes are merged into one
    component.  If False, every valid box becomes its own component.
    """
    clean: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        nb = _normalize_xyxy_box(b)
        if nb is not None:
            clean.append(nb)

    if not clean:
        return []

    if not merge_connected:
        components: List[RegionComponent] = []
        for cid, b in enumerate(clean):
            components.append(RegionComponent(component_id=cid, boxes=[b], outer_xyxy=b))
        return components

    n = len(clean)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_adjacent_or_overlap_4conn(clean[i], clean[j], tol=touch_eps):
                union(i, j)

    grouped: Dict[int, List[Tuple[float, float, float, float]]] = defaultdict(list)
    for i, b in enumerate(clean):
        grouped[find(i)].append(b)

    comps_raw: List[Tuple[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = []
    for comp_boxes in grouped.values():
        outer = (
            float(min(b[0] for b in comp_boxes)),
            float(min(b[1] for b in comp_boxes)),
            float(max(b[2] for b in comp_boxes)),
            float(max(b[3] for b in comp_boxes)),
        )
        comps_raw.append((outer, comp_boxes))

    # Stable ordering: top-to-bottom, left-to-right, then larger components first.
    comps_raw.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2], -len(item[1])))

    components: List[RegionComponent] = []
    for cid, (outer, comp_boxes) in enumerate(comps_raw):
        components.append(RegionComponent(component_id=cid, boxes=list(comp_boxes), outer_xyxy=outer))
    return components


# -----------------------------
# Persistent region-component cache
# -----------------------------
def _box_to_json_list(box: Tuple[float, float, float, float]) -> List[float]:
    return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]


def region_component_to_json(c: RegionComponent) -> Dict[str, Any]:
    return {
        "component_id": int(c.component_id),
        "outer_xyxy": _box_to_json_list(c.outer_xyxy),
        "boxes": [_box_to_json_list(b) for b in c.boxes],
    }


def region_component_from_json(obj: Dict[str, Any]) -> Optional[RegionComponent]:
    try:
        cid = int(obj.get("component_id", 0))
        outer_raw = obj.get("outer_xyxy", None)
        boxes_raw = obj.get("boxes", None)
        if not (isinstance(outer_raw, list) and len(outer_raw) == 4):
            return None
        if not isinstance(boxes_raw, list):
            return None
        outer = tuple(map(float, outer_raw))
        boxes: List[Tuple[float, float, float, float]] = []
        for b in boxes_raw:
            if isinstance(b, list) and len(b) == 4:
                nb = _normalize_xyxy_box(tuple(map(float, b)))
                if nb is not None:
                    boxes.append(nb)
        if not boxes:
            return None
        return RegionComponent(component_id=cid, boxes=boxes, outer_xyxy=tuple(map(float, outer)))
    except Exception:
        return None


def load_region_component_cache_file(path: str) -> Dict[str, Any]:
    if not path:
        return {"version": 1, "entries": {}}
    try:
        if (not os.path.exists(path)) or os.path.getsize(path) == 0:
            return {"version": 1, "entries": {}}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "entries": {}}
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}
        return {"version": 1, "entries": entries}
    except Exception:
        # Bad/partial cache should never break evaluation; just rebuild it.
        return {"version": 1, "entries": {}}


def save_region_component_cache_file(path: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # Cache is only an acceleration; ignore write errors.
        pass


def build_region_component_cache_key(
    *,
    region_bbox_path: str,
    W: int,
    H: int,
    merge_connected_region: bool,
    merge_touch_eps: float,
) -> str:
    try:
        st = os.stat(region_bbox_path)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        size = int(st.st_size)
    except Exception:
        mtime_ns = -1
        size = -1

    payload = {
        "cache_version": 2,
        "region_bbox_path": os.path.abspath(str(region_bbox_path)),
        "region_file_mtime_ns": mtime_ns,
        "region_file_size": size,
        "W": int(W),
        "H": int(H),
        "merge_connected_region": bool(merge_connected_region),
        "merge_touch_eps": float(merge_touch_eps),
        "connectivity": "4conn_edge_touch_or_overlap_no_diagonal",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def components_from_json_list(raw: Any) -> List[RegionComponent]:
    out: List[RegionComponent] = []
    if not isinstance(raw, list):
        return out
    for obj in raw:
        if isinstance(obj, dict):
            c = region_component_from_json(obj)
            if c is not None:
                out.append(c)
    return out


def rect_inside_box(
    rect: Tuple[float, float, float, float],
    box: Tuple[float, float, float, float],
    *,
    eps: float = 1e-9,
) -> bool:
    x1, y1, x2, y2 = map(float, rect)
    bx1, by1, bx2, by2 = map(float, box)
    return (
        x1 >= bx1 - eps
        and y1 >= by1 - eps
        and x2 <= bx2 + eps
        and y2 <= by2 + eps
    )


def point_in_component_union(
    x: float,
    y: float,
    component: RegionComponent,
    *,
    eps: float = 1e-9,
) -> bool:
    return any(point_in_box(float(x), float(y), b) for b in component.boxes)


def rect_inside_union(
    rect: Tuple[float, float, float, float],
    boxes: List[Tuple[float, float, float, float]],
    *,
    eps: float = 1e-9,
) -> bool:
    """
    Exact check: whether rect is fully covered by the union of boxes.

    This is needed because merged adjacent regions may form a non-rectangular
    connected component.  Using only the component outer bbox would incorrectly
    count holes / missing cells as blind area.
    """
    x1, y1, x2, y2 = map(float, rect)
    if x2 <= x1 or y2 <= y1:
        return False
    if not boxes:
        return False

    if len(boxes) == 1:
        return rect_inside_box(rect, boxes[0], eps=eps)

    # Quick reject by outer bbox.
    outer = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    if not rect_inside_box(rect, outer, eps=eps):
        return False

    xs = {x1, x2}
    ys = {y1, y2}
    relevant: List[Tuple[float, float, float, float]] = []

    for bx1, by1, bx2, by2 in boxes:
        bx1, by1, bx2, by2 = map(float, (bx1, by1, bx2, by2))
        if bx2 <= x1 + eps or bx1 >= x2 - eps or by2 <= y1 + eps or by1 >= y2 - eps:
            continue
        relevant.append((bx1, by1, bx2, by2))
        xs.add(max(x1, bx1))
        xs.add(min(x2, bx2))
        ys.add(max(y1, by1))
        ys.add(min(y2, by2))

    if not relevant:
        return False

    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)

    # Partition rect by all relevant box boundaries.  Each open sub-rectangle
    # must be contained in at least one component box.
    for i in range(len(xs_sorted) - 1):
        cx1, cx2 = xs_sorted[i], xs_sorted[i + 1]
        if cx2 <= cx1 + eps:
            continue
        for j in range(len(ys_sorted) - 1):
            cy1, cy2 = ys_sorted[j], ys_sorted[j + 1]
            if cy2 <= cy1 + eps:
                continue
            subrect = (cx1, cy1, cx2, cy2)
            if not any(rect_inside_box(subrect, b, eps=eps) for b in relevant):
                return False

    return True


def center_inside_component_with_inset(
    cx: float,
    cy: float,
    component: RegionComponent,
    *,
    inset_x: float,
    inset_y: float,
    eps: float = 1e-9,
) -> bool:
    """
    center_inset_px against a merged component union.

    If inset is 0, this reduces to center-in-component.  If inset > 0, the
    rectangle centered at (cx,cy) with half-size inset must be fully covered by
    the same component union.
    """
    ix = max(float(inset_x), 0.0)
    iy = max(float(inset_y), 0.0)
    cx = float(cx)
    cy = float(cy)

    if ix <= 0.0 and iy <= 0.0:
        return point_in_component_union(cx, cy, component, eps=eps)

    center_rect = (cx - ix - eps, cy - iy - eps, cx + ix + eps, cy + iy + eps)
    return rect_inside_union(center_rect, component.boxes, eps=eps)


def bbox_inside_component_with_edge_margin(
    bbox_xyxy: Tuple[float, float, float, float],
    component: RegionComponent,
    *,
    edge_margin_x: float,
    edge_margin_y: float,
    eps: float = 1e-9,
) -> bool:
    """
    bbox_edge_margin_px against a merged component union.

    The moved bbox expanded by edge_margin must still be fully covered by the
    same connected component.  Therefore internal seams between adjacent raw
    boxes do not count as non-blind boundaries, but gaps/holes and the component
    exterior do count as non-blind area.
    """
    bx1, by1, bx2, by2 = map(float, bbox_xyxy)
    if bx2 <= bx1 or by2 <= by1:
        return False

    mx = max(float(edge_margin_x), 0.0)
    my = max(float(edge_margin_y), 0.0)
    expanded = (bx1 - mx - eps, by1 - my - eps, bx2 + mx + eps, by2 + my + eps)
    return rect_inside_union(expanded, component.boxes, eps=eps)


def bbox_in_any_component_with_edge_margin(
    bbox_xyxy: Tuple[float, float, float, float],
    components: List[RegionComponent],
    *,
    edge_margin_x: float,
    edge_margin_y: float,
) -> bool:
    return any(
        bbox_inside_component_with_edge_margin(
            bbox_xyxy,
            comp,
            edge_margin_x=edge_margin_x,
            edge_margin_y=edge_margin_y,
        )
        for comp in components
    )


def point_in_any_region_with_inset(
    x: float,
    y: float,
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    inset_x: float,
    inset_y: float,
    eps: float = 1e-9,
) -> bool:
    """
    Check if point (x,y) is inside ANY region after applying only an inward inset.
    Inset is clamped per-region to avoid collapsing small regions.
    """
    x = float(x)
    y = float(y)
    ix0 = float(inset_x)
    iy0 = float(inset_y)

    for (rx1, ry1, rx2, ry2) in region_boxes_xyxy:
        rx1 = float(rx1)
        ry1 = float(ry1)
        rx2 = float(rx2)
        ry2 = float(ry2)
        w = rx2 - rx1
        h = ry2 - ry1
        if w <= 0 or h <= 0:
            continue

        # clamp inset to avoid collapse (try to leave >= ~2px interior when possible)
        ix = min(ix0, max(0.0, 0.5 * w - 1.0))
        iy = min(iy0, max(0.0, 0.5 * h - 1.0))

        ex1 = rx1 + ix
        ex2 = rx2 - ix
        ey1 = ry1 + iy
        ey2 = ry2 - iy
        if ex2 <= ex1 or ey2 <= ey1:
            continue

        if (x >= ex1 - eps) and (x <= ex2 + eps) and (y >= ey1 - eps) and (y <= ey2 + eps):
            return True

    return False


# -------------------------
# Visualization
# -------------------------
def visualize_prediction_pixel(
    img: Image.Image,
    x: int,
    y: int,
    save_path: str,
    *,
    bbox_xyxy: Optional[Any] = None,  # single bbox or list of bboxes
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

    # bbox: support single or list of bboxes
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
# Shift-only transform (固定画布，移动图片：WRAP-AROUND 拼图式填充)
# -------------------------
def _canonical_shift_delta(d: int, size: int) -> int:
    """
    Wrap-around shift: only the class mod size matters.
    Canonicalize to a representative with small |d| to keep budgets stable.
    """
    size = int(size)
    if size <= 1:
        return 0
    d = int(d) % size
    hi = (size - 1) // 2
    if d > hi:
        d -= size
    return int(d)


def shift_image_on_canvas(img: Image.Image, dx: int, dy: int, fill=(0, 0, 0)) -> Image.Image:
    """
    固定画布为原图尺寸 (W,H)，把原图内容整体“环绕平移”(dx,dy)：
      - dx>0: 内容向右（左侧溢出的部分从右侧补回来）
      - dy>0: 内容向下（上侧溢出的部分从下侧补回来）
    不再产生黑边/空白，像拼图一样无缝拼接。

    fill 参数保留为兼容旧调用（wrap 模式下不会用到）。
    """
    W, H = img.size
    dx = _canonical_shift_delta(dx, W)
    dy = _canonical_shift_delta(dy, H)
    return ImageChops.offset(img, dx, dy)


def filled_pixels_for_shift(W: int, H: int, dx: int, dy: int) -> int:
    """
    Wrap-around shift 时没有黑色填充像素。
    但我们仍需要一个“变化预算”来限制 shift 幅度。

    这里沿用原来的面积公式，把它解释为：
      环绕拼接产生的“边界接缝条带面积”
    且必须使用 canonical dx/dy（因为 dx=W-1 等价于 -1）。
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


def _ceil_int(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def _floor_int(x: float) -> int:
    return int(math.floor(x + 1e-9))


def bbox_inside_region_with_edge_margin(
    bbox_xyxy: Tuple[float, float, float, float],
    region_xyxy: Tuple[float, float, float, float],
    *,
    edge_margin_x: float,
    edge_margin_y: float,
    eps: float = 1e-9,
) -> bool:
    """
    True only when the entire moved bbox is inside the target blind region
    and each bbox edge is strictly greater than edge_margin pixels away
    from the non-blind area outside that target region.

    In other words, require:
      left   edge distance from region left   boundary > edge_margin_x
      right  edge distance from region right  boundary > edge_margin_x
      top    edge distance from region top    boundary > edge_margin_y
      bottom edge distance from region bottom boundary > edge_margin_y
    """
    bx1, by1, bx2, by2 = map(float, bbox_xyxy)
    rx1, ry1, rx2, ry2 = map(float, region_xyxy)

    mx = float(edge_margin_x)
    my = float(edge_margin_y)
    if mx < 0 or my < 0:
        return False

    rx1d = rx1
    rx2d = rx2
    ry1d = ry1
    ry2d = ry2

    if bx2 <= bx1 or by2 <= by1 or rx2d <= rx1d or ry2d <= ry1d:
        return False

    # Strictly greater than the requested margin.  For edge_margin=100,
    # distances equal to exactly 100 px are rejected.
    return (
        (bx1 - rx1d) > mx
        and (rx2d - bx2) > mx
        and (by1 - ry1d) > my
        and (ry2d - by2) > my
    )


def bbox_in_any_region_with_edge_margin(
    bbox_xyxy: Tuple[float, float, float, float],
    region_boxes_xyxy: List[Tuple[float, float, float, float]],
    *,
    edge_margin_x: float,
    edge_margin_y: float,
) -> bool:
    for rbox in region_boxes_xyxy:
        if bbox_inside_region_with_edge_margin(
            bbox_xyxy,
            rbox,
            edge_margin_x=edge_margin_x,
            edge_margin_y=edge_margin_y,
        ):
            return True
    return False


def feasible_shift_ranges_for_region(
    *,
    gt_xyxy: Tuple[float, float, float, float],
    region_xyxy: Tuple[float, float, float, float],
    W: int,
    H: int,
    inset_x: float,
    inset_y: float,
    edge_margin_x: float,
    edge_margin_y: float,
) -> Optional[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    Wrap-around feasible ranges, now based on the WHOLE bbox rather than
    only bbox center.

    Requirement after shift, without wrap-splitting the bbox:
      moved_bbox.x1 is strictly more than edge_margin_x inside target region left boundary
      moved_bbox.x2 is strictly more than edge_margin_x inside target region right boundary
      moved_bbox.y1 is strictly more than edge_margin_y inside target region top boundary
      moved_bbox.y2 is strictly more than edge_margin_y inside target region bottom boundary

    This implements: after moving INTO a blind region, every edge of the moved
    bbox must be >100 px away from the non-blind area when edge_margin=100.

    The old center_inset args are kept for compatibility. We take the stricter
    of center_inset and bbox_edge_margin, so old configs with larger insets still
    behave conservatively.
    """
    x1, y1, x2, y2 = map(float, gt_xyxy)
    rx1, ry1, rx2, ry2 = map(float, region_xyxy)

    bw = x2 - x1
    bh = y2 - y1
    rw = rx2 - rx1
    rh = ry2 - ry1
    if bw <= 0 or bh <= 0 or rw <= 0 or rh <= 0:
        return None

    # edge margin to non-blind region.  Keep old inset as an extra conservative margin.
    mx = max(float(edge_margin_x), float(inset_x), 0.0)
    my = max(float(edge_margin_y), float(inset_y), 0.0)

    rx1d = rx1 + mx
    rx2d = rx2 - mx
    ry1d = ry1 + my
    ry2d = ry2 - my

    # Need enough room for the entire bbox plus margins.
    if (rx2d - rx1d) < bw or (ry2d - ry1d) < bh:
        return None

    W = int(W)
    H = int(H)
    if W <= 1 or H <= 1:
        return None

    dx_dom_lo = -W // 2
    dx_dom_hi = (W - 1) // 2
    dy_dom_lo = -H // 2
    dy_dom_hi = (H - 1) // 2

    dx_ranges: List[Tuple[int, int]] = []
    dy_ranges: List[Tuple[int, int]] = []

    # Strict integer ranges:
    #   x1 + dx > rx1 + margin  -> dx >  rx1 + margin - x1
    #   x2 + dx < rx2 - margin  -> dx <  rx2 - margin - x2
    # For integer dx, use floor(lower)+1 and ceil(upper)-1.
    for k in (-1, 0, 1):
        lo = int(math.floor((rx1d + k * W) - x1) + 1)
        hi = int(math.ceil((rx2d + k * W) - x2) - 1)
        lo = max(lo, dx_dom_lo)
        hi = min(hi, dx_dom_hi)
        if hi >= lo:
            dx_ranges.append((int(lo), int(hi)))

    # Strict integer ranges for y, analogous to x.
    for k in (-1, 0, 1):
        lo = int(math.floor((ry1d + k * H) - y1) + 1)
        hi = int(math.ceil((ry2d + k * H) - y2) - 1)
        lo = max(lo, dy_dom_lo)
        hi = min(hi, dy_dom_hi)
        if hi >= lo:
            dy_ranges.append((int(lo), int(hi)))

    if not dx_ranges or not dy_ranges:
        return None
    return dx_ranges, dy_ranges

def _clamp_to_range(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _nearest_to_zero_in_range(lo: int, hi: int, *, forbid_zero: bool) -> Optional[int]:
    if hi < lo:
        return None
    if not forbid_zero and (lo <= 0 <= hi):
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
    """
    给一个整数区间，构造少量代表性候选（用于小规模枚举），避免全枚举。
    """
    if hi < lo:
        return []
    vals = set()
    vals.add(lo)
    vals.add(hi)

    for t in [0, -1, 1, -2, 2, -4, 4, -8, 8]:
        vals.add(_clamp_to_range(t, lo, hi))

    nz = _nearest_to_zero_in_range(lo, hi, forbid_zero=True)
    if nz is not None:
        vals.add(int(nz))

    if hi >= 1:
        vals.add(max(1, lo))
    if lo <= -1:
        vals.add(min(-1, hi))

    out = sorted(vals)
    return out


# -------------------------
# NEW: "move directly INTO region"
# -------------------------
def _trunc_div(v: int, denom: float) -> int:
    return int(math.trunc(float(v) / float(denom)))


def _center_after_wrap(cx: float, cy: float, dx: int, dy: int, W: int, H: int) -> Tuple[float, float]:
    return ((float(cx) + float(dx)) % float(W), (float(cy) + float(dy)) % float(H))


def _center_in_any_region(
    cx: float,
    cy: float,
    *,
    dx: int,
    dy: int,
    W: int,
    H: int,
    region_boxes_proc: List[Tuple[float, float, float, float]],
    inset_x: float,
    inset_y: float,
) -> bool:
    x2, y2 = _center_after_wrap(cx, cy, dx, dy, W, H)
    return point_in_any_region_with_inset(
        x2,
        y2,
        region_boxes_proc,
        inset_x=float(inset_x),
        inset_y=float(inset_y),
    )


def _style_score_penalty(style: str, dx: int, dy: int, W: int, H: int) -> float:
    """
    Keep the same style preference logic as before (soft constraints).
    Lower is better.
    """
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


def pick_best_shift_to_avoid_regions_style(
    *,
    cx: float,
    cy: float,
    W: int,
    H: int,
    budget_px: int,
    region_boxes_proc: List[Tuple[float, float, float, float]],
    inset_x: float,
    inset_y: float,
    style: str,
    prefer_dx: int,
    prefer_dy: int,
    forbid_zero: bool = False,
    max_step: int = 64,
) -> Optional[Dict[str, Any]]:
    """
    Find a small (dx,dy) such that center AFTER shift is OUTSIDE all regions.
    We search a small candidate set (not full enumeration).
    """
    W = int(W)
    H = int(H)
    dx_dom_lo = -W // 2
    dx_dom_hi = (W - 1) // 2
    dy_dom_lo = -H // 2
    dy_dom_hi = (H - 1) // 2

    mags = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 48, 64]
    mags = [m for m in mags if m <= max_step]

    def build_vals(dom_lo: int, dom_hi: int, pref: int) -> List[int]:
        vals = set()
        for m in mags:
            for sgn in (-1, 1):
                vals.add(_clamp_to_range(sgn * m, dom_lo, dom_hi))
        for d in [0, -1, 1, -2, 2, -4, 4, -8, 8]:
            vals.add(_clamp_to_range(int(pref) + d, dom_lo, dom_hi))
        return sorted(vals)

    dx_vals = build_vals(dx_dom_lo, dx_dom_hi, int(prefer_dx))
    dy_vals = build_vals(dy_dom_lo, dy_dom_hi, int(prefer_dy))

    best = None
    for dx0 in dx_vals:
        for dy0 in dy_vals:
            dx = _canonical_shift_delta(int(dx0), W)
            dy = _canonical_shift_delta(int(dy0), H)
            if forbid_zero and dx == 0 and dy == 0:
                continue

            filled = filled_pixels_for_shift(W, H, dx, dy)
            if filled > budget_px:
                continue

            if _center_in_any_region(
                cx,
                cy,
                dx=dx,
                dy=dy,
                W=W,
                H=H,
                region_boxes_proc=region_boxes_proc,
                inset_x=inset_x,
                inset_y=inset_y,
            ):
                continue

            s = float(filled)
            s += _style_score_penalty(style, dx, dy, W, H)
            s += 0.01 * (abs(dx - int(prefer_dx)) + abs(dy - int(prefer_dy)))

            key = (s, filled, abs(dx) + abs(dy), abs(dx), abs(dy))
            if best is None or key < best["key"]:
                best = {"dx": dx, "dy": dy, "filled_px": filled, "score": s, "key": key}

    if best is None:
        return None
    best.pop("key", None)
    return best


def reduce_enter_shift_to_fraction_but_keep_outside(
    *,
    cx: float,
    cy: float,
    dx_full: int,
    dy_full: int,
    W: int,
    H: int,
    budget_px: int,
    region_boxes_proc: List[Tuple[float, float, float, float]],
    inset_x: float,
    inset_y: float,
    style: str,
) -> Dict[str, Any]:
    """
    旧逻辑保留但不再使用：
    Start from (dx_full,dy_full) that would ENTER region.
    Apply only a fraction (default 1/3, trunc toward 0), and if it still enters,
    reduce further (1/4,1/5,...) until OUTSIDE.
    """
    dx_full = _canonical_shift_delta(int(dx_full), int(W))
    dy_full = _canonical_shift_delta(int(dy_full), int(H))

    denoms = [3, 4, 5, 6, 8, 10, 12, 16, 24, 32]

    for denom in denoms:
        dx = _canonical_shift_delta(_trunc_div(dx_full, denom), int(W))
        dy = _canonical_shift_delta(_trunc_div(dy_full, denom), int(H))

        filled = filled_pixels_for_shift(W, H, dx, dy)
        if filled > budget_px:
            continue

        inside = _center_in_any_region(
            cx,
            cy,
            dx=dx,
            dy=dy,
            W=W,
            H=H,
            region_boxes_proc=region_boxes_proc,
            inset_x=inset_x,
            inset_y=inset_y,
        )
        if not inside:
            return {
                "dx": int(dx),
                "dy": int(dy),
                "filled_px": int(filled),
                "fraction": f"1/{denom}",
                "mode": "fraction_keep_outside",
            }

    cand = pick_best_shift_to_avoid_regions_style(
        cx=float(cx),
        cy=float(cy),
        W=int(W),
        H=int(H),
        budget_px=int(budget_px),
        region_boxes_proc=region_boxes_proc,
        inset_x=float(inset_x),
        inset_y=float(inset_y),
        style=str(style),
        prefer_dx=0,
        prefer_dy=0,
        forbid_zero=False,
        max_step=64,
    )
    if cand is None:
        return {"dx": 0, "dy": 0, "filled_px": 0, "fraction": "1/INF", "mode": "failed_to_avoid_regions"}

    return {
        "dx": int(cand["dx"]),
        "dy": int(cand["dy"]),
        "filled_px": int(cand["filled_px"]),
        "fraction": "exit_search",
        "mode": "exit_regions_search",
    }



def _expand_int_ranges_unique(ranges: List[Tuple[int, int]]) -> List[int]:
    """
    Legacy helper.  Kept for compatibility, but the fast search below no longer
    expands full dx/dy ranges into all integer values.
    """
    vals: Set[int] = set()
    for lo, hi in ranges:
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            continue
        vals.update(range(lo, hi + 1))
    return sorted(vals)


def _range_contains_int(v: int, ranges: List[Tuple[int, int]]) -> bool:
    v = int(v)
    for lo, hi in ranges:
        if int(lo) <= v <= int(hi):
            return True
    return False


def _representative_ints_from_ranges(ranges: List[Tuple[int, int]]) -> List[int]:
    """
    Return a tiny set of representative dx/dy values from each feasible range.
    This avoids the old full integer expansion.
    """
    vals: Set[int] = set()
    for lo, hi in ranges:
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            continue
        vals.add(lo)
        vals.add(hi)
        vals.add((lo + hi) // 2)
        for t in (0, -1, 1, -2, 2, -4, 4, -8, 8, -16, 16, -32, 32):
            if lo <= t <= hi:
                vals.add(t)
        nz = _nearest_to_zero_in_range(lo, hi, forbid_zero=True)
        if nz is not None:
            vals.add(int(nz))
    return sorted(vals)


def _axis_candidate_score(style: str, delta: int, axis: str) -> Tuple[float, int]:
    """
    Cheap axis-wise priority used only to cap candidate counts.
    Final ranking still uses the original full style score.
    """
    d = int(delta)
    style = str(style)
    if axis == "x":
        if style == "vertical":
            return (abs(d), abs(d))
        if style == "prefer_right":
            return ((0 if d > 0 else 1e9) + abs(d), abs(d))
        if style == "diagonal":
            return ((0 if d != 0 else 1e9) + abs(d), abs(d))
        return (abs(d), abs(d))
    else:
        if style == "horizontal":
            return (abs(d), abs(d))
        if style == "prefer_down":
            return ((0 if d > 0 else 1e9) + abs(d), abs(d))
        if style == "diagonal":
            return ((0 if d != 0 else 1e9) + abs(d), abs(d))
        return (abs(d), abs(d))


def _dedupe_int_keep_order(vals: List[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for v in vals:
        v = int(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _build_fast_axis_positions(
    *,
    target_component: RegionComponent,
    axis: str,
    span: float,
    canvas_size: int,
    orig_lo: float,
    margin: float,
    ranges: List[Tuple[int, int]],
    style: str,
    max_axis_candidates: int = 96,
) -> List[int]:
    """
    Build a small but high-coverage set of candidate positions for the expanded
    rectangle along one axis.

    Instead of enumerating every integer shift, we exploit a geometry fact:
    if a fixed rectangle can be placed inside an orthogonal union, there is a
    valid placement with at least one rectangle edge aligned to a union boundary.
    Therefore we try boundary-aligned placements plus a few representative
    shift-range placements.
    """
    span = float(span)
    canvas_size = int(canvas_size)
    if span <= 0 or canvas_size <= 1 or span > canvas_size:
        return []

    raw_positions: List[int] = []

    # Candidate positions from component outer bbox and raw boxes.
    boxes = [target_component.outer_xyxy] + list(target_component.boxes)
    for b in boxes:
        x1, y1, x2, y2 = map(float, b)
        lo, hi = (x1, x2) if axis == "x" else (y1, y2)
        if hi - lo + 1e-9 < span:
            # A single raw box may be too small, but its boundary can still be
            # useful for a multi-box component.  Keep boundary candidates anyway.
            pass
        for pos in (lo, hi - span, (lo + hi - span) * 0.5):
            if -1e-6 <= pos <= canvas_size - span + 1e-6:
                raw_positions.append(int(round(pos)))

    # Candidate positions from feasible shift ranges.
    # expanded_lo = moved_bbox_lo - margin = orig_lo + delta - margin
    for d in _representative_ints_from_ranges(ranges):
        pos = float(orig_lo) + float(d) - float(margin)
        if -1e-6 <= pos <= canvas_size - span + 1e-6:
            raw_positions.append(int(round(pos)))

    # Always try canvas-safe extremes; they are cheap and often useful.
    raw_positions.extend([0, int(round(canvas_size - span))])

    raw_positions = [max(0, min(int(round(canvas_size - span)), int(v))) for v in raw_positions]
    raw_positions = _dedupe_int_keep_order(raw_positions)

    scored: List[Tuple[Tuple[float, int], int]] = []
    for pos in raw_positions:
        delta = _canonical_shift_delta(int(round(float(pos) + float(margin) - float(orig_lo))), canvas_size)
        # Keep only candidates consistent with the coarse feasible ranges.
        if ranges and not _range_contains_int(delta, ranges):
            continue
        scored.append((_axis_candidate_score(style, delta, axis), int(pos)))

    scored.sort(key=lambda x: x[0])
    if max_axis_candidates > 0 and len(scored) > max_axis_candidates:
        scored = scored[: int(max_axis_candidates)]
    return _dedupe_int_keep_order([pos for _, pos in scored])


def pick_best_shift_for_region_style(
    *,
    gt_xyxy: Tuple[float, float, float, float],
    target_component: RegionComponent,
    dx_ranges: List[Tuple[int, int]],
    dy_ranges: List[Tuple[int, int]],
    W: int,
    H: int,
    budget_px: int,
    style: str,
    forbid_zero: bool,
    inset_x: float,
    inset_y: float,
    edge_margin_x: float,
    edge_margin_y: float,
) -> Optional[Dict[str, Any]]:
    """
    Fast wrap-aware shift search against ONE merged region component.

    The previous implementation expanded dx_ranges and dy_ranges into every
    integer value and then used a nested loop.  On 3840x2160 images this can
    produce hundreds of thousands of candidates per component.

    New behavior:
      - search boundary-aligned placements of the EXPANDED bbox;
      - cap candidate count per axis;
      - still run the exact component-union validation for center_inset_px and
        bbox_edge_margin_px;
      - return the best validated candidate under the original style score.
    """
    W = int(W)
    H = int(H)
    budget_px = int(budget_px)

    x1, y1, x2, y2 = map(float, gt_xyxy)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    mx = max(float(edge_margin_x), 0.0)
    my = max(float(edge_margin_y), 0.0)
    expanded_w = bw + 2.0 * mx
    expanded_h = bh + 2.0 * my
    if expanded_w > W or expanded_h > H:
        return None

    # This limit is intentionally conservative.  96x96 is at most 9216 checks
    # per component, instead of potentially millions.
    x_positions = _build_fast_axis_positions(
        target_component=target_component,
        axis="x",
        span=expanded_w,
        canvas_size=W,
        orig_lo=x1,
        margin=mx,
        ranges=dx_ranges,
        style=str(style),
        max_axis_candidates=96,
    )
    y_positions = _build_fast_axis_positions(
        target_component=target_component,
        axis="y",
        span=expanded_h,
        canvas_size=H,
        orig_lo=y1,
        margin=my,
        ranges=dy_ranges,
        style=str(style),
        max_axis_candidates=96,
    )
    if not x_positions or not y_positions:
        return None

    best = None
    checked_pairs = 0
    budget_ok_pairs = 0
    center_ok_pairs = 0
    margin_ok_pairs = 0

    def style_penalty(dx: int, dy: int) -> float:
        return _style_score_penalty(str(style), int(dx), int(dy), int(W), int(H))

    # Evaluate cheaper-looking candidates first.  We still keep the best among
    # all capped boundary candidates.
    pairs: List[Tuple[Tuple[float, int, int], int, int, int, int]] = []
    for ex1 in x_positions:
        dx = _canonical_shift_delta(int(round(float(ex1) + mx - x1)), W)
        if dx_ranges and not _range_contains_int(dx, dx_ranges):
            continue
        for ey1 in y_positions:
            dy = _canonical_shift_delta(int(round(float(ey1) + my - y1)), H)
            if dy_ranges and not _range_contains_int(dy, dy_ranges):
                continue
            filled = filled_pixels_for_shift(W, H, dx, dy)
            cheap_key = (float(filled) + style_penalty(dx, dy), abs(dx) + abs(dy), abs(dx))
            pairs.append((cheap_key, int(ex1), int(ey1), int(dx), int(dy)))

    pairs.sort(key=lambda t: t[0])

    for _, ex1, ey1, dx, dy in pairs:
        checked_pairs += 1
        if forbid_zero and dx == 0 and dy == 0:
            continue

        filled = filled_pixels_for_shift(W, H, dx, dy)
        if filled > budget_px:
            continue
        budget_ok_pairs += 1

        moved_bbox = bbox_after_shift_wrap_single(gt_xyxy, dx=dx, dy=dy, W=W, H=H)
        if moved_bbox is None:
            continue

        mcx, mcy = bbox_center_xy(moved_bbox)
        if not center_inside_component_with_inset(
            mcx,
            mcy,
            target_component,
            inset_x=float(inset_x),
            inset_y=float(inset_y),
        ):
            continue
        center_ok_pairs += 1

        if not bbox_inside_component_with_edge_margin(
            moved_bbox,
            target_component,
            edge_margin_x=float(edge_margin_x),
            edge_margin_y=float(edge_margin_y),
        ):
            continue
        margin_ok_pairs += 1

        s = float(filled) + style_penalty(dx, dy)
        key = (s, filled, abs(dx) + abs(dy), abs(dx), abs(dy))
        if best is None or key < best["key"]:
            best = {
                "dx": int(dx),
                "dy": int(dy),
                "filled_px": int(filled),
                "score": float(s),
                "key": key,
                "search_mode": "fast_boundary_component_union_shifts",
                "checked_pairs": int(checked_pairs),
                "budget_ok_pairs": int(budget_ok_pairs),
                "center_ok_pairs": int(center_ok_pairs),
                "margin_ok_pairs": int(margin_ok_pairs),
                "x_candidates": int(len(x_positions)),
                "y_candidates": int(len(y_positions)),
                "candidate_pairs": int(len(pairs)),
            }

            # Ultra-fast mode: pairs are already sorted by the same main
            # objective, so return the first exact-valid placement for this
            # component instead of scanning the remaining candidates.
            break

    if best is None:
        return None
    best.pop("key", None)
    return best

def build_multi_run_shift_plans(
    *,
    gt_xyxy: Tuple[float, float, float, float],
    region_boxes_proc: List[Tuple[float, float, float, float]],
    W: int,
    H: int,
    budget_px: int,
    inset_x: float,
    inset_y: float,
    edge_margin_x: float,
    edge_margin_y: float,
    num_runs: int,
    u0: BBoxUnion,
    merge_connected_region: bool = True,
    merge_touch_eps: float = 1.0,
    region_components: Optional[List[RegionComponent]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Behavior:
      - First merge adjacent/overlapping region boxes into connected components.
      - Search shifts that move the whole GT bbox into ONE target component.
      - The moved bbox must remain one intact rectangle after wrap-around.
      - center_inset_px is checked against the full union of the component.
      - bbox_edge_margin_px is checked by expanding moved bbox and requiring
        the expanded rectangle to be fully covered by the same component union.
      - Internal seams between adjacent raw boxes do NOT count as non-blind area;
        holes/gaps and component exterior DO count as non-blind area.
      - If one candidate shift fails, keep searching all other integer shifts.
      - Prefer unused components across runs; if none works, retry all components.
    """
    _ = u0
    eff_inset_x = max(float(inset_x), 0.0)
    eff_inset_y = max(float(inset_y), 0.0)
    eff_edge_margin_x = max(float(edge_margin_x), 0.0)
    eff_edge_margin_y = max(float(edge_margin_y), 0.0)

    if region_components is not None:
        components = list(region_components)
        components_source = "precomputed_or_disk_cached"
    else:
        components = make_region_components(
            region_boxes_proc,
            merge_connected=bool(merge_connected_region),
            touch_eps=float(merge_touch_eps),
        )
        components_source = "computed_inside_build_multi_run_shift_plans"

    styles_base = ["horizontal", "vertical", "diagonal", "prefer_right", "prefer_down"]
    styles: List[str] = [styles_base[i % len(styles_base)] for i in range(num_runs)]

    bbox0 = bbox_after_shift_wrap_single(gt_xyxy, dx=0, dy=0, W=W, H=H)
    bbox_in_any0 = False
    if bbox0 is not None:
        bbox_in_any0 = bbox_in_any_component_with_edge_margin(
            bbox0,
            components,
            edge_margin_x=float(eff_edge_margin_x),
            edge_margin_y=float(eff_edge_margin_y),
        )

    component_sizes = [len(c.boxes) for c in components]
    debug: Dict[str, Any] = {
        "budget_px": int(budget_px),
        "bbox_in_any_component_with_edge_margin_at_dxdy0": bool(bbox_in_any0),
        "num_regions_proc": int(len(region_boxes_proc)),
        "num_region_components": int(len(components)),
        "component_sizes": component_sizes,
        "merge_connected_region": bool(merge_connected_region),
        "merge_touch_eps": float(merge_touch_eps),
        "center_inset_xy": [float(eff_inset_x), float(eff_inset_y)],
        "bbox_edge_margin_xy": [float(eff_edge_margin_x), float(eff_edge_margin_y)],
        "shift_mode": "wrap",
        "bbox_integrity_required": True,
        "shift_search_mode": "fast_boundary_component_union_search",
        "rule": (
            "merge adjacent/overlapping blind-region boxes into connected components; "
            "fast boundary-search shifts into each merged component; require center_inset_px and "
            "bbox_edge_margin_px to be satisfied against the full union of boxes "
            "inside the same component, not against each raw box and not against "
            "the global union of all regions; reject bbox splitting across wrap seam"
        ),
        "canonical_domain_dx": [-int(W) // 2, (int(W) - 1) // 2],
        "canonical_domain_dy": [-int(H) // 2, (int(H) - 1) // 2],
    }

    plans: List[Dict[str, Any]] = []
    used_components: Set[int] = set()

    if not components:
        for run_i, style in enumerate(styles, 1):
            plans.append({
                "run": int(run_i),
                "skipped": True,
                "reason": "no_valid_region_components",
                "style": str(style),
                "center_inset_xy": [float(eff_inset_x), float(eff_inset_y)],
                "bbox_edge_margin_xy": [float(eff_edge_margin_x), float(eff_edge_margin_y)],
                "search_mode": "fast_boundary_component_union_search",
            })
        debug["used_components_count"] = 0
        return plans, debug

    for run_i, style in enumerate(styles, 1):
        best_global = None

        search_passes = [
            (
                "unused_components_fast",
                [cid for cid in range(len(components)) if cid not in used_components],
            ),
            ("all_components_fast_fallback", list(range(len(components)))),
        ]

        for search_pass, component_ids in search_passes:
            if best_global is not None:
                break
            if not component_ids:
                continue

            for cid in component_ids:
                comp = components[cid]
                outer = comp.outer_xyxy

                already_inside_this_component = False
                if bbox0 is not None:
                    bcx0, bcy0 = bbox_center_xy(bbox0)
                    already_inside_this_component = (
                        center_inside_component_with_inset(
                            bcx0,
                            bcy0,
                            comp,
                            inset_x=float(eff_inset_x),
                            inset_y=float(eff_inset_y),
                        )
                        and bbox_inside_component_with_edge_margin(
                            bbox0,
                            comp,
                            edge_margin_x=float(eff_edge_margin_x),
                            edge_margin_y=float(eff_edge_margin_y),
                        )
                    )

                # Coarse candidate ranges from component outer bbox.  Final check
                # below uses exact component union, so holes/non-rectangular shapes
                # are handled correctly.
                coarse_margin_x = max(eff_inset_x, eff_edge_margin_x)
                coarse_margin_y = max(eff_inset_y, eff_edge_margin_y)
                rngs = feasible_shift_ranges_for_region(
                    gt_xyxy=gt_xyxy,
                    region_xyxy=outer,
                    W=W,
                    H=H,
                    inset_x=0.0,
                    inset_y=0.0,
                    edge_margin_x=coarse_margin_x,
                    edge_margin_y=coarse_margin_y,
                )
                if rngs is None:
                    continue

                dx_ranges, dy_ranges = rngs

                cand_full = pick_best_shift_for_region_style(
                    gt_xyxy=gt_xyxy,
                    target_component=comp,
                    dx_ranges=dx_ranges,
                    dy_ranges=dy_ranges,
                    W=W,
                    H=H,
                    budget_px=budget_px,
                    style=style,
                    forbid_zero=True,
                    inset_x=eff_inset_x,
                    inset_y=eff_inset_y,
                    edge_margin_x=eff_edge_margin_x,
                    edge_margin_y=eff_edge_margin_y,
                )

                if cand_full is not None:
                    dx = int(cand_full["dx"])
                    dy = int(cand_full["dy"])
                    filled = int(cand_full["filled_px"])
                    applied_fraction = "full"
                    fraction_mode = "enter_component_directly_center_inset_bbox_edge_margin_fast"
                    dx_full = int(dx)
                    dy_full = int(dy)
                    checked_pairs = int(cand_full.get("checked_pairs", 0))
                    budget_ok_pairs = int(cand_full.get("budget_ok_pairs", 0))
                    center_ok_pairs = int(cand_full.get("center_ok_pairs", 0))
                    margin_ok_pairs = int(cand_full.get("margin_ok_pairs", 0))
                else:
                    # If bbox already satisfies both center inset and edge margin
                    # in this component but no non-zero shift works, allow zero as
                    # a last resort.
                    if already_inside_this_component:
                        dx = 0
                        dy = 0
                        filled = 0
                        applied_fraction = "zero"
                        fraction_mode = "already_inside_component_with_center_inset_bbox_edge_margin_zero_fallback"
                        dx_full = 0
                        dy_full = 0
                        checked_pairs = 0
                        budget_ok_pairs = 0
                        center_ok_pairs = 0
                        margin_ok_pairs = 0
                    else:
                        continue

                s = float(filled) + _style_score_penalty(str(style), int(dx), int(dy), int(W), int(H))
                key = (s, filled, abs(dx) + abs(dy), abs(dx), abs(dy))

                if best_global is None or key < best_global["key"]:
                    best_global = {
                        "key": key,
                        "cid": int(cid),
                        "component_id": int(comp.component_id),
                        "component_outer": tuple(map(float, comp.outer_xyxy)),
                        "component_boxes": [tuple(map(float, b)) for b in comp.boxes],
                        "component_size": int(len(comp.boxes)),
                        "style": str(style),
                        "dx_full": int(dx_full),
                        "dy_full": int(dy_full),
                        "dx": int(dx),
                        "dy": int(dy),
                        "filled_px": int(filled),
                        "applied_fraction": str(applied_fraction),
                        "fraction_mode": str(fraction_mode),
                        "search_pass": str(search_pass),
                        "component_reused": bool(cid in used_components),
                        "checked_pairs": int(checked_pairs),
                        "budget_ok_pairs": int(budget_ok_pairs),
                        "center_ok_pairs": int(center_ok_pairs),
                        "margin_ok_pairs": int(margin_ok_pairs),
                    }
                    # Fast mode: do not scan all other components after a valid one is found.
                    break

        if best_global is None:
            plans.append({
                "run": int(run_i),
                "skipped": True,
                "reason": "no_feasible_shift_into_component_with_center_inset_bbox_edge_margin_after_fast_search",
                "style": str(style),
                "center_inset_xy": [float(eff_inset_x), float(eff_inset_y)],
                "bbox_edge_margin_xy": [float(eff_edge_margin_x), float(eff_edge_margin_y)],
                "search_mode": "fast_boundary_component_union_search",
            })
            continue

        used_components.add(int(best_global["cid"]))
        plans.append({
            "run": int(run_i),
            "skipped": False,
            "style": str(best_global["style"]),
            # Keep region_id for backward compatibility, but it now means merged component id.
            "region_id": int(best_global["component_id"]),
            "component_id": int(best_global["component_id"]),
            "component_size": int(best_global["component_size"]),
            "component_outer_xyxy_proc": [float(x) for x in best_global["component_outer"]],
            "component_boxes_xyxy_proc": [[float(v) for v in b] for b in best_global["component_boxes"]],
            # Backward-compatible field: visualization/old readers can still use a rectangle.
            "region_xyxy_proc": [float(x) for x in best_global["component_outer"]],
            "dx": int(best_global["dx"]),
            "dy": int(best_global["dy"]),
            "dx_full": int(best_global["dx_full"]),
            "dy_full": int(best_global["dy_full"]),
            "applied_fraction": str(best_global["applied_fraction"]),
            "fraction_mode": str(best_global["fraction_mode"]),
            "filled_px": int(best_global["filled_px"]),
            "filled_ratio": (0.0 if (W * H) == 0 else float(best_global["filled_px"]) / float(W * H)),
            "center_inset_xy": [float(eff_inset_x), float(eff_inset_y)],
            "bbox_edge_margin_xy": [float(eff_edge_margin_x), float(eff_edge_margin_y)],
            "search_mode": "fast_boundary_component_union_search",
            "search_pass": str(best_global["search_pass"]),
            "region_reused": bool(best_global["component_reused"]),
            "component_reused": bool(best_global["component_reused"]),
            "checked_pairs": int(best_global["checked_pairs"]),
            "budget_ok_pairs": int(best_global["budget_ok_pairs"]),
            "center_ok_pairs": int(best_global["center_ok_pairs"]),
            "margin_ok_pairs": int(best_global["margin_ok_pairs"]),
        })

    debug["used_components_count"] = int(len(used_components))
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


_ORIGINAL_CORRECTNESS_CACHE: Dict[str, Dict[str, str]] = {}


def get_original_correctness_from_subset(subset_path: Any, key: Any) -> Optional[str]:
    if not subset_path or not key:
        return None
    subset_path_s = str(subset_path)
    key_s = str(key)
    if subset_path_s not in _ORIGINAL_CORRECTNESS_CACHE:
        correctness_by_key: Dict[str, str] = {}
        try:
            for rec in load_records_from_path(subset_path_s):
                rec_key = rec.get("key", None)
                if not rec_key:
                    rec_key = stable_key_from_screenspot(rec)
                c = normalize_correctness(rec.get("correctness", "wrong_format"))
                if c != "skipped":
                    correctness_by_key[str(rec_key)] = c
        except Exception:
            correctness_by_key = {}
        _ORIGINAL_CORRECTNESS_CACHE[subset_path_s] = correctness_by_key
    return _ORIGINAL_CORRECTNESS_CACHE[subset_path_s].get(key_s)


def get_original_accuracy_from_subset(subset_path: Any) -> Optional[float]:
    if not subset_path:
        return None
    subset_path_s = str(subset_path)
    if subset_path_s not in _ORIGINAL_CORRECTNESS_CACHE:
        get_original_correctness_from_subset(subset_path_s, "__missing_key__")
    vals = list(_ORIGINAL_CORRECTNESS_CACHE.get(subset_path_s, {}).values())
    if not vals:
        return None
    correct = sum(1 for c in vals if normalize_correctness(c) == "correct")
    return float(correct) / float(len(vals))


def get_original_correctness(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("original_correctness", "source_correctness", "baseline_correctness", "correctness_before_transform"):
        if k in obj:
            c = normalize_correctness(obj.get(k))
            if c != "skipped":
                return c

    debug = obj.get("_debug", None)
    if isinstance(debug, dict):
        for k in ("original_correctness", "source_correctness", "baseline_correctness", "correctness_before_transform"):
            if k in debug:
                c = normalize_correctness(debug.get(k))
                if c != "skipped":
                    return c
        subset_c = get_original_correctness_from_subset(debug.get("subset_path", None), obj.get("key", None))
        if subset_c is not None:
            return subset_c
    return None


def correctness_with_original_for_skipped(obj: Dict[str, Any], correctness: Any) -> str:
    c = normalize_correctness(correctness)
    if c != "skipped":
        return c
    original_c = get_original_correctness(obj)
    return original_c if original_c is not None else c


def stats_to_metrics_including_skipped_original_acc(s: Stats, original_acc: Optional[float]) -> Dict[str, Any]:
    baseline_acc = 0.0 if original_acc is None else float(original_acc)
    denom = int(s.scored + s.skipped)
    correct_adjusted = float(s.correct) + float(s.skipped) * baseline_acc
    return {
        "num_total": s.total,
        "num_scored_including_skipped": denom,
        "num_scored": s.scored,
        "num_skipped": s.skipped,
        "num_correct": s.correct,
        "num_correct_adjusted": correct_adjusted,
        "original_acc_used_for_skipped": baseline_acc,
        "acc": (0.0 if denom == 0 else correct_adjusted / float(denom)),
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
    """
    Sample-level vote:
    - Return "correct" only when strict majority (>50%) are correct.
    - Otherwise vote among non-correct labels (tie-breaker: first appearance).
    """
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
        for run_k, run_v in trials.items():
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
        for run_k, run_v in trials.items():
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


def summarize_output_paths(output_paths: List[str], *, num_runs: int = 0) -> Dict[str, Any]:
    overall_stats = Stats()
    overall_stats_skipped_as_original = Stats()
    transformed_stats = Stats()
    transformed_stats_skipped_as_original = Stats()
    overall_run_stats: List[Stats] = [Stats() for _ in range(max(0, int(num_runs)))]
    overall_run_stats_skipped_as_original: List[Stats] = [Stats() for _ in range(max(0, int(num_runs)))]
    transformed_run_stats: List[Stats] = [Stats() for _ in range(max(0, int(num_runs)))]
    transformed_run_stats_skipped_as_original: List[Stats] = [Stats() for _ in range(max(0, int(num_runs)))]
    transformed_count = 0
    transformed_op_counter = Counter()
    lr_mode_counter = Counter()
    seen: Set[str] = set()
    original_subset_path: Optional[str] = None

    for p in output_paths:
        if not os.path.exists(p):
            continue
        for obj in iter_jsonl(p):
            k = obj.get("key", None)
            if not k:
                k = stable_key_from_screenspot(obj)
            if k in seen:
                continue
            seen.add(k)
            debug = obj.get("_debug", None)
            if original_subset_path is None and isinstance(debug, dict) and debug.get("subset_path"):
                original_subset_path = str(debug.get("subset_path"))

            c = normalize_correctness(obj.get("correctness", "wrong_format"))
            overall_stats.add(c)
            overall_stats_skipped_as_original.add(correctness_with_original_for_skipped(obj, c))

            multi_corr = obj.get("multi_run_correctnesses", None)
            if isinstance(multi_corr, list) and multi_corr:
                while len(overall_run_stats) < len(multi_corr):
                    overall_run_stats.append(Stats())
                while len(overall_run_stats_skipped_as_original) < len(multi_corr):
                    overall_run_stats_skipped_as_original.append(Stats())
                for i, ci in enumerate(multi_corr):
                    ci_norm = normalize_correctness(ci)
                    overall_run_stats[i].add(ci_norm)
                    overall_run_stats_skipped_as_original[i].add(correctness_with_original_for_skipped(obj, ci_norm))
            else:
                for s in overall_run_stats:
                    s.add(c)
                for s in overall_run_stats_skipped_as_original:
                    s.add(correctness_with_original_for_skipped(obj, c))

            if is_transformed_record(obj):
                transformed_stats.add(c)
                transformed_stats_skipped_as_original.add(correctness_with_original_for_skipped(obj, c))
                transformed_count += 1

                if isinstance(multi_corr, list) and multi_corr:
                    while len(transformed_run_stats) < len(multi_corr):
                        transformed_run_stats.append(Stats())
                    while len(transformed_run_stats_skipped_as_original) < len(multi_corr):
                        transformed_run_stats_skipped_as_original.append(Stats())
                    for i, ci in enumerate(multi_corr):
                        ci_norm = normalize_correctness(ci)
                        transformed_run_stats[i].add(ci_norm)
                        transformed_run_stats_skipped_as_original[i].add(correctness_with_original_for_skipped(obj, ci_norm))
                else:
                    for s in transformed_run_stats:
                        s.add(c)
                    for s in transformed_run_stats_skipped_as_original:
                        s.add(correctness_with_original_for_skipped(obj, c))

                ops = extract_transformed_ops(obj)
                if not ops:
                    transformed_op_counter["unknown"] += 1
                else:
                    for op in ops:
                        transformed_op_counter[str(op)] += 1

            mode = obj.get("lr_mode", None)
            if mode:
                lr_mode_counter[str(mode)] += 1

    return {
        "overall": overall_stats,
        "overall_skipped_as_original": overall_stats_skipped_as_original,
        "transformed": transformed_stats,
        "transformed_skipped_as_original": transformed_stats_skipped_as_original,
        "overall_runs": overall_run_stats,
        "overall_runs_skipped_as_original": overall_run_stats_skipped_as_original,
        "transformed_runs": transformed_run_stats,
        "transformed_runs_skipped_as_original": transformed_run_stats_skipped_as_original,
        "transformed_count": int(transformed_count),
        "transformed_ops": transformed_op_counter,
        "lr_mode_counts": lr_mode_counter,
        "dedup_keys": len(seen),
        "original_subset_path": original_subset_path,
        "original_acc": get_original_accuracy_from_subset(original_subset_path),
    }


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
    merged_stats_skipped_as_original = Stats()
    transformed_stats = Stats()
    transformed_stats_skipped_as_original = Stats()
    multi_run_merged_stats: List[Stats] = []
    multi_run_merged_stats_skipped_as_original: List[Stats] = []
    multi_run_transformed_stats: List[Stats] = []
    multi_run_transformed_stats_skipped_as_original: List[Stats] = []
    transformed_count = 0
    transformed_op_counter = Counter()
    seen: Set[str] = set()
    lr_mode_counter = Counter()
    original_subset_path: Optional[str] = None

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
                debug = obj.get("_debug", None)
                if original_subset_path is None and isinstance(debug, dict) and debug.get("subset_path"):
                    original_subset_path = str(debug.get("subset_path"))

                c = normalize_correctness(obj.get("correctness", "wrong_format"))
                merged_stats.add(c)
                merged_stats_skipped_as_original.add(correctness_with_original_for_skipped(obj, c))

                multi_corr = obj.get("multi_run_correctnesses", None)
                if isinstance(multi_corr, list) and multi_corr:
                    while len(multi_run_merged_stats) < len(multi_corr):
                        multi_run_merged_stats.append(Stats())
                    while len(multi_run_merged_stats_skipped_as_original) < len(multi_corr):
                        multi_run_merged_stats_skipped_as_original.append(Stats())
                    for i, ci in enumerate(multi_corr):
                        ci_norm = normalize_correctness(ci)
                        multi_run_merged_stats[i].add(ci_norm)
                        multi_run_merged_stats_skipped_as_original[i].add(correctness_with_original_for_skipped(obj, ci_norm))

                if is_transformed_record(obj):
                    transformed_stats.add(c)
                    transformed_stats_skipped_as_original.add(correctness_with_original_for_skipped(obj, c))
                    transformed_count += 1
                    if isinstance(multi_corr, list) and multi_corr:
                        while len(multi_run_transformed_stats) < len(multi_corr):
                            multi_run_transformed_stats.append(Stats())
                        while len(multi_run_transformed_stats_skipped_as_original) < len(multi_corr):
                            multi_run_transformed_stats_skipped_as_original.append(Stats())
                        for i, ci in enumerate(multi_corr):
                            ci_norm = normalize_correctness(ci)
                            multi_run_transformed_stats[i].add(ci_norm)
                            multi_run_transformed_stats_skipped_as_original[i].add(correctness_with_original_for_skipped(obj, ci_norm))

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

    original_acc = get_original_accuracy_from_subset(original_subset_path)
    merged_summary = {
        "metrics": {
            "overall_raw_scored_only": stats_to_metrics(merged_stats),
            "overall": stats_to_metrics_including_skipped_original_acc(merged_stats, original_acc),
            "overall_including_skipped_original_acc": stats_to_metrics_including_skipped_original_acc(
                merged_stats,
                original_acc,
            ),
            "overall_skipped_as_original": stats_to_metrics(merged_stats_skipped_as_original),
            "transformed_selected": stats_to_metrics(transformed_stats),
            "transformed_selected_skipped_as_original": stats_to_metrics(transformed_stats_skipped_as_original),
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
            "per_run": [
                stats_to_metrics_including_skipped_original_acc(s, original_acc)
                for s in multi_run_merged_stats
            ],
            "acc_runs": [
                stats_to_metrics_including_skipped_original_acc(s, original_acc)["acc"]
                for s in multi_run_merged_stats
            ],
            "acc_mean": mean_and_variance([
                stats_to_metrics_including_skipped_original_acc(s, original_acc)["acc"]
                for s in multi_run_merged_stats
            ])[0],
            "acc_variance": mean_and_variance([
                stats_to_metrics_including_skipped_original_acc(s, original_acc)["acc"]
                for s in multi_run_merged_stats
            ])[1],
        }
        acc_runs_adj = [s.acc for s in multi_run_merged_stats_skipped_as_original]
        acc_mean_adj, acc_var_adj = mean_and_variance(acc_runs_adj)
        merged_summary["metrics"]["multi_run_overall_skipped_as_original"] = {
            "num_runs": len(multi_run_merged_stats_skipped_as_original),
            "per_run": [stats_to_metrics(s) for s in multi_run_merged_stats_skipped_as_original],
            "acc_runs": acc_runs_adj,
            "acc_mean": acc_mean_adj,
            "acc_variance": acc_var_adj,
        }
        acc_runs_orig_acc = [
            stats_to_metrics_including_skipped_original_acc(s, original_acc)["acc"]
            for s in multi_run_merged_stats
        ]
        acc_mean_orig_acc, acc_var_orig_acc = mean_and_variance(acc_runs_orig_acc)
        merged_summary["metrics"]["multi_run_overall_including_skipped_original_acc"] = {
            "num_runs": len(multi_run_merged_stats),
            "per_run": [
                stats_to_metrics_including_skipped_original_acc(s, original_acc)
                for s in multi_run_merged_stats
            ],
            "acc_runs": acc_runs_orig_acc,
            "acc_mean": acc_mean_orig_acc,
            "acc_variance": acc_var_orig_acc,
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
        acc_runs_t_adj = [s.acc for s in multi_run_transformed_stats_skipped_as_original]
        acc_mean_t_adj, acc_var_t_adj = mean_and_variance(acc_runs_t_adj)
        merged_summary["metrics"]["multi_run_transformed_selected_skipped_as_original"] = {
            "num_runs": len(multi_run_transformed_stats_skipped_as_original),
            "per_run": [stats_to_metrics(s) for s in multi_run_transformed_stats_skipped_as_original],
            "acc_runs": acc_runs_t_adj,
            "acc_mean": acc_mean_t_adj,
            "acc_variance": acc_var_t_adj,
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


def get_venus_processed_size(inputs: Dict[str, Any], batch_index: int = 0) -> Optional[Tuple[int, int]]:
    grid = inputs.get("image_grid_thw", None)
    if grid is None:
        return None
    try:
        row = grid[batch_index]
        input_h = int(row[1].item() * 14)
        input_w = int(row[2].item() * 14)
        return input_w, input_h
    except Exception:
        return None


# -------------------------
# Main
# -------------------------
def main():
    enabled_dist, rank, world_size, local_rank, device = dist_init()

    ap = argparse.ArgumentParser(
        description=(
            "Standalone in-region mover for inclusionAI/UI-Venus-Ground-7B. "
            "Supports region jsonl in absolute bbox format or relative bounds format "
            "and supports UI-Venus bbox outputs."
        )
    )

    # ---- model ----
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["", "flash_attention_2", "sdpa", "eager"])

    # ---- data ----
    ap.add_argument("--json_file_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/annotations")
    ap.add_argument("--base_image_dir", type=str, default="/pasteur/u/yiming/chenyue/MVP/data/screenspot-pro/images")
    ap.add_argument("--subset_path", type=str, default="")

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

    # ---- center inset (avoid landing exactly on region boundary) ----
    ap.add_argument(
        "--center_inset_px",
        type=float,
        default=10.0,
        help="Require GT bbox center to be at least this many pixels inside the target region (avoid center stuck on region boundary). Set 0 to disable.",
    )
    ap.add_argument("--center_inset_px_x", type=float, default=None)
    ap.add_argument("--center_inset_px_y", type=float, default=None)

    # ---- bbox edge margin (distance from moved bbox edges to non-blind area) ----
    ap.add_argument(
        "--bbox_edge_margin_px",
        type=float,
        default=0.0,
        help="After moving into a blind region, require every edge of the moved GT bbox to be strictly greater than this many pixels away from non-blind area. Default 100.",
    )
    ap.add_argument("--bbox_edge_margin_px_x", type=float, default=None)
    ap.add_argument("--bbox_edge_margin_px_y", type=float, default=None)

    # Merge adjacent blind-zone boxes into connected components.
    ap.add_argument("--merge_connected_region", type=str, default="False")
    ap.add_argument("--merge_touch_eps", type=float, default=2.0)
    ap.add_argument(
        "--merged_region_cache_path",
        type=str,
        default="",
        help=(
            "Path to save/load precomputed merged region components. "
            "Default: <output_path>.merged_region_components.json. "
            "The cache key includes region file path/mtime/size, W/H, "
            "merge_connected_region, and merge_touch_eps."
        ),
    )
    ap.add_argument(
        "--no_merged_region_disk_cache",
        action="store_true",
        help="Disable disk cache for merged region components; in-memory cache is still used.",
    )

    # ---- change budget ----
    ap.add_argument(
        "--max_change_pct",
        type=float,
        default=99.0,
        help="Max 'seam area' pixels percent of original (wrap shift): seam=|dx|H+|dy|W-|dx||dy| with canonical dx/dy.",
    )
    ap.add_argument("--num_crop_runs", type=int, default=5, help="Number of runs (shift plans). Default 5. (kept arg name)")

    # ---- run control ----
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--summary_path", type=str, default="")
    ap.add_argument("--language", type=str, default="en", choices=["en", "cn"])
    ap.add_argument("--inst_style", type=str, default="instruction")
    ap.add_argument("--gt_type", type=str, default="all", choices=["positive", "negative", "all"])
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of shifted runs to generate together per sample.",
    )
    ap.add_argument(
        "--coord_output_mode",
        type=str,
        default="venus_bbox",
        help=(
            "Coordinate format of model output. "
            "Use 'venus_bbox' for inclusionAI/UI-Venus-Ground-7B [x1,y1,x2,y2] outputs; "
            "Use 'pixel' for absolute-pixel models such as HelloKKMe/GTA1-7B; "
            "use 'norm1000' for Qwen-style normalized outputs; "
            "use 'rel01' for [0,1] relative outputs; "
            "use 'auto' for the legacy heuristic."
        ),
    )

    # processor image scaling
    ap.add_argument("--min_pixels", type=int, default=2000000)
    ap.add_argument("--max_pixels", type=int, default=4800000)

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

    # resolve center inset
    inset_x = args.center_inset_px if args.center_inset_px_x is None else args.center_inset_px_x
    inset_y = args.center_inset_px if args.center_inset_px_y is None else args.center_inset_px_y
    if inset_x < 0 or inset_y < 0:
        raise ValueError(f"center_inset_px must be non-negative. Got inset_x={inset_x}, inset_y={inset_y}")


    # resolve bbox edge margin
    edge_margin_x = args.bbox_edge_margin_px if args.bbox_edge_margin_px_x is None else args.bbox_edge_margin_px_x
    edge_margin_y = args.bbox_edge_margin_px if args.bbox_edge_margin_px_y is None else args.bbox_edge_margin_px_y
    if edge_margin_x < 0 or edge_margin_y < 0:
        raise ValueError(f"bbox_edge_margin_px must be non-negative. Got edge_margin_x={edge_margin_x}, edge_margin_y={edge_margin_y}")

    merge_connected = str2bool(args.merge_connected_region)

    merged_region_cache_path = str(args.merged_region_cache_path or "").strip()
    if not merged_region_cache_path:
        merged_region_cache_path = args.output_path + ".merged_region_components.json"
    use_merged_region_disk_cache = not bool(args.no_merged_region_disk_cache)

    # per-rank output/summary
    output_path_rank = f"{args.output_path}.rank{rank}.jsonl"
    summary_path_rank = f"{args.summary_path}.rank{rank}.json"
    transformed_output_path = f"{args.output_path}.transformed.jsonl"
    transformed_output_path_rank = f"{args.output_path}.transformed.rank{rank}.jsonl"
    os.makedirs(os.path.dirname(output_path_rank) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(summary_path_rank) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(transformed_output_path_rank) or ".", exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # resume default: ON if not explicitly disabled
    resume = args.resume and (not args.no_resume)
    if (not args.resume) and (not args.no_resume):
        resume = True

    rank0_print(rank, f"[Mode] ✅ SHIFT-ONLY fixed canvas: WRAP-AROUND shift (no black fill); NO crop.")
    rank0_print(rank, f"[Preset] model_path={args.model_path}")
    rank0_print(rank, f"[Preset] region_bbox_path={args.region_bbox_path}")
    rank0_print(rank, f"[Budget] max_change_pct={args.max_change_pct:.2f}% -> seam_px <= budget_px")
    rank0_print(rank, f"[MultiRun] num_runs={args.num_crop_runs} (styles: horizontal/vertical/diagonal/prefer_right/prefer_down)")
    rank0_print(rank, "[Coverage] directly move WHOLE bbox INTO selected region; regions fixed in canvas coords; shifted bbox must remain one intact box")
    rank0_print(rank, f"[CenterInset] center_inset_px_xy=({float(inset_x)},{float(inset_y)}) (legacy; bbox edge margin is stricter by default)")
    rank0_print(rank, f"[BBoxEdgeMargin] bbox_edge_margin_px_xy=({float(edge_margin_x)},{float(edge_margin_y)})")
    rank0_print(rank, f"[AspectFix] aspect_ratio_limit={args.aspect_ratio_limit} (<200), will pad if too skinny")
    rank0_print(rank, f"[Parse] coord_output_mode={args.coord_output_mode}")
    rank0_print(rank, f"[Parse] region jsonl auto mode: bbox(abs) / bounds(rel01) / bounds(abs)")
    rank0_print(rank, f"[Infer] batch_size={args.batch_size} shifted runs per generate call")
    rank0_print(rank, f"[RegionMerge] merge_connected_region={merge_connected}, merge_touch_eps={float(args.merge_touch_eps)}")
    if use_merged_region_disk_cache:
        rank0_print(rank, f"[RegionMergeCache] enabled path={merged_region_cache_path}")
    else:
        rank0_print(rank, "[RegionMergeCache] disabled by --no_merged_region_disk_cache")

    # ----------------------------
    # Load model/processor
    # ----------------------------
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", None)
    if mt is None:
        raise RuntimeError("Missing model_type in config. Please check --model_path.")

    ModelCls = pick_model_cls(str(mt))

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
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

    component_disk_cache: Dict[str, Any] = {"version": 1, "entries": {}}
    if use_merged_region_disk_cache:
        component_disk_cache = load_region_component_cache_file(merged_region_cache_path)
        rank0_print(
            rank,
            f"[RegionMergeCache] loaded entries={len(component_disk_cache.get('entries', {}))} from {merged_region_cache_path}",
        )

    union_cache: Dict[Tuple[int, int, str], Dict[str, Any]] = {}

    def get_union_for_original_image(
        W: int,
        H: int,
    ) -> Tuple[
        BBoxUnion,
        List[Tuple[float, float, float, float]],
        List[Tuple[float, float, float, float]],
        List[RegionComponent],
    ]:
        """
        返回：
          - u0: region boxes（proc）用于判定/搜索
          - region_bboxes_abs_orig: 原始 region boxes（用于 region_orig 可视化）
          - region_boxes_proc: raw region boxes（用于 model-input 可视化）
          - region_components: 已合并好的 connected components（用于 center_inset / bbox_edge_margin 判断）

        重要：region component merge 只依赖 region 文件、W/H 和 merge 参数，
        所以这里会先查内存 cache，再查磁盘 cache。第一次合并后会保存到
        --merged_region_cache_path，后续 sample / 后续重新运行都直接复用。
        """
        disk_key = build_region_component_cache_key(
            region_bbox_path=args.region_bbox_path,
            W=int(W),
            H=int(H),
            merge_connected_region=bool(merge_connected),
            merge_touch_eps=float(args.merge_touch_eps),
        )
        mem_key = (int(W), int(H), disk_key)
        if mem_key in union_cache:
            c = union_cache[mem_key]
            return c["u0"], c["region_bboxes_abs_orig"], c["region_boxes_proc"], c["region_components"]

        entries = component_disk_cache.get("entries", {}) if isinstance(component_disk_cache, dict) else {}
        cached_entry = entries.get(disk_key, None) if use_merged_region_disk_cache else None

        if isinstance(cached_entry, dict):
            try:
                region_bboxes_abs_orig = [
                    tuple(map(float, b))
                    for b in cached_entry.get("region_bboxes_abs_orig", [])
                    if isinstance(b, list) and len(b) == 4
                ]
                region_boxes_proc = [
                    tuple(map(float, b))
                    for b in cached_entry.get("region_boxes_proc", [])
                    if isinstance(b, list) and len(b) == 4
                ]
                region_components = components_from_json_list(cached_entry.get("region_components", []))
                if region_bboxes_abs_orig and region_boxes_proc and region_components:
                    # Recreate u0 cheaply for compatibility with older code paths.
                    u0 = BBoxUnion(
                        region_bboxes_abs_orig,
                        shift_dx=0.0,
                        shift_dy=0.0,
                        drop_collapsed=True,
                    )
                    union_cache[mem_key] = {
                        "u0": u0,
                        "region_bboxes_abs_orig": region_bboxes_abs_orig,
                        "region_boxes_proc": region_boxes_proc,
                        "region_components": region_components,
                        "cache_source": "disk",
                    }
                    if rank == 0 and len(union_cache) <= 3:
                        rank0_print(
                            rank,
                            f"[RegionMergeCache] HIT W,H=({W},{H}) components={len(region_components)} raw_boxes={len(region_boxes_proc)}",
                        )
                    return u0, region_bboxes_abs_orig, region_boxes_proc, region_components
            except Exception:
                # Bad entry: fall through and rebuild.
                pass

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
        region_components = make_region_components(
            region_boxes_proc,
            merge_connected=bool(merge_connected),
            touch_eps=float(args.merge_touch_eps),
        )

        union_cache[mem_key] = {
            "u0": u0,
            "region_bboxes_abs_orig": region_bboxes_abs_orig,
            "region_boxes_proc": region_boxes_proc,
            "region_components": region_components,
            "cache_source": "computed",
        }

        if use_merged_region_disk_cache:
            entries = component_disk_cache.setdefault("entries", {})
            entries[disk_key] = {
                "cache_key": disk_key,
                "W": int(W),
                "H": int(H),
                "region_bbox_path": os.path.abspath(str(args.region_bbox_path)),
                "merge_connected_region": bool(merge_connected),
                "merge_touch_eps": float(args.merge_touch_eps),
                "region_bboxes_abs_orig": [_box_to_json_list(b) for b in region_bboxes_abs_orig],
                "region_boxes_proc": [_box_to_json_list(b) for b in region_boxes_proc],
                "region_components": [region_component_to_json(c) for c in region_components],
            }
            save_region_component_cache_file(merged_region_cache_path, component_disk_cache)
            if rank == 0 and len(union_cache) <= 3:
                rank0_print(
                    rank,
                    f"[RegionMergeCache] MISS -> saved W,H=({W},{H}) components={len(region_components)} raw_boxes={len(region_boxes_proc)} to {merged_region_cache_path}",
                )

        return u0, region_bboxes_abs_orig, region_boxes_proc, region_components

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

            u0, region_bboxes_abs_orig, region_boxes_proc, region_components = get_union_for_original_image(W, H)

            budget_px = compute_change_budget_pixels(W, H, float(args.max_change_pct))

            lr_crop_search: Dict[str, Any] = {
                "_mode": "shift_only_fixed_canvas_wrap",
                "_budget": {"budget_px": int(budget_px), "max_change_pct": float(args.max_change_pct)},
            }
            selected_run_plans: List[Dict[str, Any]] = []

            if sample_gt_type == "negative":
                selected_run_plans = [
                    {"run": i + 1, "skipped": False, "style": "none", "region_id": None, "dx": 0, "dy": 0, "filled_px": 0, "filled_ratio": 0.0}
                    for i in range(num_runs)
                ]
                lr_crop_search["_debug"] = {"reason": "negative_sample_no_shift"}
            else:
                assert gt_xyxy0 is not None
                plans, dbg = build_multi_run_shift_plans(
                    gt_xyxy=gt_xyxy0,
                    region_boxes_proc=region_boxes_proc,
                    W=W,
                    H=H,
                    budget_px=budget_px,
                    inset_x=float(inset_x),
                    inset_y=float(inset_y),
                    edge_margin_x=float(edge_margin_x),
                    edge_margin_y=float(edge_margin_y),
                    num_runs=num_runs,
                    u0=u0,
                    merge_connected_region=bool(merge_connected),
                    merge_touch_eps=float(args.merge_touch_eps),
                    region_components=region_components,
                )
                selected_run_plans = plans
                lr_crop_search["_search_debug"] = dbg

            any_non_skipped = any((not p.get("skipped", True)) for p in selected_run_plans)
            any_nonzero_shift = any(
                (not p.get("skipped", True)) and (int(p.get("dx", 0)) != 0 or int(p.get("dy", 0)) != 0) for p in selected_run_plans
            )

            if not any_non_skipped:
                lr_mode = "skipped_no_feasible_shift"
            elif not any_nonzero_shift:
                lr_mode = "none_or_zero_shift_only"
            else:
                lr_mode = "shift_selected"
            lr_mode_counter[lr_mode] += 1

            # --- eval one plan ---
            def _prepare_one_shift(
                plan: Dict[str, Any],
                run_tag: str,
            ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
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
                    return trial, None

                dx = int(plan.get("dx", 0))
                dy = int(plan.get("dy", 0))
                dx = _canonical_shift_delta(dx, W)
                dy = _canonical_shift_delta(dy, H)
                filled = int(plan.get("filled_px", filled_pixels_for_shift(W, H, dx, dy)))

                trial["transform"] = {"kind": "shift", "mode": "wrap", "dx": int(dx), "dy": int(dy)}
                trial["shift_dx"] = int(dx)
                trial["shift_dy"] = int(dy)
                trial["filled_px"] = int(filled)
                trial["budget_px"] = int(budget_px)
                trial["target_region_id"] = plan.get("region_id", None)
                trial["target_region_xyxy_proc"] = plan.get("region_xyxy_proc", None)

                img_shift = shift_image_on_canvas(img_rgb, dx=dx, dy=dy, fill=(0, 0, 0))
                trial["img_size_transformed"] = [int(W), int(H)]

                gt_box_after: Optional[Tuple[float, float, float, float]] = None
                if (sample_gt_type != "negative" and gt_xyxy0 is not None):
                    gt_box_after = bbox_after_shift_wrap_single(gt_xyxy0, dx=dx, dy=dy, W=W, H=H)

                if gt_box_after is None:
                    trial["bbox_transformed"] = None
                    trial["bbox_transformed_parts"] = None
                else:
                    trial["bbox_transformed"] = [
                        float(gt_box_after[0]),
                        float(gt_box_after[1]),
                        float(gt_box_after[2]),
                        float(gt_box_after[3]),
                    ]
                    trial["bbox_transformed_parts"] = None

                img_for_model, pad4 = pad_to_aspect_limit(img_shift, max_ratio=float(args.aspect_ratio_limit), fill=(0, 0, 0))
                pad_left, pad_top, pad_right, pad_bottom = pad4
                Wp, Hp = img_for_model.size
                trial["pad"] = {"left": int(pad_left), "top": int(pad_top), "right": int(pad_right), "bottom": int(pad_bottom)}
                trial["img_size_padded"] = [int(Wp), int(Hp)]

                region_boxes_for_model = []
                for (rx1, ry1, rx2, ry2) in region_boxes_proc:
                    region_boxes_for_model.append((rx1 + pad_left, ry1 + pad_top, rx2 + pad_left, ry2 + pad_top))

                user_prompt = (
                    SYSTEM_PROMPT.format(height=Hp, width=Wp).strip()
                    + "\n\nElement description: "
                    + instruction
                )

                request = {
                    "trial": trial,
                    "run_tag": str(run_tag),
                    "img_for_model": img_for_model,
                    "user_prompt": user_prompt,
                    "Wp": int(Wp),
                    "Hp": int(Hp),
                    "pad_left": int(pad_left),
                    "pad_top": int(pad_top),
                    "dx": int(dx),
                    "dy": int(dy),
                    "gt_box_after": gt_box_after,
                    "region_boxes_for_model": region_boxes_for_model,
                }
                return trial, request

            def _mark_infer_failed(req: Dict[str, Any], error: str) -> None:
                req["trial"].update({
                    "raw_response": "",
                    "pred_pixel": None,
                    "pred_pixel_padded": None,
                    "correctness": "wrong_format",
                    "_debug": {"reason": "infer_failed", "error": str(error)},
                })

            def _finalize_one_shift(req: Dict[str, Any], raw: str, prompt_len: int) -> None:
                trial = req["trial"]
                Wp = int(req["Wp"])
                Hp = int(req["Hp"])
                pad_left = int(req["pad_left"])
                pad_top = int(req["pad_top"])
                dx = int(req["dx"])
                dy = int(req["dy"])
                gt_box_after = req["gt_box_after"]
                region_boxes_for_model = req["region_boxes_for_model"]
                img_for_model = req["img_for_model"]

                trial["raw_response"] = raw

                xpad, ypad, parsed, coord_dbg = extract_coordinates(
                    raw,
                    W=Wp,
                    H=Hp,
                    coord_output_mode=args.coord_output_mode,
                    source_size=req.get("venus_processed_size", None),
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
                    if (not parsed) or (gt_box_after is None):
                        correctness = "wrong_format"
                    else:
                        correctness = "wrong" if (not in_bounds_unpad) else ("correct" if point_in_box(float(x0), float(y0), gt_box_after) else "wrong")
                correctness = normalize_correctness(correctness)
                trial["correctness"] = correctness

                cx0, cy0 = bbox_center_xy(gt_xyxy0)
                if gt_box_after is not None:
                    cx2, cy2 = bbox_center_xy(gt_box_after)
                else:
                    cx2 = (float(cx0) + float(dx)) % float(W)
                    cy2 = (float(cy0) + float(dy)) % float(H)
                center_in_region_after = point_in_any_region_with_inset(
                    cx2,
                    cy2,
                    region_boxes_proc,
                    inset_x=float(inset_x),
                    inset_y=float(inset_y),
                )
                trial["_debug"] = {
                    "parsed": bool(parsed),
                    "in_bounds_unpad": bool(in_bounds_unpad),
                    "prompt_len": int(prompt_len),
                    "coord_parse": coord_dbg,
                    "center_in_region_after_shift": center_in_region_after,
                    "center_xy_before": [float(cx0), float(cy0)],
                    "center_xy_after_shift": [float(cx2), float(cy2)],
                    "center_inset_xy": [float(inset_x), float(inset_y)],
                    "region_vis_count_model": int(len(region_boxes_for_model)),
                    "region_proc_count": int(len(region_boxes_proc)),
                    "shift_mode": "wrap",
                    "bbox_integrity_required": True,
                    "bbox_split_after_shift": bool(
                        (sample_gt_type != "negative") and (gt_xyxy0 is not None) and (gt_box_after is None)
                    ),
                }

                if args.vis_dir and args.vis_every and (src_idx % args.vis_every == 0):
                    vis_path = os.path.join(args.vis_dir, f"idx_{src_idx:06d}.shift.{req['run_tag']}.rank{rank}.png")

                    bbox_for_vis_model = None
                    if gt_box_after is not None:
                        bbox_for_vis_model = (
                            float(gt_box_after[0]) + float(pad_left),
                            float(gt_box_after[1]) + float(pad_top),
                            float(gt_box_after[2]) + float(pad_left),
                            float(gt_box_after[3]) + float(pad_top),
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

                    vis_path_region = os.path.join(args.vis_dir, f"idx_{src_idx:06d}.shift.{req['run_tag']}.rank{rank}.region_orig.png")
                    visualize_prediction_pixel(
                        img_rgb.copy(),
                        0,
                        0,
                        vis_path_region,
                        bbox_xyxy=gt_xyxy0 if gt_xyxy0 is not None else None,
                        region_boxes_xyxy=region_bboxes_abs_orig,
                        parsed=False,
                    )

            def _eval_one_shift(plan: Dict[str, Any], run_tag: str) -> Dict[str, Any]:
                trial, req = _prepare_one_shift(plan, run_tag=run_tag)
                if req is None:
                    return trial

                img_for_model = req["img_for_model"]
                user_prompt = str(req["user_prompt"])

                try:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_for_model},
                            {"type": "text", "text": user_prompt},
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
                            {"type": "image", "image": img_for_model},
                            {"type": "text", "text": user_prompt},
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
                req["venus_processed_size"] = get_venus_processed_size(inputs, 0)

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
                    _finalize_one_shift(req, raw, prompt_len)
                except Exception as e:
                    _mark_infer_failed(req, str(e))

                return trial

            def _infer_shift_requests(requests: List[Dict[str, Any]]) -> None:
                batch_size = int(args.batch_size)

                def _infer_one(req: Dict[str, Any]) -> None:
                    img_for_model = req["img_for_model"]
                    user_prompt = str(req["user_prompt"])
                    try:
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_for_model},
                                {"type": "text", "text": user_prompt},
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
                                {"type": "image", "image": img_for_model},
                                {"type": "text", "text": user_prompt},
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
                    req["venus_processed_size"] = get_venus_processed_size(inputs, 0)

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
                        _finalize_one_shift(req, raw, prompt_len)
                    except Exception as e:
                        _mark_infer_failed(req, str(e))

                for start in range(0, len(requests), batch_size):
                    chunk = requests[start:start + batch_size]
                    try:
                        texts: List[str] = []
                        images: List[Any] = []
                        for req in chunk:
                            img_for_model = req["img_for_model"]
                            user_prompt = str(req["user_prompt"])
                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": img_for_model},
                                    {"type": "text", "text": user_prompt},
                                ],
                            }]
                            texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                            images.append(img_for_model)

                        inputs = processor(
                            text=texts,
                            images=images,
                            return_tensors="pt",
                            padding=True,
                        )
                        inputs.pop("token_type_ids", None)
                        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
                        for i, req in enumerate(chunk):
                            req["venus_processed_size"] = get_venus_processed_size(inputs, i)

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
                        if len(out_texts) != len(chunk):
                            raise RuntimeError(f"batch_decode returned {len(out_texts)} texts for batch size {len(chunk)}")
                        for i, (req, raw) in enumerate(zip(chunk, out_texts)):
                            _finalize_one_shift(req, raw, prompt_lens[i])
                    except Exception:
                        for req in chunk:
                            _infer_one(req)

            # ----------------------------
            # Run / summarize
            # ----------------------------
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
                    f".frac{plan.get('applied_fraction', '')}"
                    f".style{plan.get('style', 'shift')}"
                )
                run_items.append((ri, plan, run_tag))

            trials_by_ri: Dict[int, Dict[str, Any]] = {}
            if args.batch_size <= 1:
                for ri, plan, run_tag in run_items:
                    trials_by_ri[ri] = _eval_one_shift(plan, run_tag=run_tag)
            else:
                requests_to_infer: List[Dict[str, Any]] = []
                for ri, plan, run_tag in run_items:
                    trial, req = _prepare_one_shift(plan, run_tag=run_tag)
                    trials_by_ri[ri] = trial
                    if req is not None:
                        requests_to_infer.append(req)
                _infer_shift_requests(requests_to_infer)

            for ri, plan, run_tag in run_items:
                trial = trials_by_ri[ri]

                run_key = f"run_{ri + 1}"
                multi_run_trials[run_key] = {"shift": trial}

                multi_run_choices.append({
                    "run": int(ri + 1),
                    "style": str(plan.get("style", "shift")),
                    "dx": int(plan.get("dx", 0)),
                    "dy": int(plan.get("dy", 0)),
                    "region_id": plan.get("region_id", None),
                    "skipped": bool(plan.get("skipped", False)),
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
                "original_correctness": normalize_correctness(sample.get("correctness", "wrong_format")),
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
                    "center_inset_px_xy": [float(inset_x), float(inset_y)],
                    "bbox_edge_margin_px_xy": [float(edge_margin_x), float(edge_margin_y)],
                    "movement_rule": "shift-only fixed canvas; WRAP shift; choose per-run different blind region; directly move WHOLE bbox INTO region under seam-area budget; moved bbox edges must be > bbox_edge_margin_px away from non-blind area using original target-region boundary; reject any shift that would split bbox across wrap boundary",
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
    rank_file_metrics = summarize_output_paths([output_path_rank], num_runs=num_runs)
    stats_overall_summary: Stats = rank_file_metrics["overall"]
    stats_overall_skipped_as_original_summary: Stats = rank_file_metrics["overall_skipped_as_original"]
    stats_transformed_selected_summary: Stats = rank_file_metrics["transformed"]
    stats_transformed_selected_skipped_as_original_summary: Stats = rank_file_metrics["transformed_skipped_as_original"]
    stats_overall_runs_summary: List[Stats] = rank_file_metrics["overall_runs"]
    stats_overall_runs_skipped_as_original_summary: List[Stats] = rank_file_metrics["overall_runs_skipped_as_original"]
    stats_transformed_selected_runs_summary: List[Stats] = rank_file_metrics["transformed_runs"]
    stats_transformed_selected_runs_skipped_as_original_summary: List[Stats] = rank_file_metrics["transformed_runs_skipped_as_original"]
    transformed_count_summary = int(rank_file_metrics["transformed_count"])
    transformed_op_counter_summary: Counter = rank_file_metrics["transformed_ops"]
    lr_mode_counter_summary: Counter = rank_file_metrics["lr_mode_counts"]
    original_acc_summary = rank_file_metrics.get("original_acc", None)

    overall_acc_runs = [s.acc for s in stats_overall_runs_summary]
    overall_acc_mean, overall_acc_var = mean_and_variance(overall_acc_runs)
    overall_acc_runs_including_skipped_original_acc = [
        stats_to_metrics_including_skipped_original_acc(s, original_acc_summary)["acc"]
        for s in stats_overall_runs_summary
    ]
    overall_acc_mean_including_skipped_original_acc, overall_acc_var_including_skipped_original_acc = mean_and_variance(
        overall_acc_runs_including_skipped_original_acc
    )
    overall_acc_runs_skipped_as_original = [s.acc for s in stats_overall_runs_skipped_as_original_summary]
    overall_acc_mean_skipped_as_original, overall_acc_var_skipped_as_original = mean_and_variance(overall_acc_runs_skipped_as_original)
    transformed_acc_runs = [s.acc for s in stats_transformed_selected_runs_summary]
    transformed_acc_mean, transformed_acc_var = mean_and_variance(transformed_acc_runs)
    transformed_acc_runs_skipped_as_original = [s.acc for s in stats_transformed_selected_runs_skipped_as_original_summary]
    transformed_acc_mean_skipped_as_original, transformed_acc_var_skipped_as_original = mean_and_variance(transformed_acc_runs_skipped_as_original)

    summary_rank = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "metrics": {
            "overall_raw_scored_only": stats_to_metrics(stats_overall_summary),
            "overall": stats_to_metrics_including_skipped_original_acc(
                stats_overall_summary,
                original_acc_summary,
            ),
            "overall_including_skipped_original_acc": stats_to_metrics_including_skipped_original_acc(
                stats_overall_summary,
                original_acc_summary,
            ),
            "overall_skipped_as_original": stats_to_metrics(stats_overall_skipped_as_original_summary),
            "transformed_selected": stats_to_metrics(stats_transformed_selected_summary),
            "transformed_selected_skipped_as_original": stats_to_metrics(stats_transformed_selected_skipped_as_original_summary),
            "transformed_selected_count": int(transformed_count_summary),
            "transformed_selected_baseline_before_transform": perfect_correct_metrics(transformed_count_summary),
            "transformed_selected_ops": dict(transformed_op_counter_summary),
            "lr_mode_counts": dict(lr_mode_counter_summary),
            "dedup_keys": int(rank_file_metrics["dedup_keys"]),
            "new_records_written_this_run": int(n_written),
            "new_transformed_records_written_this_run": int(n_transformed_written),
            "multi_run_overall": {
                "num_runs": int(num_runs),
                "per_run": [
                    stats_to_metrics_including_skipped_original_acc(s, original_acc_summary)
                    for s in stats_overall_runs_summary
                ],
                "acc_runs": overall_acc_runs_including_skipped_original_acc,
                "acc_mean": overall_acc_mean_including_skipped_original_acc,
                "acc_variance": overall_acc_var_including_skipped_original_acc,
            },
            "multi_run_overall_including_skipped_original_acc": {
                "num_runs": int(num_runs),
                "per_run": [
                    stats_to_metrics_including_skipped_original_acc(s, original_acc_summary)
                    for s in stats_overall_runs_summary
                ],
                "acc_runs": overall_acc_runs_including_skipped_original_acc,
                "acc_mean": overall_acc_mean_including_skipped_original_acc,
                "acc_variance": overall_acc_var_including_skipped_original_acc,
            },
            "multi_run_overall_skipped_as_original": {
                "num_runs": int(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_overall_runs_skipped_as_original_summary],
                "acc_runs": overall_acc_runs_skipped_as_original,
                "acc_mean": overall_acc_mean_skipped_as_original,
                "acc_variance": overall_acc_var_skipped_as_original,
            },
            "multi_run_transformed_selected": {
                "num_runs": int(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_transformed_selected_runs_summary],
                "acc_runs": transformed_acc_runs,
                "acc_mean": transformed_acc_mean,
                "acc_variance": transformed_acc_var,
            },
            "multi_run_transformed_selected_skipped_as_original": {
                "num_runs": int(num_runs),
                "per_run": [stats_to_metrics(s) for s in stats_transformed_selected_runs_skipped_as_original_summary],
                "acc_runs": transformed_acc_runs_skipped_as_original,
                "acc_mean": transformed_acc_mean_skipped_as_original,
                "acc_variance": transformed_acc_var_skipped_as_original,
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
            "center_inset_px_xy": [float(inset_x), float(inset_y)],
            "bbox_edge_margin_px_xy": [float(edge_margin_x), float(edge_margin_y)],
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

    rank0_print(rank, f"[Rank {rank}] Done. acc={stats_overall_summary.acc:.6f} (scored={stats_overall_summary.scored}, skipped={stats_overall_summary.skipped})")
    rank0_print(
        rank,
        f"[Rank {rank}] transformed_selected_acc={stats_transformed_selected_summary.acc:.6f} "
        f"(count={transformed_count_summary}, scored={stats_transformed_selected_summary.scored}, skipped={stats_transformed_selected_summary.skipped})",
    )
    rank0_print(rank, f"[Rank {rank}] multi_run_overall_accs={overall_acc_runs} mean={overall_acc_mean:.6f} var={overall_acc_var:.6f}")
    rank0_print(rank, f"[Rank {rank}] multi_run_transformed_accs={transformed_acc_runs} mean={transformed_acc_mean:.6f} var={overall_acc_var:.6f}")
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
