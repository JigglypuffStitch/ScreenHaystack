"""Shared helpers; model-specific behavior stays in each evaluator."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist


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


def cleanup_dist(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def _get_pad_token_id(processor: Any) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return 0
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return 0


def compute_icon_placement(
    r: int,
    c: int,
    grid_w: float,
    grid_h: float,
    icon_w: int,
    icon_h: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int, int, int]:
    px = int(c * grid_w + (grid_w - icon_w) / 2)
    py = int(r * grid_h + (grid_h - icon_h) / 2)
    px = max(0, min(px, width - icon_w))
    py = max(0, min(py, height - icon_h))
    x_min, x_max = px, px + icon_w
    y_min, y_max = py, py + icon_h
    return px, py, x_min, x_max, y_min, y_max


def record_to_accuracy(record: Dict[str, Any]) -> float:
    try:
        return float(int(record.get("hit", 0)))
    except (TypeError, ValueError):
        return 0.0


def load_existing_jsonl_progress(
    *,
    jsonl_path: str,
    rows: int,
    cols: int,
    bg_name: str,
    bg_index: int,
    icon_name: str,
    clean_malformed: bool = True,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    records_by_cell: Dict[Tuple[int, int], Dict[str, Any]] = {}
    valid_records: List[Dict[str, Any]] = []
    malformed_lines = 0

    if not os.path.exists(jsonl_path):
        return records_by_cell

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                print(f"⚠️ 忽略损坏的 JSONL 行: {jsonl_path}:{line_no}")
                continue

            valid_records.append(record)
            record_bg = record.get("background") or record.get("background_file")
            if (
                record_bg == bg_name
                and record.get("background_index") == bg_index
                and record.get("icon_name") == icon_name
            ):
                r = record.get("grid_row", record.get("row"))
                c = record.get("grid_col", record.get("col"))
                if isinstance(r, int) and isinstance(c, int) and 0 <= r < rows and 0 <= c < cols:
                    records_by_cell[(r, c)] = record

    if malformed_lines and clean_malformed:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in valid_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"🧹 已清理 {malformed_lines} 行损坏 JSONL，保留 {len(valid_records)} 行有效记录")
    elif malformed_lines:
        print(f"⚠️ 忽略 {malformed_lines} 行损坏 JSONL: {jsonl_path}")

    return records_by_cell


def ensure_jsonl_append_boundary(jsonl_path: str) -> None:
    if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) == 0:
        return

    with open(jsonl_path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last_byte = f.read(1)
    if last_byte != b"\n":
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write("\n")
