from __future__ import annotations

import torch

import move_in_region_rel as base


DEFAULT_MODEL_PATH = "inclusionAI/UI-Venus-Ground-7B"


def main() -> None:
    base.DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATH
    base.main()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
