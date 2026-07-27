#!/usr/bin/env python3
from pathlib import Path
import os
import shutil

SRC_ROOT = Path("/data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params")
DST_ROOT = Path("/data/wenhesun/checkpoints/grpo_dapo_math_17k/qwen3_8b_custom_reward_paper_params_hf")

DST_ROOT.mkdir(parents=True, exist_ok=True)

for step_dir in sorted(SRC_ROOT.glob("global_step_*")):
    hf_src = step_dir / "actor" / "huggingface"
    if not hf_src.is_dir():
        print("[WARN] skip", step_dir, "(no actor/huggingface)")
        continue

    # AlphaRL 期望的目录：DST_ROOT/global_step_xxx
    dst = DST_ROOT / step_dir.name
    if not dst.exists():
        # 用软链接最省空间；不想用软链接也可以 shutil.copytree
        os.symlink(hf_src, dst, target_is_directory=True)

    # 在 huggingface 目录内准备 pytorch_model.bin
    bin_path = hf_src / "pytorch_model.bin"
    if not bin_path.exists():
        src_bin = hf_src / "model_world_size_8_rank_0.pt"
        if not src_bin.exists():
            print("[WARN] no model_world_size_8_rank_0.pt in", hf_src)
            continue
        # 软链接或者拷贝都行：
        os.symlink(src_bin, bin_path)
        # 如果不想用链接：
        # shutil.copy2(src_bin, bin_path)

    print("[OK] prepared", dst)

print("Done.")