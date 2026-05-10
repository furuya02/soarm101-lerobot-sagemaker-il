"""SageMaker Training Job entrypoint for LeRobot ACT training (SO-ARM101)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    data_dir: str = os.environ["SM_CHANNEL_TRAIN"]
    ckpt_dir: str = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    model_dir: str = os.environ["SM_MODEL_DIR"]

    hf_root: Path = Path.home() / ".cache/huggingface/lerobot/hirauchi"
    hf_root.mkdir(parents=True, exist_ok=True)
    link_path: Path = hf_root / "duck_pickup_v1"
    if not link_path.exists():
        link_path.symlink_to(data_dir)

    cmd: list[str] = [
        "lerobot-train",
        "--dataset.repo_id=hirauchi/duck_pickup_v1",
        "--dataset.video_backend=pyav",
        "--policy.type=act",
        "--policy.device=cuda",
        f"--output_dir={ckpt_dir}",
        "--steps=30000",
        "--save_freq=5000",
        "--batch_size=8",
        "--wandb.enable=false",
        "--policy.push_to_hub=false",
        "--resume=true",
    ]
    subprocess.check_call(cmd)

    src: Path = Path(ckpt_dir) / "checkpoints" / "last" / "pretrained_model"
    dst: Path = Path(model_dir) / "pretrained_model"
    shutil.copytree(src, dst, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
