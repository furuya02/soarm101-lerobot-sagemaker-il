"""SageMaker Training Job entrypoint for LeRobot ACT training (SO-ARM101)."""

import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    data_dir = os.environ["SM_CHANNEL_TRAIN"]
    model_dir = os.environ["SM_MODEL_DIR"]
    ckpt_dir = "/opt/ml/checkpoints/lerobot"

    hf_root = Path.home() / ".cache/huggingface/lerobot/hirauchi"
    hf_root.mkdir(parents=True, exist_ok=True)
    link_path = hf_root / "duck_pickup_v1"
    if not link_path.exists():
        link_path.symlink_to(data_dir)

    config_path = Path(ckpt_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    resume_args = [f"--config_path={config_path}", "--resume=true"] if config_path.exists() else []

    subprocess.check_call([
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
        *resume_args,
    ])

    src = Path(ckpt_dir) / "checkpoints" / "last" / "pretrained_model"
    dst = Path(model_dir) / "pretrained_model"
    shutil.copytree(src, dst, dirs_exist_ok=True)


if __name__ == "__main__":
    main()
