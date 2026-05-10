"""Submit SageMaker Training Job for LeRobot ACT (SO-ARM101) on Managed Spot."""

from __future__ import annotations

import os
import time

from sagemaker.pytorch import PyTorch


def main() -> None:
    role: str = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket: str = os.environ["S3_BUCKET"]
    use_spot: bool = os.environ.get("USE_SPOT", "true").lower() in ("1", "true", "yes")

    job_name: str = f"soarm101-il-{int(time.time())}"
    max_run_sec: int = int(os.environ.get("MAX_RUN_SECONDS", str(4 * 3600)))
    max_wait_sec: int = int(os.environ.get("MAX_WAIT_SECONDS", str(8 * 3600)))

    kwargs: dict = dict(
        entry_point="train_sagemaker.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        role=role,
        framework_version="2.5",
        py_version="py311",
        instance_type=os.environ.get("INSTANCE_TYPE", "ml.g4dn.xlarge"),
        instance_count=1,
        output_path=f"s3://{bucket}/output/",
        max_run=max_run_sec,
        metric_definitions=[
            {"Name": "train_loss", "Regex": r"loss:\s+([\-0-9.]+)"},
        ],
    )
    if use_spot:
        kwargs.update(
            use_spot_instances=True,
            max_wait=max_wait_sec,
            checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/",
            checkpoint_local_path="/opt/ml/checkpoints",
        )

    estimator: PyTorch = PyTorch(**kwargs)
    estimator.fit(
        inputs={"train": f"s3://{bucket}/datasets/duck_pickup_v1/"},
        job_name=job_name,
        wait=False,
    )
    print(f"Submitted: {job_name} (spot={use_spot})")


if __name__ == "__main__":
    main()
