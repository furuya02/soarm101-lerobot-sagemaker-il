"""Submit SageMaker Training Job for LeRobot ACT (SO-ARM101) on Managed Spot."""

import os
import time

from sagemaker.pytorch import PyTorch


def main() -> None:
    role = os.environ["SAGEMAKER_ROLE_ARN"]
    bucket = os.environ["S3_BUCKET"]
    job_name = f"soarm101-il-{int(time.time())}"

    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=os.path.dirname(os.path.abspath(__file__)),
        role=role,
        framework_version="2.8",
        py_version="py312",
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        output_path=f"s3://{bucket}/output/",
        max_run=2 * 3600,
        max_wait=3 * 3600,
        use_spot_instances=True,
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/{job_name}/",
        checkpoint_local_path="/opt/ml/checkpoints",
        metric_definitions=[{"Name": "train_loss", "Regex": r"loss:\s+([\-0-9.]+)"}],
    )
    estimator.fit(
        inputs={"train": f"s3://{bucket}/datasets/duck_pickup_v1/"},
        job_name=job_name,
        wait=False,
    )
    print(f"Submitted: {job_name}")


if __name__ == "__main__":
    main()
