# soarm101-lerobot-sagemaker-il

Run LeRobot ACT imitation learning for SO-ARM101 on Amazon SageMaker Training Jobs (Managed Spot).

This repository contains the AWS CDK stack and SageMaker entrypoint scripts for training the `duck_pickup_v1` dataset on `ml.g4dn.xlarge` (T4 16GB) with Managed Spot Training.

## Prerequisites

- AWS account / credentials configured for `ap-northeast-1`
- Node.js + `pnpm` (npm/npx is not used in this repo)
- Python 3.12 (for `submit.py`)
- The dataset `hirauchi/duck_pickup_v1` available locally under `~/.cache/huggingface/lerobot/hirauchi/duck_pickup_v1/`

## Setup

```bash
git clone https://github.com/furuya02/soarm101-lerobot-sagemaker-il.git
cd soarm101-lerobot-sagemaker-il

# CDK
cd cdk
pnpm install
pnpm exec cdk bootstrap
pnpm exec cdk deploy

# Optional: override bucket suffix
# pnpm exec cdk deploy -c bucket_suffix=20260511
```

Outputs:
- `BucketName`: `soarm101-lerobot-sagemaker-il-<account-id>`
- `SageMakerRoleArn`: `arn:aws:iam::<account-id>:role/soarm101-lerobot-sagemaker-il-sagemaker-execution-role`

## Upload the dataset

```bash
aws s3 sync \
    ~/.cache/huggingface/lerobot/hirauchi/duck_pickup_v1/ \
    s3://soarm101-lerobot-sagemaker-il-<account-id>/datasets/duck_pickup_v1/
```

## Submit a training job (Managed Spot)

```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/soarm101-lerobot-sagemaker-il-sagemaker-execution-role
export S3_BUCKET=soarm101-lerobot-sagemaker-il-<account-id>
export USE_SPOT=true

pip install sagemaker
python src/submit.py
```

Training runs ACT for 30,000 steps with `save_freq=5,000`, `batch_size=8`, `video_backend=pyav`, and resumes from checkpoint on Spot interruption.

## Download the trained model

```bash
aws s3 cp \
    s3://soarm101-lerobot-sagemaker-il-<account-id>/output/<job-name>/output/model.tar.gz \
    ./model.tar.gz
mkdir -p outputs/sagemaker_model
tar xzf model.tar.gz -C outputs/sagemaker_model/
```

## Cost note

Estimated cost is around USD 0.30 (about JPY 50) for one 30,000-step Spot run on `ml.g4dn.xlarge`. Make sure to delete the S3 bucket after the project is over, or rely on the lifecycle rule that expires `checkpoints/` after 7 days.

## License

MIT
