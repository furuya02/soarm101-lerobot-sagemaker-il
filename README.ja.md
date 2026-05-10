# soarm101-lerobot-sagemaker-il

SO-ARM101 向け LeRobot ACT 模倣学習を Amazon SageMaker Training Job (Managed Spot) で実行するための CDK・SageMaker エントリポイント一式です。

学習対象は `duck_pickup_v1` データセット、インスタンスは `ml.g4dn.xlarge`（T4 16GB）/ Managed Spot Training を想定しています。

## 前提

- `ap-northeast-1` 用の AWS クレデンシャル
- Node.js + `pnpm`（本リポジトリでは npm/npx は使用しません）
- Python 3.12（`submit.py` 用）
- ローカルに `~/.cache/huggingface/lerobot/hirauchi/duck_pickup_v1/` が存在すること

## セットアップ

```bash
git clone https://github.com/furuya02/soarm101-lerobot-sagemaker-il.git
cd soarm101-lerobot-sagemaker-il

# CDK
cd cdk
pnpm install
pnpm exec cdk bootstrap
pnpm exec cdk deploy

# bucket suffix を上書きしたい場合
# pnpm exec cdk deploy -c bucket_suffix=20260511
```

出力:
- `BucketName`: `soarm101-lerobot-sagemaker-il-<account-id>`
- `SageMakerRoleArn`: `arn:aws:iam::<account-id>:role/soarm101-lerobot-sagemaker-il-sagemaker-execution-role`

## データセットのアップロード

```bash
aws s3 sync \
    ~/.cache/huggingface/lerobot/hirauchi/duck_pickup_v1/ \
    s3://soarm101-lerobot-sagemaker-il-<account-id>/datasets/duck_pickup_v1/
```

## 学習ジョブの起動（Managed Spot）

```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/soarm101-lerobot-sagemaker-il-sagemaker-execution-role
export S3_BUCKET=soarm101-lerobot-sagemaker-il-<account-id>
export USE_SPOT=true

pip install sagemaker
python src/submit.py
```

ACT を 30,000 step、`save_freq=5,000`、`batch_size=8`、`video_backend=pyav` で学習します。Spot 中断時はチェックポイントから自動再開します。

## 学習済みモデルのダウンロード

```bash
aws s3 cp \
    s3://soarm101-lerobot-sagemaker-il-<account-id>/output/<job-name>/output/model.tar.gz \
    ./model.tar.gz
mkdir -p outputs/sagemaker_model
tar xzf model.tar.gz -C outputs/sagemaker_model/
```

## コストに関する注意

`ml.g4dn.xlarge` Spot で 30,000 step を 1 回回した場合、推定費用は約 USD 0.30（約 50 円）です。終了後は S3 バケットの削除、または `checkpoints/` を 7 日でライフサイクル削除する設定で放置費用を抑えてください。

## ライセンス

MIT
