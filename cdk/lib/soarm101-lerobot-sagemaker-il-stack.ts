import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface SoarmStackProps extends cdk.StackProps {
  projectName: string;
  bucketSuffix: string;
}

export class SoarmStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: SoarmStackProps) {
    super(scope, id, props);
    const { projectName, bucketSuffix } = props;

    const bucket = new s3.Bucket(this, "ArtifactBucket", {
      bucketName: `${projectName}-${bucketSuffix}`,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        { prefix: "checkpoints/", expiration: cdk.Duration.days(7) },
      ],
    });

    const sagemakerRole = new iam.Role(this, "SageMakerExecutionRole", {
      roleName: `${projectName}-sagemaker-execution-role`,
      assumedBy: new iam.ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ],
    });

    sagemakerRole.attachInlinePolicy(
      new iam.Policy(this, "S3RwPolicy", {
        policyName: `${projectName}-sagemaker-s3-rw-policy`,
        statements: [
          new iam.PolicyStatement({
            actions: [
              "s3:GetObject",
              "s3:PutObject",
              "s3:DeleteObject",
              "s3:ListBucket",
            ],
            resources: [bucket.bucketArn, `${bucket.bucketArn}/*`],
          }),
        ],
      })
    );

    new cdk.CfnOutput(this, "BucketName", { value: bucket.bucketName });
    new cdk.CfnOutput(this, "SageMakerRoleArn", { value: sagemakerRole.roleArn });
  }
}
