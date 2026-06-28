import io
import json
import os
import time
import zipfile

import boto3
from botocore.exceptions import ClientError
from bedrock_agentcore_starter_toolkit.operations.gateway import GatewayClient

REGION = os.getenv("AWS_REGION", "ap-south-1")

LAMBDA_NAME = "finbot-yahoo-news"
LAMBDA_ROLE_NAME = "finbot-yahoo-news-role"
# Local file packaged into the function. Anchored to this script's directory so
# it resolves no matter what the current working directory is.
LAMBDA_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gateway_lambda.py")

# Unique per run so re-running never collides with a previously created gateway.
GATEWAY_NAME = "finbot-gateway-auto-" + GatewayClient.generate_random_id()
TARGET_NAME = "finbotyahoo"  # becomes the tool prefix: finbotyahoo___yahoo_finance_news

# The tool's interface, described for the LLM (same schema as the console).
TOOL_SCHEMA = {
    "inlinePayload": [
        {
            "name": "yahoo_finance_news",
            "description": "Get recent Yahoo Finance news headlines for a stock ticker.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"}
                },
                "required": ["ticker"],
            },
        }
    ]
}


def ensure_lambda() -> str:
    """Create the execution role + Lambda from gateway_lambda.py. Returns its ARN."""
    iam = boto3.client("iam", region_name=REGION)
    lam = boto3.client("lambda", region_name=REGION)

    # 1. Execution role the Lambda runs as (basic logging permissions).
    assume = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        role_arn = iam.create_role(
            RoleName=LAMBDA_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(assume),
        )["Role"]["Arn"]
        iam.attach_role_policy(
            RoleName=LAMBDA_ROLE_NAME,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )
        print(f"  ✓ created Lambda execution role {LAMBDA_ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=LAMBDA_ROLE_NAME)["Role"]["Arn"]
        print(f"  • reusing Lambda execution role {LAMBDA_ROLE_NAME}")

    # 2. Package gateway_lambda.py into a zip (as lambda_function.py).
    buf = io.BytesIO()
    with open(LAMBDA_SOURCE, "r", encoding="utf-8") as f:
        code = f.read()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("lambda_function.py", code)
    code_zip = buf.getvalue()

    # 3. Create the function (retry while the new role propagates).
    try:
        for attempt in range(6):
            try:
                fn = lam.create_function(
                    FunctionName=LAMBDA_NAME,
                    Runtime="python3.12",
                    Role=role_arn,
                    Handler="lambda_function.handler",
                    Code={"ZipFile": code_zip},
                    Timeout=15,
                )
                print(f"  ✓ created Lambda {LAMBDA_NAME}")
                break
            except lam.exceptions.InvalidParameterValueException:
                # role not assumable yet — wait and retry
                print(f"  … waiting for role to propagate ({attempt + 1}/6)")
                time.sleep(5)
        else:
            raise RuntimeError("Lambda role did not propagate in time")
    except lam.exceptions.ResourceConflictException:
        # already exists — update its code so the latest source is live
        lam.update_function_code(FunctionName=LAMBDA_NAME, ZipFile=code_zip)
        fn = lam.get_function(FunctionName=LAMBDA_NAME)["Configuration"]
        print(f"  • updated existing Lambda {LAMBDA_NAME}")

    return fn["FunctionArn"]


def main():
    print("→ Creating the Lambda...")
    lambda_arn = ensure_lambda()

    client = GatewayClient(region_name=REGION)

    print("→ Creating Cognito auth...")
    cognito = client.create_oauth_authorizer_with_cognito(GATEWAY_NAME)

    print("→ Creating Gateway...")
    gateway = client.create_mcp_gateway(
        name=GATEWAY_NAME,
        authorizer_config=cognito["authorizer_config"],
        enable_semantic_search=False,
        enable_observability=False,
    )

    print("→ Granting the gateway role permission to invoke the Lambda...")
    role_name = gateway["roleArn"].split("/")[-1]
    boto3.client("iam", region_name=REGION).put_role_policy(
        RoleName=role_name,
        PolicyName="invoke-finbot-yahoo-news",
        PolicyDocument=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "lambda:InvokeFunction",
                        "Resource": lambda_arn,
                    }
                ],
            }
        ),
    )

    print("→ Adding the Lambda target...")
    # IAM permissions take a few seconds to propagate, so the gateway role may not
    # yet be allowed to invoke the Lambda. Retry until the permission is live.
    for attempt in range(6):
        try:
            client.create_mcp_gateway_target(
                gateway=gateway,
                name=TARGET_NAME,
                target_type="lambda",
                target_payload={"lambdaArn": lambda_arn, "toolSchema": TOOL_SCHEMA},
            )
            break
        except ClientError as e:
            if "lacks permission" in str(e) and attempt < 5:
                print(f"  … waiting for invoke permission to propagate ({attempt + 1}/6)")
                time.sleep(10)
                continue
            raise

    print("→ Fetching an access token...")
    token = client.get_access_token_for_cognito(cognito["client_info"])

    print("\n✅ Everything created from scratch. Paste these into your .env:\n")
    print(f"GATEWAY_URL={gateway['gatewayUrl']}")
    print(f"GATEWAY_ACCESS_TOKEN={token}")
    print("\nThen run:  uv run python gateway/gateway_demo.py \"latest news on NVDA\"")


if __name__ == "__main__":
    main()
