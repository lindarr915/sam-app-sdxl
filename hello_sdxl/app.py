import json
import boto3
import base64
import io

import sagemaker
from stability_sdk_sagemaker.predictor import StabilityPredictor
from stability_sdk.api import GenerationRequest, TextPrompt


sagemaker_session = sagemaker.Session()


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    data = json.loads(event['body'])

    # get the parameters from the event and provide the default values
    prompt = data["prompt"] or  "A beautiful sunset over the ocean."

    seed = data["seed"]
    cfg_scale = data["cfg_scale"]
    style_preset = data["style_preset"]
    sampler=data["sampler"]

    try:
        deployed_model = StabilityPredictor(
            endpoint_name="sdxl-1-jumpstart", sagemaker_session=sagemaker_session)

        output = deployed_model.predict(GenerationRequest(
            # [TextPrompt(text=prompt)]
            text_prompts=[TextPrompt(text=prompt)],
            seed=seed,  # payload['seed']
            style_preset=style_preset,
            cfg_scale=cfg_scale,
            sampler=sampler
        ))
        print(output)
        response = io.BytesIO(base64.b64decode(
            (output.artifacts[0].base64.encode()))).getvalue()

    except Exception as err:
        # Send some context about this error to Lambda Logs
        response = err
        print(response)

    return {
        "statusCode": 200,
        'headers': {"Content-Type": "image/png"},
        'body': base64.b64encode(response).decode('utf-8'),
        'isBase64Encoded': True
    }
