FROM public.ecr.aws/lambda/python:3.10

RUN yum install git -y
COPY requirements.txt ./

RUN python3.10 -m pip install -r requirements.txt -t .
RUN python3.10 -m pip install 'stability-sdk[sagemaker] @ git+https://github.com/Stability-AI/stability-sdk.git@sagemaker'
# Command can be overwritten by providing a different command in the template directly.
COPY app.py ./

CMD ["app.lambda_handler"]
