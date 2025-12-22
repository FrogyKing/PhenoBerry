FROM public.ecr.aws/lambda/python:3.11

# Instalar git para el sdk si lo requiere
RUN yum install -y git

# Instalar SDK de Lightning
RUN pip install lightning-sdk requests boto3

COPY src/aws_lambda/lightning_bridge ${LAMBDA_TASK_ROOT}/src/aws_lambda/lightning_bridge

CMD [ "src.aws_lambda.lightning_bridge.trigger_job.lambda_handler" ]