FROM public.ecr.aws/lambda/python:3.11

# 1. Instalar GCC, G++ y herramientas de compilación necesarias
# También incluimos mesa-libGL para OpenCV (Ultralytics)
RUN yum update -y && \
    yum install -y gcc gcc-c++ make mesa-libGL && \
    yum clean all

# 2. Copiar requerimientos
COPY src/aws_lambda/inference_cv/requirements.txt .

# 3. Actualizar pip e instalar librerías
# Mantenemos el límite de numpy para evitar el problema de versiones de GCC
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código del proyecto (Contexto es la raíz ..)
COPY . ${LAMBDA_TASK_ROOT}

CMD [ "src.aws_lambda.inference_cv.inference.lambda_handler" ]