# Usamos una imagen base de AWS Lambda para Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# 1. Instalar dependencias del sistema operativo necesarias para OpenCV
# (OpenCV necesita librerias gráficas como libGL, aunque sea headless)
RUN yum install -y mesa-libGL \
    && yum clean all \
    && rm -rf /var/cache/yum

# 2. Copiar el archivo de requerimientos e instalar librerias Python
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir --only-binary=:all: \
    -r requirements.txt \
    --target "${LAMBDA_TASK_ROOT}"

# 3. Copiar el codigo fuente
# OJO: Copiamos la carpeta 'common' dentro de una carpeta src en el contenedor
# para mantener la estructura de imports que usas en tu código (from src.common import...)
COPY src/common ${LAMBDA_TASK_ROOT}/src/common
COPY src/aws_lambda/inference_coordinator ${LAMBDA_TASK_ROOT}/src/aws_lambda/inference_coordinator

# 4. Configurar la variable de entorno para que Python encuentre src
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# 5. Definir el comando que se ejecutara (el handler de la Lambda)
# Apuntamos al archivo tiler.py que crearemos en el siguiente paso
CMD [ "src.aws_lambda.inference_coordinator.tiler.lambda_handler" ]