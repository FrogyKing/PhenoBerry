import os
import boto3
import json
from lightning_sdk import Machine, Studio, LightningClient

# Estas variables vendrán del entorno (Secrets)
LIGHTNING_USER_ID = os.environ['LIGHTNING_USER_ID']
LIGHTNING_API_KEY = os.environ['LIGHTNING_API_KEY']
LIGHTNING_PROJECT = os.environ['LIGHTNING_PROJECT'] # Nombre de tu proyecto en Lightning

def lambda_handler(event, context):
    print("🚀 Iniciando solicitud de Reentrenamiento a Lightning AI...")

    try:
        # 1. Autenticación (El cliente usa variables de entorno o config file)
        # Configuración manual del cliente si es necesario, o seteo de env vars
        os.environ['LIGHTNING_API_KEY'] = LIGHTNING_API_KEY
        
        client = LightningClient()
        project = client.projects_service_list_memberships()[0].project_id # Obtiene el ID del proyecto por defecto
        
        # 2. Definir el Job
        # Esto le dice a Lightning: "Arranca una máquina T4, clona mi repo y corre este comando"
        
        # NOTA: Para MLOps real, tu código de entrenamiento debe estar en el repo.
        # Aquí asumimos que Lightning clona tu repo de GitHub antes de correr.
        
        cmd = "git clone https://github.com/[TU_USUARIO]/PhenoBerry.git && " \
              "pip install -r PhenoBerry/requirements.txt && " \
              "pip install ultralytics boto3 && " \
              "python PhenoBerry/src/lightning_workspace/yolo_model/train.py"

        print(f"Comando a ejecutar: {cmd}")

        # 3. Lanzar el Job (Usando la API de Jobs)
        # Nota: La SDK de Python de Lightning a veces cambia, esta es la forma genérica de 'run job'
        # Si usas 'Studio', puedes hacer studio.run(cmd)
        
        # Simplificación: Usamos un Studio existente o creamos un Job
        # Para tu tesis, lo más robusto es lanzar un Job en una máquina nueva
        
        job = client.job_service_create_job(
            project_id=project,
            name=f"retrain-yolo-{context.aws_request_id[:8]}",
            machine_name="gpu-t4", # O la máquina que tengas créditos (ej: A10G)
            command=cmd,
            env=[
                # Pasamos las credenciales de AWS al Job para que pueda leer/escribir en S3
                {"name": "AWS_ACCESS_KEY_ID", "value": os.environ['AWS_ACCESS_KEY_ID']},
                {"name": "AWS_SECRET_ACCESS_KEY", "value": os.environ['AWS_SECRET_ACCESS_KEY']},
                {"name": "AWS_REGION", "value": os.environ['AWS_REGION']}
            ]
        )

        print(f"✅ Job enviado a Lightning AI. ID: {job.id}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"Entrenamiento iniciado: {job.name}")
        }

    except Exception as e:
        print(f"❌ Error lanzando job: {str(e)}")
        raise e