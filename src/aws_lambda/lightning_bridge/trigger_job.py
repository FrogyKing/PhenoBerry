import os
import json
# NUEVA IMPORTACIÓN (La correcta para la versión actual)
from lightning_sdk import Studio, Machine, Job

def lambda_handler(event, context):
    print("🚀 Iniciando solicitud de Reentrenamiento a Lightning AI...")

    try:
        # 1. Configuración (Variables de Entorno)
        user_id = os.environ['LIGHTNING_USER_ID']
        api_key = os.environ['LIGHTNING_API_KEY']
        
        # OJO: Ajusta esto si tu Studio tiene otro nombre o teamspace
        # Si creaste el Studio manualmente, pon su nombre aquí.
        STUDIO_NAME = "lightning_GPU"  
        TEAMSPACE = "Vision-model" # Por defecto suele ser tu usuario, o el nombre de tu equipo
        USER_ID = "luisgonzalezalvarez991"
        #USER = "Luis Gonzalez Alvarez"
        # 2. Referenciar el Studio (El "Controlador")
        # Esto no crea uno nuevo, se conecta al que ya tienes para lanzar el Job desde ahí
        studio = Studio(name=STUDIO_NAME, teamspace=TEAMSPACE, user=USER_ID)
        studio.start()
        
        # 3. Definir el Comando
        # Usamos --no-cache-dir para evitar problemas de espacio
        # Y 'python -m' para asegurar ejecución correcta
        cmd = "git clone https://github.com/FrogyKing/PhenoBerry.git && " \
              "pip install --no-cache-dir -r PhenoBerry/requirements.txt && " \
              "pip install --no-cache-dir ultralytics PyYAML boto3 && " \
              "python PhenoBerry/src/lightning_workspace/yolo_model/train.py"

        print(f"Comando: {cmd}")

        # 4. Lanzar el Job (Fire & Forget)
        # Usamos Machine.T4 (Barata y buena) o Machine.A10G (Más rápida)
        job = Job.run(
            command=cmd,
            machine=Machine.A100,
            studio=studio,
            name=f"retrain-yolo-{context.aws_request_id[:8]}",
            env={
                "AWS_ACCESS_KEY_ID": os.environ['MY_AWS_KEY'],
                "AWS_SECRET_ACCESS_KEY": os.environ['MY_AWS_SECRET'],
                "AWS_REGION": "us-east-1"
            }
        )
        studio.stop()
        print(f"✅ Job enviado exitosamente: {job.name}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"Job iniciado: {job.name}")
        }

    except Exception as e:
        print(f"❌ Error lanzando job: {str(e)}")
        # Importante: No lanzamos 'raise' si queremos evitar reintentos infinitos de Lambda
        # Pero para debug, déjalo.
        raise e