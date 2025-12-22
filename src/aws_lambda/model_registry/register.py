import boto3
import urllib.parse
import os

sagemaker = boto3.client('sagemaker')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    # Detecta que apareció un archivo .pt nuevo
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(record['s3']['object']['key'])
    
    model_url = f"s3://{bucket}/{key}"
    model_name = key.split('/')[-1].replace('.pt', '') # ej: yolo_v20251221
    
    print(f"Nuevo modelo detectado: {model_name}")

    # 1. Registrar en SageMaker Model Registry (Opcional pero muy Pro)
    # Crea un 'Model Package Group' si no existe y agrega la versión.
    try:
        # Simplificado: Guardar solo en DynamoDB como "Latest Model"
        table = dynamodb.Table(os.environ['DYNAMO_TABLE'])
        
        table.put_item(Item={
            'media_id': 'LATEST_YOLO_MODEL', # ID Fijo para buscarlo rápido
            'model_version': model_name,
            's3_path': model_url,
            'status': 'READY_TO_DEPLOY',
            'timestamp': record['eventTime']
        })
        print("Modelo registrado en DynamoDB como Production Candidate")
        
    except Exception as e:
        print(e)
        raise e