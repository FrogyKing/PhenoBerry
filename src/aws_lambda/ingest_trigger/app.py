import json
import os
import boto3
import uuid
from datetime import datetime, timezone
import urllib.parse

# Inicializar clientes fuera del handler para reusar conexiones
dynamodb = boto3.resource('dynamodb')
TABLE_NAME = os.environ['DYNAMO_TABLE']
table = dynamodb.Table(TABLE_NAME)

def lambda_handler(event, context):
    try:
        # 1. Leer evento de S3
        # El evento puede traer múltiples registros, iteramos (aunque usualmente es 1)
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            # Decodificar el nombre del archivo (espacios suelen venir como + o %20)
            file_key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
            
            print(f"Procesando archivo: {file_key} del bucket: {bucket_name}")

            # 2. Generar Metadata Inicial
            # Usamos un nombre unico para el ID
            media_id = str(uuid.uuid4()) 
            timestamp = datetime.now(timezone.utc).isoformat()
            
            item = {
                'media_id': media_id,         # Primary Key
                's3_bucket': bucket_name,
                's3_key': file_key,
                'upload_timestamp': timestamp,
                'status': 'UPLOADED',         # Estado inicial
                'ml_stage': 'pending_tiling', # Siguiente paso en el flujo
                'original_filename': file_key.split('/')[-1]
            }

            # 3. Guardar en DynamoDB
            table.put_item(Item=item)
            print(f"Registro creado en DynamoDB para: {media_id}")

        return {
            'statusCode': 200,
            'body': json.dumps('Ingesta exitosa')
        }

    except Exception as e:
        print(f"Error en ingesta: {str(e)}")
        # Lanzar el error para que AWS reintente o mande a logs
        raise e