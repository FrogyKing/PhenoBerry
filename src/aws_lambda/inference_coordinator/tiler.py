import json
import os
import boto3
import uuid
import shutil
import urllib.parse
from datetime import datetime, timezone
from src.common.tiling import process_tiling

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET')
TABLE_NAME = os.environ.get('DYNAMO_TABLE')
table = dynamodb.Table(TABLE_NAME)

def lambda_handler(event, context):
    # Directorios temporales
    local_input_path = "/tmp/input_image.jpg"
    output_dir = "/tmp/tiles"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        record = event['Records'][0]
        source_bucket = record['s3']['bucket']['name']
        # Decodificar nombre (evita errores con espacios o tildes)
        source_key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
        
        print(f"Procesando: {source_key}")

        # --- 1. REGISTRO MLOps (DynamoDB) ---
        # Registramos que llegó la imagen y empezamos a procesar
        media_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        item = {
            'media_id': media_id,
            'original_filename': source_key,
            's3_raw_bucket': source_bucket,
            'upload_timestamp': timestamp,
            'status': 'PROCESSING_TILING', # Estado intermedio
            'ml_stage': 'preprocessing'
        }
        table.put_item(Item=item)

        # --- 2. DESCARGA ---
        s3_client.download_file(source_bucket, source_key, local_input_path)

        # --- 3. TILING (Lógica Compartida) ---
        # CORRECCIÓN: Usamos rsplit('.', 1) para quitar SOLO la extensión final (.jpg)
        # Esto respeta los puntos intermedios en el nombre del archivo (ej: 18.0)
        relative_path = source_key.replace("uploads/", "")
        filename = os.path.basename(relative_path)
        filename_prefix = filename.rsplit('.', 1)[0]
        
        generated_files = process_tiling(
            img_path=local_input_path,
            output_dir_img=output_dir,
            output_dir_lbl=None, # Inferencia = Sin etiquetas
            lbl_path=None,
            filename_prefix=filename_prefix
        )
        
        # --- 4. SUBIDA DE TILES ---
        uploaded_tiles = []
        for file_path in generated_files:
            file_name = os.path.basename(file_path)
            # Guardamos en carpeta con el nombre de la foto original dentro de tiles
            s3_dest_key = f"tiles/{filename_prefix}/{file_name}"
            
            s3_client.upload_file(file_path, PROCESSED_BUCKET, s3_dest_key)
            uploaded_tiles.append(s3_dest_key)

        # --- 5. ACTUALIZACIÓN MLOps ---
        # Actualizamos DynamoDB para decir que terminamos
        table.update_item(
            Key={'media_id': media_id},
            UpdateExpression="set #st = :s, total_tiles = :t, processed_timestamp = :pt",
            ExpressionAttributeNames={'#st': 'status'},
            ExpressionAttributeValues={
                ':s': 'TILED_COMPLETE',
                ':t': len(uploaded_tiles),
                ':pt': datetime.now(timezone.utc).isoformat()
            }
        )

        return {
            'statusCode': 200,
            'body': json.dumps({'tiles_created': len(uploaded_tiles), 'media_id': media_id})
        }

    except Exception as e:
        print(f"Error critico: {str(e)}")
        raise e