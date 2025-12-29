import json, os
import boto3
from src.sagemaker_training.yolo_task.infer_yolo import run_inference

s3_client = boto3.client('s3')
PROCESSED_BUCKET = os.environ['PROCESSED_BUCKET']

def lambda_handler(event, context):
    for record in event['Records']:
        s3_key = record['s3']['object']['key'] # ej: tiles/test_grid4x3_r0c0.jpg
        
        # Extraemos el nombre del archivo sin la extensión y sin el prefijo
        file_name = os.path.basename(s3_key) # test_grid4x3_r0c0.jpg
        tile_id = os.path.splitext(file_name)[0] # test_grid4x3_r0c0
        
        run_inference(f"s3://{PROCESSED_BUCKET}/{s3_key}", tile_id)
    
    return {'statusCode': 200, 'body': json.dumps('Inference done')}
