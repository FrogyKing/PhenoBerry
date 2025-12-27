import boto3
import os
import json
from datetime import datetime

sm_client = boto3.client('sagemaker')

def lambda_handler(event, context):
    # 1. Generar un nombre único para el entrenamiento
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"phenoberry-yolo-{timestamp}"
    
    print(f"🚀 Iniciando SageMaker Training Job: {job_name}")

    try:
        # 2. Configurar el Job
        response = sm_client.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                # Imagen oficial de AWS para PyTorch (incluye soporte GPU)
                'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310',
                'TrainingInputMode': 'File'
            },
            RoleArn=os.environ['SAGEMAKER_ROLE_ARN'],
            InputDataConfig=[{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f"s3://{os.environ['RAW_BUCKET']}/training-dataset/",
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                }
            }],
            OutputDataConfig={
                'S3OutputPath': f"s3://{os.environ['ARTIFACTS_BUCKET']}/sagemaker-runs/"
            },
            ResourceConfig={
                'InstanceType': 'ml.g4dn.xlarge', # Instancia con GPU (Económica)
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': 86400 # 24 horas máximo
            },
            # --- MLOps: Aquí conectamos con tu código en GitHub ---
            HyperParameters={
                'sagemaker_program': 'src/sagemaker_training/yolo_task/train_yolo.py',
                'sagemaker_submit_directory': f"https://github.com/FrogyKing/PhenoBerry/archive/refs/heads/main.tar.gz"
            }
        )

        print(f"✅ Job creado exitosamente. ARN: {response['TrainingJobArn']}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({'TrainingJobArn': response['TrainingJobArn']})
        }

    except Exception as e:
        print(f"❌ Error al lanzar el Job: {str(e)}")
        raise e