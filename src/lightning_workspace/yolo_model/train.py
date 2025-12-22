import os
import boto3
import shutil
import yaml
from ultralytics import YOLO
from src.common.tiling import process_tiling

# --- CONFIGURACIÓN ---
BUCKET_RAW = "phenoberry-dev-raw-038876987034"       # <--- PON TU BUCKET REAL
BUCKET_ARTIFACTS = "phenoberry-dev-artifacts-038876987034" # <--- PON TU BUCKET REAL
S3_TRAIN_PREFIX = "training-dataset/"  # Carpeta en S3 donde subiste la data etiquetada

LOCAL_RAW = "data_temp/raw"
LOCAL_TILED = "data_temp/tiled"

s3 = boto3.client('s3')

def download_s3_folder(bucket, prefix, local_dir):
    """Descarga recursiva de S3"""
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'): continue # Es carpeta
            
            # Estructura local
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            s3.download_file(bucket, key, local_path)
    print(f"Descarga completa de {prefix} a {local_dir}")

def prepare_data():
    """Descarga raw y ejecuta el Tiling"""
    if os.path.exists(LOCAL_RAW): shutil.rmtree(LOCAL_RAW)
    if os.path.exists(LOCAL_TILED): shutil.rmtree(LOCAL_TILED)
    
    print("--- 1. Descargando datos de S3 ---")
    download_s3_folder(BUCKET_RAW, S3_TRAIN_PREFIX, LOCAL_RAW)
    
    print("--- 2. Ejecutando Tiling (con etiquetas) ---")
    # Estructura esperada en S3: training-dataset/images y training-dataset/labels
    
    # Creamos carpetas YOLO standard
    splits = ['train', 'val']
    for split in splits:
        os.makedirs(f"{LOCAL_TILED}/images/{split}", exist_ok=True)
        os.makedirs(f"{LOCAL_TILED}/labels/{split}", exist_ok=True)

    # Obtenemos lista de imagenes
    import glob
    images = glob.glob(f"{LOCAL_RAW}/images/*.jpg") + glob.glob(f"{LOCAL_RAW}/images/*.png")
    
    # Simple split 80/20 manual
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    for idx, img_path in enumerate(images):
        # Determinar si es train o val
        split = 'train' if idx < split_idx else 'val'
        
        filename = os.path.basename(img_path).rsplit('.', 1)[0]
        lbl_path = os.path.join(LOCAL_RAW, 'labels', filename + '.txt')
        
        if not os.path.exists(lbl_path):
            print(f"Skipping {filename}, no label found.")
            continue
            
        # LLAMADA A TU CÓDIGO COMÚN
        process_tiling(
            img_path=img_path,
            output_dir_img=f"{LOCAL_TILED}/images/{split}",
            output_dir_lbl=f"{LOCAL_TILED}/labels/{split}",
            lbl_path=lbl_path,
            filename_prefix=filename
        )
    print("Tiling finalizado.")

def create_yaml():
    """Crea el archivo data.yaml para YOLO"""
    data = {
        'path': os.path.abspath(LOCAL_TILED),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'flor', 1: 'arandano'} # Ajusta según tus clases
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data, f)

def train_and_upload():
    print("--- 3. Iniciando Entrenamiento YOLO ---")
    create_yaml()
    
    # Cargar modelo base (nano para probar rápido, usa 'yolov8m.pt' o 'l' para tesis)
    model = YOLO('yolov8n.pt') 
    
    # Entrenar
    results = model.train(
        data='data.yaml',
        epochs=50,  # Ajusta para tesis
        imgsz=640,
        batch=16,
        project='phenoberry_runs',
        name='yolo_experiment'
    )
    
    print("--- 4. Subiendo modelo a S3 ---")
    best_model_path = f"phenoberry_runs/yolo_experiment/weights/best.pt"
    
    if os.path.exists(best_model_path):
        # Versionado simple por fecha
        from datetime import datetime
        ver = datetime.now().strftime("%Y%m%d-%H%M")
        s3_key = f"models/yolo/yolo_v{ver}.pt"
        
        s3.upload_file(best_model_path, BUCKET_ARTIFACTS, s3_key)
        print(f"✅ Modelo guardado en: s3://{BUCKET_ARTIFACTS}/{s3_key}")
    else:
        print("❌ Error: No se encontró el modelo entrenado.")

if __name__ == "__main__":
    prepare_data()
    train_and_upload()