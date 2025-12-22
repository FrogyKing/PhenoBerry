import os
import boto3
import shutil
import yaml
import glob
import random
from datetime import datetime
from ultralytics import YOLO
# Importamos tu lógica compartida de tiling
from src.common.tiling import process_tiling

# --- CONFIGURACIÓN DE INFRAESTRUCTURA ---
BUCKET_RAW = "phenoberry-dev-raw-038876987034"       # <--- AJUSTA ESTO
BUCKET_ARTIFACTS = "phenoberry-dev-artifacts-038876987034" # <--- AJUSTA ESTO
S3_TRAIN_PREFIX = "training-dataset/" 

# Directorios Temporales Locales
LOCAL_RAW = "data_temp/raw"
LOCAL_TILED = "data_temp/tiled"
LOCAL_RUNS = "phenoberry_runs" 

# --- CONFIGURACIÓN DE DATASET Y BALANCEO ---
TARGET_EMPTY_RATIO = 0.15       # 15% del dataset serán imágenes sin objetos (background puro)
MIN_FLOWER_RATIO = 0.8          # Si un tile tiene >50% flores -> Aumentar
FLOWER_OVERSAMPLE_FACTOR = 10    
FLOWER_CLASS_ID = 0             
BLUEBERRY_CLASS_ID = 1

s3 = boto3.client('s3')

def download_s3_folder(bucket, prefix, local_dir):
    """Descarga recursiva de S3"""
    print(f"⬇️ Descargando datos desde {bucket}/{prefix}...")
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'): continue 
            rel_path = os.path.relpath(key, prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)

def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    """Sube resultados completos a S3"""
    print(f"Subiendo resultados a s3://{bucket}/{s3_prefix} ...")
    count = 0
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            s3.upload_file(local_path, bucket, s3_key)
            count += 1
    print(f"Subidos {count} archivos.")

def get_class_counts(label_dir):
    counts = {FLOWER_CLASS_ID: 0, BLUEBERRY_CLASS_ID: 0}
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for txt in txt_files:
        try:
            with open(txt, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        cls = int(parts[0])
                        if cls in counts: counts[cls] += 1
        except: pass
    return counts

def simple_report(name, c):
    tot = sum(c.values())
    if tot == 0: 
        print(f"   {name}: (Sin objetos / Posibles vacíos)")
        return
    print(f"   {name}: Flor={c[0]} ({c[0]/tot:.1%}), Aran={c[1]} ({c[1]/tot:.1%}) | Total Obj: {tot}")

def apply_balancing(train_img_dir, train_lbl_dir):
    print("\n Aplicando Balanceo de Clases...")
    txt_files = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
    stats = {'aug_imgs': 0, 'copies_made': 0}

    for txt_path in txt_files:
        n_target = 0
        total = 0
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        cls = int(line.split()[0])
                        if cls == FLOWER_CLASS_ID: n_target += 1
                        total += 1
                    except: pass
        except: continue # Archivo vacío o error
        
        ratio = (n_target / total) if total > 0 else 0.0
        
        if ratio >= MIN_FLOWER_RATIO:
            stats['aug_imgs'] += 1
            basename = os.path.basename(txt_path).replace('.txt', '')
            img_pattern = os.path.join(train_img_dir, basename + '.*')
            img_candidates = glob.glob(img_pattern)
            
            if not img_candidates: continue
            src_img = img_candidates[0]
            img_ext = os.path.splitext(src_img)[1]

            for i in range(FLOWER_OVERSAMPLE_FACTOR):
                new_name = f"{basename}_aug_{i}"
                shutil.copy(src_img, os.path.join(train_img_dir, new_name + img_ext))
                shutil.copy(txt_path, os.path.join(train_lbl_dir, new_name + ".txt"))
                stats['copies_made'] += 1
    
    print(f"Tiles clonados: {stats['aug_imgs']} | Copias creadas: {stats['copies_made']}")

def prepare_data_and_train():
    # 1. Limpieza
    if os.path.exists(LOCAL_RAW): shutil.rmtree(LOCAL_RAW)
    if os.path.exists(LOCAL_TILED): shutil.rmtree(LOCAL_TILED)
    if os.path.exists(LOCAL_RUNS): shutil.rmtree(LOCAL_RUNS)
    
    # 2. Descarga
    download_s3_folder(BUCKET_RAW, S3_TRAIN_PREFIX, LOCAL_RAW)
    
    # 3. Preparar directorios salida
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(f"{LOCAL_TILED}/images/{split}", exist_ok=True)
        os.makedirs(f"{LOCAL_TILED}/labels/{split}", exist_ok=True)

    # --- LÓGICA DE CLASIFICACIÓN (LLENOS vs VACÍOS) ---
    raw_images = glob.glob(f"{LOCAL_RAW}/images/*.jpg") + glob.glob(f"{LOCAL_RAW}/images/*.JPG") + glob.glob(f"{LOCAL_RAW}/images/*.png")
    
    populated_imgs = []
    empty_imgs = []
    
    print(f"\n Clasificando {len(raw_images)} imágenes originales (Con Objetos vs Vacías)...")
    for img_path in raw_images:
        filename = os.path.basename(img_path).rsplit('.', 1)[0]
        lbl_path = os.path.join(LOCAL_RAW, 'labels', filename + '.txt')
        
        has_objects = False
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                if len(f.read().strip()) > 0:
                    has_objects = True
        
        if has_objects:
            populated_imgs.append(img_path)
        else:
            empty_imgs.append(img_path)

    # Calcular cuántos vacíos necesitamos
    n_pop = len(populated_imgs)
    # Fórmula: ratio = n_empty / (n_pop + n_empty)  => despejando n_empty
    if n_pop > 0:
        n_empty_target = int((TARGET_EMPTY_RATIO * n_pop) / (1 - TARGET_EMPTY_RATIO))
    else:
        n_empty_target = len(empty_imgs)

    # Seleccionar vacíos aleatorios
    random.seed(42)
    random.shuffle(empty_imgs)
    selected_empty = empty_imgs[:min(n_empty_target, len(empty_imgs))]
    
    final_dataset = populated_imgs + selected_empty
    random.shuffle(final_dataset)
    
    print(f" Dataset Base: {len(populated_imgs)} con objetos + {len(selected_empty)} vacíos = {len(final_dataset)} Total")

    # 4. SPLIT (80/10/10)
    n_total = len(final_dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    split_map = {
        'train': final_dataset[:n_train],
        'val': final_dataset[n_train:n_train + n_val],
        'test': final_dataset[n_train + n_val:]
    }

    # 5. TILING
    print("\n  Ejecutando Tiling...")
    for split_name, img_list in split_map.items():
        for img_path in img_list:
            filename = os.path.basename(img_path).rsplit('.', 1)[0]
            lbl_path = os.path.join(LOCAL_RAW, 'labels', filename + '.txt')
            
            # Si es vacío, puede que no exista el txt, o esté vacío.
            # process_tiling maneja lbl_path=None generando txts vacíos si no se pasa output_lbl
            # Pero queremos generar txts vacíos explícitos para YOLO
            
            current_lbl_path = lbl_path if os.path.exists(lbl_path) else None
                
            process_tiling(
                img_path=img_path,
                output_dir_img=f"{LOCAL_TILED}/images/{split_name}",
                output_dir_lbl=f"{LOCAL_TILED}/labels/{split_name}",
                lbl_path=current_lbl_path,
                filename_prefix=filename
            )

    # 6. REPORTES Y BALANCEO
    print("\n --- ESTADÍSTICAS PRE-BALANCEO ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))
    simple_report("Val  ", get_class_counts(f"{LOCAL_TILED}/labels/val"))

    apply_balancing(f"{LOCAL_TILED}/images/train", f"{LOCAL_TILED}/labels/train")

    print("\n --- ESTADÍSTICAS POST-BALANCEO ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))

    # 7. YAML
    data_yaml = {
        'path': os.path.abspath(LOCAL_TILED),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'flor', 1: 'arandano'}
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # 8. ENTRENAMIENTO
    print("\n Iniciando Entrenamiento YOLOv8 (100 Épocas)...")
    model = YOLO('yolov8n.pt') 
    experiment_name = 'yolo_final_experiment'
    
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        project=LOCAL_RUNS,
        name=experiment_name,
        patience=15,
        # --- Aumentos geométricos ---
        mosaic=0.0,      # desactivado
        mixup=0.2,       # mezcla suave de imágenes
        degrees=10.0,    # rotación aleatoria +-10°
        translate=0.1,   # traslación +-10% del tamaño
        scale=0.1,       # escalado +-10%
        shear=5.0,       # cizallamiento +-5°
        perspective=0.0, # sin perspectiva
        flipud=0.1,      # volteo vertical ocasional
        fliplr=0.5,      # volteo horizontal frecuente
        # --- Aumentos de color (HSV) ---
        hsv_h=0.01,      # cambio leve de tono
        hsv_s=0.2,       # saturación moderada
        hsv_v=0.2        # brillo moderado
    )
    
    # 9. SUBIDA A S3
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    results_dir = os.path.join(LOCAL_RUNS, experiment_name)
    
    best_model = os.path.join(results_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        s3_key_model = f"models/yolo/yolo_v{run_id}.pt"
        s3.upload_file(best_model, BUCKET_ARTIFACTS, s3_key_model)
        print(f"\n🏆 Modelo: s3://{BUCKET_ARTIFACTS}/{s3_key_model}")
    
    s3_prefix_logs = f"training_logs/run_{run_id}"
    if os.path.exists(results_dir):
        upload_folder_to_s3(results_dir, BUCKET_ARTIFACTS, s3_prefix_logs)

if __name__ == "__main__":
    prepare_data_and_train()