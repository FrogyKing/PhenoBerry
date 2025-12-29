import os
import sys
import shutil
import yaml
import glob
import random
import json
from datetime import datetime
from ultralytics import YOLO
import boto3
# ---------------------------------------------------------
# CONFIGURACIÓN DE RUTAS SAGEMAKER (ESTÁNDAR AWS)
# ---------------------------------------------------------
# SageMaker descarga los datos de S3 aquí:

if "/opt/ml/code" not in sys.path:
    sys.path.insert(0, "/opt/ml/code")

DATA_PATH = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
# Todo lo que guardes aquí, SageMaker lo sube a S3 como model.tar.gz al terminar:
MODEL_OUTPUT = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
# Carpeta para gráficas, matriz de confusión, etc:
OUTPUT_DATA = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
# Bucket de artifacts
S3_BUCKET = os.environ.get('SM_HP_ARTIFACTS_BUCKET')

# Verificación de seguridad
if not S3_BUCKET:
    print("❌ ERROR: No se recibió la variable S3_BUCKET desde los hiperparámetros.")

# El código del repo se extrae en /opt/ml/code
sys.path.append("/opt/ml/code")
from src.common.tiling import process_tiling

# Usamos /tmp para el procesamiento intermedio (es el disco local del contenedor)
LOCAL_TILED = "/tmp/tiled"
LOCAL_RUNS = "/tmp/runs"

# Parámetros del Dataset
TARGET_EMPTY_RATIO = 0.15
MIN_FLOWER_RATIO = 0.8
FLOWER_OVERSAMPLE_FACTOR = 10
FLOWER_CLASS_ID = 0
BLUEBERRY_CLASS_ID = 1

s3_client = boto3.client('s3')

# =========================================================
# UTILIDADES DE REPORTE
# =========================================================

def get_class_counts(label_dir):
    counts = {FLOWER_CLASS_ID: 0, BLUEBERRY_CLASS_ID: 0}
    for txt in glob.glob(os.path.join(label_dir, "*.txt")):
        try:
            with open(txt) as f:
                for line in f:
                    cls = int(line.split()[0])
                    if cls in counts:
                        counts[cls] += 1
        except:
            pass
    return counts

def simple_report(name, counts):
    total = sum(counts.values())
    if total == 0:
        print(f"{name}: sin objetos")
        return
    print(
        f"{name}: Flor={counts[0]} ({counts[0]/total:.1%}) | "
        f"Arándano={counts[1]} ({counts[1]/total:.1%}) | Total={total}"
    )

# =========================================================
# BALANCE BACKGROUND AUTOMÁTICO
# =========================================================

def enforce_background_ratio_train(img_dir, lbl_dir, background_ratio=0.15, seed=42):
    random.seed(seed)
    label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    empty_tiles, non_empty_tiles = [], []

    for lbl in label_files:
        try:
            with open(lbl, "r") as f:
                content = f.read().strip()
            if content == "":
                empty_tiles.append(lbl)
            else:
                non_empty_tiles.append(lbl)
        except:
            continue

    total_tiles = len(label_files)
    if total_tiles == 0: return

    target_empty = int(total_tiles * background_ratio)
    print(f"\n🎯 Control background: Tot={total_tiles}, Vacíos={len(empty_tiles)}, Obj={target_empty}")

    if len(empty_tiles) > target_empty:
        random.shuffle(empty_tiles)
        to_remove = empty_tiles[target_empty:]
        for lbl in to_remove:
            base = os.path.splitext(os.path.basename(lbl))[0]
            for img in glob.glob(os.path.join(img_dir, base + ".*")):
                os.remove(img)
            os.remove(lbl)
        print(f"🧹 Eliminados {len(to_remove)} tiles vacíos adicionales.")

# =========================================================
# OVERSAMPLING DE FLORES
# =========================================================

def apply_balancing(train_img_dir, train_lbl_dir):
    print("\n🌸 Aplicando oversampling de flores...")
    txt_files = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
    stats = {"aug_imgs": 0, "copies": 0}

    for txt_path in txt_files:
        try:
            with open(txt_path) as f:
                lines = f.readlines()
        except: continue

        total = flower_count = 0
        for l in lines:
            try:
                cls = int(l.split()[0])
                total += 1
                if cls == FLOWER_CLASS_ID: flower_count += 1
            except: pass

        ratio = flower_count / total if total > 0 else 0.0

        if ratio >= MIN_FLOWER_RATIO:
            basename = os.path.basename(txt_path).replace(".txt", "")
            img_candidates = glob.glob(os.path.join(train_img_dir, basename + ".*"))
            if not img_candidates: continue

            src_img = img_candidates[0]
            ext = os.path.splitext(src_img)[1]
            stats["aug_imgs"] += 1

            for i in range(FLOWER_OVERSAMPLE_FACTOR):
                new_name = f"{basename}_aug_{i}"
                shutil.copy(src_img, os.path.join(train_img_dir, new_name + ext))
                shutil.copy(txt_path, os.path.join(train_lbl_dir, new_name + ".txt"))
                stats["copies"] += 1

    print(f"✔ Tiles aumentados: {stats['aug_imgs']} | Copias: {stats['copies']}")

def upload_dir_to_s3(local_dir, bucket, s3_prefix):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, relative_path).replace("\\","/")
            s3_client.upload_file(local_path, bucket, s3_path)
    print(f"✅ Subido {local_dir} a s3://{bucket}/{s3_prefix}")
# =========================================================
# PIPELINE PRINCIPAL (MIGRADO A SAGEMAKER)
# =========================================================

def prepare_and_train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_version = "dataset_v001"  # Ajusta según tu dataset
    s3_prefix_base = f"sagemaker-runs/yolo/{dataset_version}_{timestamp}"
    # 1. Limpieza de carpetas temporales
    if os.path.exists(LOCAL_TILED): shutil.rmtree(LOCAL_TILED)
    if os.path.exists(LOCAL_RUNS): shutil.rmtree(LOCAL_RUNS)

    for split in ["train", "val", "test"]:
        os.makedirs(f"{LOCAL_TILED}/images/{split}", exist_ok=True)
        os.makedirs(f"{LOCAL_TILED}/labels/{split}", exist_ok=True)

    # 2. Clasificar imágenes desde DATA_PATH (S3 montado por SageMaker)
    print(f"📂 Leyendo datos desde: {DATA_PATH}")
    raw_images = []
    for ext in ["*.jpg", "*.png", "*.JPG"]:
        raw_images.extend(glob.glob(os.path.join(DATA_PATH, "images", ext)))

    populated, empty = [], []
    for img in raw_images:
        name = os.path.basename(img).rsplit(".", 1)[0]
        lbl = os.path.join(DATA_PATH, "labels", name + ".txt")
        if os.path.exists(lbl) and open(lbl).read().strip():
            populated.append(img)
        else:
            empty.append(img)

    random.shuffle(populated)
    random.shuffle(empty)
    final_dataset = populated + empty

    n_total = len(final_dataset)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    split_map = {
        "train": final_dataset[:n_train],
        "val": final_dataset[n_train:n_train + n_val],
        "test": final_dataset[n_train + n_val:]
    }

    # 3. TILING
    print("\n🧩 Ejecutando tiling...")
    for split, imgs in split_map.items():
        for img_path in imgs:
            name = os.path.basename(img_path).rsplit(".", 1)[0]
            lbl_path = os.path.join(DATA_PATH, "labels", name + ".txt")
            lbl_path = lbl_path if os.path.exists(lbl_path) else None

            process_tiling(
                img_path=img_path,
                output_dir_img=f"{LOCAL_TILED}/images/{split}",
                output_dir_lbl=f"{LOCAL_TILED}/labels/{split}",
                lbl_path=lbl_path,
                filename_prefix=name,
            )

    # 4. BALANCEO Y REPORTES
    enforce_background_ratio_train(f"{LOCAL_TILED}/images/train", f"{LOCAL_TILED}/labels/train", TARGET_EMPTY_RATIO)
    
    print("\n--- ESTADÍSTICAS PRE OVERSAMPLING ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))
    
    apply_balancing(f"{LOCAL_TILED}/images/train", f"{LOCAL_TILED}/labels/train")

    print("\n--- ESTADÍSTICAS POST OVERSAMPLING ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))

    # 5. CONFIGURAR YAML PARA YOLO
    data_yaml = {
        "path": LOCAL_TILED,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "flor", 1: "arandano"},
    }
    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    # 6. ENTRENAMIENTO YOLOv8
    print("\n🚀 Entrenando YOLOv8 en SageMaker...")
    model = YOLO("yolov8n.pt") # SageMaker descargará los pesos si hay internet

    # Nota: Bajamos imgsz a 640 para evitar CUDA Out of Memory en instancias G4dn de AWS
    model.train(
        data="data.yaml",
        epochs=5, # Cambiando a 5 epocas para probar
        imgsz=640,
        batch=16,
        project=LOCAL_RUNS,
        name='yolo_aws',
        patience=20,
        mosaic=0.0,
        mixup=0.05,
        degrees=5,
        translate=0.05,
        scale=0.1,
        shear=1.0,
        perspective=0.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        multi_scale=True
    )


    # 7. EXPORTAR RESULTADOS A S3 (Vía SageMaker)
    run_dir = os.path.join(LOCAL_RUNS, "yolo_aws")
    
    # El modelo 'best.pt' se guarda en MODEL_OUTPUT para ser registrado
    best_model = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        shutil.copy(best_model, os.path.join(MODEL_OUTPUT, "model.pt"))
        shutil.copy(best_model, os.path.join(run_dir, "best.pt"))
        print(f"✅ Modelo copiado a {MODEL_OUTPUT}")
    
    # Subiendo a S3 artifacts
    upload_dir_to_s3(run_dir, S3_BUCKET, f"{s3_prefix_base}/runs")
    upload_dir_to_s3(MODEL_OUTPUT, S3_BUCKET, f"{s3_prefix_base}/model")

    # Manifest.json
    manifest = {
        "dataset_version": dataset_version,
        "timestamp": timestamp,
        "model_s3_path": f"s3://{S3_BUCKET}/{s3_prefix_base}/model/model.pt",
        "runs_s3_path": f"s3://{S3_BUCKET}/{s3_prefix_base}/runs"
    }

    manifest_path = os.path.join(LOCAL_RUNS, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    s3_client.upload_file(manifest_path, S3_BUCKET, f"{s3_prefix_base}/manifest.json")
    print(f"✅ Entrenamiento completo. Manifest subido a S3")

if __name__ == "__main__":
    prepare_and_train()