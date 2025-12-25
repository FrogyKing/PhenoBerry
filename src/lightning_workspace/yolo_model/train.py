import os
import sys
import boto3
import shutil
import yaml
import glob
import random
from datetime import datetime
from ultralytics import YOLO

# Agrega la raíz del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.tiling import process_tiling

# ---------------- CONFIG ----------------
BUCKET_RAW = "phenoberry-dev-raw-038876987034"
BUCKET_ARTIFACTS = "phenoberry-dev-artifacts-038876987034"
S3_TRAIN_PREFIX = "training-dataset/"

LOCAL_RAW = "data_temp/raw"
LOCAL_TILED = "data_temp/tiled"
LOCAL_RUNS = "phenoberry_runs"

# Dataset control
TARGET_EMPTY_RATIO = 0.15
MIN_FLOWER_RATIO = 0.8
FLOWER_OVERSAMPLE_FACTOR = 10

FLOWER_CLASS_ID = 0
BLUEBERRY_CLASS_ID = 1

s3 = boto3.client("s3")


# =========================================================
# UTILIDADES
# =========================================================

def download_s3_folder(bucket, prefix, local_dir):
    print(f"⬇️ Descargando datos desde {bucket}/{prefix}")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)


def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    print(f"⬆️ Subiendo resultados a s3://{bucket}/{s3_prefix}")
    count = 0
    for root, _, files in os.walk(local_folder):
        for f in files:
            local_path = os.path.join(root, f)
            rel = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_prefix, rel).replace("\\", "/")
            s3.upload_file(local_path, bucket, s3_key)
            count += 1
    print(f"✔ Subidos {count} archivos")


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
# BALANCE BACKGROUND AUTOMÁTICO (SOLO TRAIN)
# =========================================================

def enforce_background_ratio_train(img_dir, lbl_dir, background_ratio=0.15, seed=42):
    random.seed(seed)

    label_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
    empty_tiles = []
    non_empty_tiles = []

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
    if total_tiles == 0:
        print("⚠️ No hay tiles en train")
        return

    target_empty = int(total_tiles * background_ratio)

    print(
        f"\n🎯 Control background (train)"
        f"\n   tiles totales : {total_tiles}"
        f"\n   vacíos actuales: {len(empty_tiles)}"
        f"\n   vacíos objetivo: {target_empty}"
    )

    if len(empty_tiles) <= target_empty:
        print("✅ Ya cumple el ratio de background")
        return

    random.shuffle(empty_tiles)
    to_remove = empty_tiles[target_empty:]

    removed = 0
    for lbl in to_remove:
        base = os.path.splitext(os.path.basename(lbl))[0]

        for img in glob.glob(os.path.join(img_dir, base + ".*")):
            os.remove(img)

        if os.path.exists(lbl):
            os.remove(lbl)

        removed += 1

    print(f"🧹 Eliminados {removed} tiles vacíos")


# =========================================================
# OVERSAMPLING DE FLORES
# =========================================================

def apply_balancing(train_img_dir, train_lbl_dir):
    print("\n Aplicando oversampling de flores...")
    txt_files = glob.glob(os.path.join(train_lbl_dir, "*.txt"))

    stats = {"aug_imgs": 0, "copies": 0}

    for txt_path in txt_files:
        try:
            with open(txt_path) as f:
                lines = f.readlines()
        except:
            continue

        total = 0
        flower_count = 0
        for l in lines:
            try:
                cls = int(l.split()[0])
                total += 1
                if cls == FLOWER_CLASS_ID:
                    flower_count += 1
            except:
                pass

        ratio = flower_count / total if total > 0 else 0.0

        if ratio >= MIN_FLOWER_RATIO:
            basename = os.path.basename(txt_path).replace(".txt", "")
            img_candidates = glob.glob(os.path.join(train_img_dir, basename + ".*"))
            if not img_candidates:
                continue

            src_img = img_candidates[0]
            ext = os.path.splitext(src_img)[1]

            stats["aug_imgs"] += 1

            for i in range(FLOWER_OVERSAMPLE_FACTOR):
                new_name = f"{basename}_aug_{i}"
                shutil.copy(src_img, os.path.join(train_img_dir, new_name + ext))
                shutil.copy(txt_path, os.path.join(train_lbl_dir, new_name + ".txt"))
                stats["copies"] += 1

    print(
        f"✔ Tiles aumentados: {stats['aug_imgs']} | "
        f"Copias generadas: {stats['copies']}"
    )


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def prepare_data_and_train():

    # Limpieza
    for d in [LOCAL_RAW, LOCAL_TILED, LOCAL_RUNS]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Descarga
    download_s3_folder(BUCKET_RAW, S3_TRAIN_PREFIX, LOCAL_RAW)

    # Crear carpetas
    for split in ["train", "val", "test"]:
        os.makedirs(f"{LOCAL_TILED}/images/{split}", exist_ok=True)
        os.makedirs(f"{LOCAL_TILED}/labels/{split}", exist_ok=True)

    # ----------------------------------------
    # Clasificar imágenes originales
    # ----------------------------------------
    raw_images = (
        glob.glob(f"{LOCAL_RAW}/images/*.jpg")
        + glob.glob(f"{LOCAL_RAW}/images/*.png")
        + glob.glob(f"{LOCAL_RAW}/images/*.JPG")
    )

    populated, empty = [], []

    for img in raw_images:
        name = os.path.basename(img).rsplit(".", 1)[0]
        lbl = os.path.join(LOCAL_RAW, "labels", name + ".txt")

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

    # ----------------------------------------
    # TILING
    # ----------------------------------------
    print("\n🧩 Ejecutando tiling...")
    for split, imgs in split_map.items():
        for img_path in imgs:
            name = os.path.basename(img_path).rsplit(".", 1)[0]
            lbl_path = os.path.join(LOCAL_RAW, "labels", name + ".txt")
            lbl_path = lbl_path if os.path.exists(lbl_path) else None

            process_tiling(
                img_path=img_path,
                output_dir_img=f"{LOCAL_TILED}/images/{split}",
                output_dir_lbl=f"{LOCAL_TILED}/labels/{split}",
                lbl_path=lbl_path,
                filename_prefix=name,
            )

    # ----------------------------------------
    # CONTROL BACKGROUND SOLO TRAIN
    # ----------------------------------------
    enforce_background_ratio_train(
        img_dir=f"{LOCAL_TILED}/images/train",
        lbl_dir=f"{LOCAL_TILED}/labels/train",
        background_ratio=TARGET_EMPTY_RATIO,
    )

    # ----------------------------------------
    # REPORTES PRE BALANCE
    # ----------------------------------------
    print("\n--- ESTADÍSTICAS PRE OVERSAMPLING ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))
    simple_report("Val  ", get_class_counts(f"{LOCAL_TILED}/labels/val"))

    # ----------------------------------------
    # OVERSAMPLING
    # ----------------------------------------
    apply_balancing(
        f"{LOCAL_TILED}/images/train",
        f"{LOCAL_TILED}/labels/train",
    )

    print("\n--- ESTADÍSTICAS POST OVERSAMPLING ---")
    simple_report("Train", get_class_counts(f"{LOCAL_TILED}/labels/train"))

    # ----------------------------------------
    # YAML
    # ----------------------------------------
    data_yaml = {
        "path": os.path.abspath(LOCAL_TILED),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "flor", 1: "arandano"},
    }

    with open("data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # ----------------------------------------
    # TRAIN
    # ----------------------------------------
    print("\n🚀 Entrenando YOLOv8...")
    model = YOLO("yolov8n.pt")

    run_name = "yolo_final_experiment"

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project=LOCAL_RUNS,
        name=run_name,
        patience=15,
        mosaic=0.0,
        mixup=0.2,
        degrees=10,
        translate=0.1,
        scale=0.1,
        shear=5.0,
        perspective=0.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
    )

    # ----------------------------------------
    # SUBIR RESULTADOS
    # ----------------------------------------
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    results_dir = os.path.join(LOCAL_RUNS, run_name)

    best_model = os.path.join(results_dir, "weights", "best.pt")
    if os.path.exists(best_model):
        s3.upload_file(
            best_model,
            BUCKET_ARTIFACTS,
            f"models/yolo/yolo_v{run_id}.pt",
        )

    upload_folder_to_s3(results_dir, BUCKET_ARTIFACTS, f"training_logs/run_{run_id}")


# =========================================================
if __name__ == "__main__":
    prepare_data_and_train()
