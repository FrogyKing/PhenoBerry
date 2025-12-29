import os
import boto3
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw
from pathlib import Path

s3_client = boto3.client('s3')
# Asegúrate de que esta variable solo tenga el nombre del bucket: "mi-bucket-name"
OUTPUT_BUCKET = os.environ['OUTPUT_BUCKET']

# --- CARGA DEL MODELO (Solo una vez al inicio para optimizar) ---
def load_model():
    local_model_path = "/tmp/model.pt"
    # Parseo manual o usa una función auxiliar para s3://
    bucket = "phenoberry-dev-artifacts-038876987034"
    key = "model_legacy/yolo_nano_gpu.pt"
    
    if not os.path.exists(local_model_path):
        print(f"Descargando modelo desde S3: {key}...")
        s3_client.download_file(bucket, key, local_model_path)
    
    return YOLO(local_model_path)

model = load_model()

def run_inference(tile_s3_path, tile_id):
    # 1. Descarga del tile
    # tile_s3_path asumiendo formato s3://bucket/key
    path_parts = tile_s3_path.replace("s3://", "").split("/", 1)
    tile_bucket = path_parts[0]
    tile_key = path_parts[1]
    
    local_tile = f"/tmp/{os.path.basename(tile_key)}"
    s3_client.download_file(tile_bucket, tile_key, local_tile)
    
    # 2. Inferencia
    results = model(local_tile, conf=0.25) # Agregamos confianza mínima
    result = results[0]
    
    # 3. Guardar resultados en JSON (Formato YOLOv8+)
    json_results = []
    for box in result.boxes:
        json_results.append({
            "box": box.xyxy[0].tolist(),
            "conf": float(box.conf[0]),
            "cls": int(box.cls[0]),
            "name": result.names[int(box.cls[0])]
        })
    
    json_path = f"/tmp/{tile_id}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f)
    
    s3_client.upload_file(json_path, OUTPUT_BUCKET, f"results/{tile_id}.json")
    
    # 4. Crear overlay
    img = Image.open(local_tile)
    draw = ImageDraw.Draw(img)
    
    # Dibujamos las cajas detectadas
    for res in json_results:
        draw.rectangle(res["box"], outline="red", width=3)
        # Opcional: Escribir la etiqueta
        draw.text((res["box"][0], res["box"][1]), f"{res['name']} {res['conf']:.2f}")

    overlay_path = f"/tmp/{tile_id}_overlay.jpg"
    img.save(overlay_path)
    s3_client.upload_file(overlay_path, OUTPUT_BUCKET, f"overlays/{tile_id}_detected.jpg")
    
    print(f"✅ Inferencia completada para {tile_id}")