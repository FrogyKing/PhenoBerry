import boto3
import os

# --- CONFIGURA TUS NOMBRES DE BUCKET AQUÍ ---
BUCKET_RAW = "phenoberry-dev-raw-038876987034" 
BUCKET_PROCESSED = "phenoberry-dev-processed-038876987034"
# --------------------------------------------

s3 = boto3.client('s3')

def get_all_objects(bucket, prefix=""):
    keys = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
    return keys

print("Iniciando auditoría...")

# 1. Obtener lista de RAW (Nombres de archivo base)
# Quitamos la extensión para comparar fácil
raw_files = get_all_objects(BUCKET_RAW)
raw_names = set()
for f in raw_files:
    # Ignorar carpetas o archivos ocultos
    if f.endswith('/') or 'training-dataset' in f: 
        continue
    # Extraer nombre sin extensión (ej: 'foto1.jpg' -> 'foto1')
    name = os.path.basename(f).rsplit('.', 1)[0]
    raw_names.add(name)

print(f"Archivos en RAW: {len(raw_names)}")

# 2. Obtener lista de PROCESSED (Nombres de carpetas)
processed_files = get_all_objects(BUCKET_PROCESSED)
processed_folders = set()
for f in processed_files:
    # La estructura es: nombre_foto/tile_01.jpg
    # Queremos 'nombre_foto'
    folder_name = f.split('/')[0]
    processed_folders.add(folder_name)

print(f"Carpetas en PROCESSED: {len(processed_folders)}")

# 3. Comparar (Resta de conjuntos)
missing = raw_names - processed_folders

if len(missing) == 0:
    print("\n✅ ¡Todo perfecto! No falta nada.")
else:
    print(f"\n❌ FALTAN {len(missing)} IMÁGENES:")
    print("-" * 30)
    for m in missing:
        print(f" - {m}")
    print("-" * 30)
    print("💡 Revisa CloudWatch Logs buscando estos nombres.")