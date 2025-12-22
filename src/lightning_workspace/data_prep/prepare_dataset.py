import os
import glob
import shutil
from tqdm import tqdm
# Importamos la función desde common
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.common.tiling import process_tiling

# --- CONFIGURACION ---
RAW_DIR = "data/raw_originals"  # Donde pusiste tus fotos gigantes
OUTPUT_DIR = "data/processed_tiles" # Donde saldrán los recortes

def run_preparation():
    # 1. Limpiar y crear directorios
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    out_images = os.path.join(OUTPUT_DIR, 'images')
    out_labels = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 2. Copiar classes.txt
    classes_src = os.path.join(RAW_DIR, 'classes.txt')
    if os.path.exists(classes_src):
        shutil.copy(classes_src, os.path.join(OUTPUT_DIR, 'classes.txt'))

    # 3. Buscar imagenes
    img_extensions = ['*.jpg', '*.JPG', '*.png']
    images = []
    for ext in img_extensions:
        images.extend(glob.glob(os.path.join(RAW_DIR, 'images', ext)))

    print(f"Procesando {len(images)} imágenes desde {RAW_DIR}...")

    # 4. Loop de procesamiento
    for img_path in tqdm(images):
        filename = os.path.basename(img_path).rsplit('.', 1)[0]
        
        # Buscar su txt correspondiente
        lbl_path = os.path.join(RAW_DIR, 'labels', filename + '.txt')
        
        if not os.path.exists(lbl_path):
            print(f"Advertencia: No hay label para {filename}, se salta o se procesa como vacío.")
            # Si quieres procesarlo igual sin cajas, pasa lbl_path=None
            continue 

        # LLAMAR A LA FUNCION COMUN
        process_tiling(
            img_path=img_path,
            output_dir_img=out_images,
            output_dir_lbl=out_labels,
            lbl_path=lbl_path,
            filename_prefix=filename
        )

    print(f"Dataset listo en: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_preparation()