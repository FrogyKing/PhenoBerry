import os
import cv2
import numpy as np

# Configuracion global que usabas
OVERLAP = 0.15
MIN_AREA_THRESHOLD = 0.002  # Si queda menos del 30% de la caja, la descarta

def load_yolo_boxes(lbl_path, img_w, img_h):
    """Lee las cajas YOLO y las convierte a coordenadas absolutas (píxeles)"""
    boxes = []
    if lbl_path and os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])

                    # Convertir coordenadas normalizadas a pixeles absolutos
                    x_center = xc * img_w
                    y_center = yc * img_h
                    width = w * img_w
                    height = h * img_h

                    # Bounding box absoluto
                    x1 = x_center - (width / 2)
                    x2 = x_center + (width / 2)
                    y1 = y_center - (height / 2)
                    y2 = y_center + (height / 2)

                    boxes.append({
                        'cls_id': cls_id,
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2
                    })
    return boxes

def process_tiling(img_path, output_dir_img, output_dir_lbl=None, lbl_path=None, filename_prefix="tile"):
    """
    Función Universal de Tiling.
    - Si output_dir_lbl y lbl_path tienen valor -> Genera tiles + etiquetas (TRAINING).
    - Si son None -> Solo genera tiles de imagen (INFERENCE).
    """
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error leyendo imagen: {img_path}")
        return []

    h, w = img.shape[:2]

    # Decidir cuadricula segun orientacion
    if h > w:
        ROWS, COLS = 4, 3
    else:
        ROWS, COLS = 3, 4

    # Cargar cajas solo si estamos en modo entrenamiento
    boxes = []
    if lbl_path:
        boxes = load_yolo_boxes(lbl_path, w, h)

    # Tamano de celdas
    base_w = w / COLS
    base_h = h / ROWS

    tile_w = int(base_w * (1 + OVERLAP))
    tile_h = int(base_h * (1 + OVERLAP))

    stride_x = int(base_w)
    stride_y = int(base_h)

    generated_files = []

    for r in range(ROWS):
        for c in range(COLS):

            x_start = int(min(c * stride_x, w - tile_w))
            y_start = int(min(r * stride_y, h - tile_h))

            # Asegurar límites
            x_start = max(0, x_start)
            y_start = max(0, y_start)

            x_end = min(x_start + tile_w, w)
            y_end = min(y_start + tile_h, h)

            crop = img[y_start:y_end, x_start:x_end]
            cur_h, cur_w = crop.shape[:2]

            # Nombre base del archivo
            base_name = f"{filename_prefix}_grid{ROWS}x{COLS}_r{r}c{c}"
            save_img_path = os.path.join(output_dir_img, base_name + '.jpg')
            
            # Guardar imagen
            cv2.imwrite(save_img_path, crop)
            generated_files.append(save_img_path)

            # --- Logica de Etiquetas (Solo si hay cajas y carpeta de salida) ---
            if output_dir_lbl and boxes:
                new_lines = []
                for box in boxes:
                    # Interseccion
                    inter_x1 = max(box['x1'], x_start)
                    inter_y1 = max(box['y1'], y_start)
                    inter_x2 = min(box['x2'], x_end)
                    inter_y2 = min(box['y2'], y_end)

                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        box_w_visible = inter_x2 - inter_x1
                        box_h_visible = inter_y2 - inter_y1
                        area_visible = box_w_visible * box_h_visible

                        box_area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
                        
                        # Filtro por area visible
                        if box_area > 0 and (area_visible / box_area >= MIN_AREA_THRESHOLD):
                            
                            # Coordenadas relativas al tile
                            new_x1 = inter_x1 - x_start
                            new_y1 = inter_y1 - y_start
                            new_x2 = inter_x2 - x_start
                            new_y2 = inter_y2 - y_start

                            new_w = new_x2 - new_x1
                            new_h = new_y2 - new_y1

                            # Normalizar a formato YOLO (0-1)
                            nxc = (new_x1 + new_w / 2) / cur_w
                            nyc = (new_y1 + new_h / 2) / cur_h
                            nwn = new_w / cur_w
                            nhn = new_h / cur_h

                            # Clip para seguridad
                            nxc = np.clip(nxc, 0, 1)
                            nyc = np.clip(nyc, 0, 1)
                            nwn = np.clip(nwn, 0, 1)
                            nhn = np.clip(nhn, 0, 1)

                            new_lines.append(f"{box['cls_id']} {nxc:.6f} {nyc:.6f} {nwn:.6f} {nhn:.6f}")

                # Guardar txt solo si hay etiquetas validas en este tile (o crear vacio si prefieres)
                # YOLO v8 maneja archivos vacíos como "background", es seguro crearlo.
                with open(os.path.join(output_dir_lbl, base_name + '.txt'), 'w') as f:
                    f.write('\n'.join(new_lines))
    
    return generated_files