import os
import random
import cv2
import matplotlib.pyplot as plt

# --- CONFIG ---
DATASET_DIR = "data/processed_tiles"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

# Colores BGR (OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),    # flor → verde
    1: (0, 0, 255),    # arándano → rojo
}

def draw_yolo_boxes(img, label_path):
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, xc, yc, bw, bh = map(float, parts)
            cls_id = int(cls_id)

            # YOLO → píxeles
            x_center = xc * w
            y_center = yc * h
            box_w = bw * w
            box_h = bh * h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    return img

def show_random_sample():
    images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not images:
        print("❌ No hay imágenes")
        return

    img_name = random.choice(images)
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error leyendo imagen")
        return

    img = draw_yolo_boxes(img, lbl_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(img_name)
    plt.show()

if __name__ == "__main__":
    show_random_sample()
