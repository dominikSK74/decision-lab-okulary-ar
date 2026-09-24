import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2
import numpy as np
import tensorflow as tf

# ==== ŚCIEŻKI I ZMIENNE DO USTAWIENIA ====
model_path = "dl-model-alfa_21-09-2026_07-53-37-best.keras"
image_path = "data/datasets/dataset-test/images/0b3cee81b907aa24.jpg"

ANCHORS_ALL = [
    [0.05104393990357092, 0.06014114143304636], 
    [0.09490463494569387, 0.1952284804047026], 
    [0.2298436003698795, 0.13198910339736175], 
    [0.1897030639536864, 0.3876231734719182], 
    [0.4739456158183004, 0.2638858236071376], 
    [0.3562504250555475, 0.5864582651273655], 
    [0.8253495390792391, 0.3649001093644429], 
    [0.5510612091362792, 0.835999052685124], 
    [0.9090646502581039, 0.6362451307960698]
]
# ==========================================

ANCHORS_SMALL = ANCHORS_ALL[0:3]  # 80x80
ANCHORS_MEDIUM = ANCHORS_ALL[3:6] # 40x40
ANCHORS_LARGE = ANCHORS_ALL[6:9]  # 20x20

CONFIDENCE_THRESHOLD = 0.99
NMS_IOU_THRESHOLD = 0.0045    

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'loss_function': dummy_loss}
)

img = cv2.imread(image_path)
original_height = img.shape[0]
original_width = img.shape[1]
input_tensor = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

raw_predictions = model.predict(input_tensor)

outputs_to_check = [
    ("large", raw_predictions[0][0], 20, ANCHORS_LARGE),
    ("medium", raw_predictions[1][0], 40, ANCHORS_MEDIUM),
    ("small", raw_predictions[2][0], 80, ANCHORS_SMALL),
]

COLORS = {
    "large": (0, 255, 0),    
    "medium": (0, 0, 255),   
    "small": (255, 0, 0),    
}

boxes = []
scores = []
class_ids = []
sources = []

for name, predictions, grid_size, current_anchors in outputs_to_check:
    for y in range(grid_size):
        for x in range(grid_size):
            for anchor_idx in range(3):
                data = predictions[y, x, anchor_idx]

                xx = np.clip(data[4], -87.0, 87.0)
                objectness = sigmoid(xx)

                if objectness > CONFIDENCE_THRESHOLD:
                    class_id = int(np.argmax(data[5:8]))

                    raw_x = np.clip(data[0], -87.0, 87.0)
                    raw_y = np.clip(data[1], -87.0, 87.0)
                    raw_w = np.clip(data[2], -87.0, 87.0)
                    raw_h = np.clip(data[3], -87.0, 87.0)

                    sx = sigmoid(raw_x)
                    sy = sigmoid(raw_y)
                    sw = sigmoid(raw_w)
                    sh = sigmoid(raw_h)

                    anchor_w, anchor_h = current_anchors[anchor_idx]

                    x_center_norm = (x + sx) / grid_size
                    y_center_norm = (y + sy) / grid_size

                    width_norm = ((sw * 2.0) ** 2) * anchor_w
                    height_norm = ((sh * 2.0) ** 2) * anchor_h

                    x_center_px = x_center_norm * original_width
                    y_center_px = y_center_norm * original_height
                    width_px = width_norm * original_width
                    height_px = height_norm * original_height

                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)

                    boxes.append([x1, y1, int(width_px), int(height_px)])
                    scores.append(float(objectness))
                    class_ids.append(class_id)
                    sources.append(name)

print(f"Liczba kandydatów przed NMS: {len(boxes)}")

final_indices = []
class_ids_np = np.array(class_ids)

for cls in np.unique(class_ids_np):
    cls_indices = np.where(class_ids_np == cls)[0]
    cls_boxes = [boxes[i] for i in cls_indices]
    cls_scores = [scores[i] for i in cls_indices]

    if len(cls_boxes) == 0:
        continue

    keep = cv2.dnn.NMSBoxes(
        cls_boxes,
        cls_scores,
        score_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_IOU_THRESHOLD
    )

    if len(keep) > 0:
        keep = np.array(keep).flatten()
        for k in keep:
            final_indices.append(cls_indices[k])

print(f"Liczba boxów po NMS: {len(final_indices)}")

for i in final_indices:
    x1, y1, w, h = boxes[i]
    x2, y2 = x1 + w, y1 + h
    color = COLORS[sources[i]]
    label = f"{sources[i]} id{class_ids[i]} {scores[i]:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    print(f"[{sources[i]}] ID: {class_ids[i]} | box: ({x1},{y1})-({x2},{y2}) | Pewność: {scores[i]:.2f}")

cv2.imwrite("wynik_poprawiony.jpg", img)
print("Zapisano wynik_poprawiony.jpg")