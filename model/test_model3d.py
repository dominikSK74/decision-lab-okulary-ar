import cv2
import numpy as np
import tensorflow as tf
 
# ==== ŚCIEŻKI DO USTAWIENIA ====
model_path = "dl-model-alfa-23-07-26-last-37epoch.keras"
image_path = "data/letterbox/dataset-test/images/val/88859419055de2c5.jpg"
# ================================
 
CONFIDENCE_THRESHOLD = 0.95
NMS_IOU_THRESHOLD = 0.001     # im niżej, tym agresywniej usuwa nachodzące boxy
 
def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)
 
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
    ("large", raw_predictions[0][0], 20),
    ("medium", raw_predictions[1][0], 40),
    ("small", raw_predictions[2][0], 80),
]
 
COLORS = {
    "large": (0, 255, 0),    # zielony
    "medium": (0, 0, 255),   # czerwony
    "small": (255, 0, 0),    # niebieski
}
 
boxes = []       # [x, y, w, h] w pikselach, format wymagany przez cv2.dnn.NMSBoxes
scores = []      # pewność (objectness)
class_ids = []   # id klasy
sources = []     # z której siatki pochodzi ("large"/"medium"/"small") - do kolorowania
 
for name, predictions, grid_size in outputs_to_check:
    for y in range(grid_size):
        for x in range(grid_size):
            for anchor_idx in range(3):
                data = predictions[y, x, anchor_idx]
 
                xx = np.clip(data[4], -87.0, 87.0)
                objectness = 1.0 / (1.0 + np.exp(-xx))
 
                if objectness > CONFIDENCE_THRESHOLD:
                    class_id = int(np.argmax(data[5:8]))
 
                    raw_x = data[0]
                    raw_y = data[1]
                    raw_w = data[2]
                    raw_h = data[3]
 
                    x_center_norm = (x + raw_x) / grid_size
                    y_center_norm = (y + raw_y) / grid_size
 
                    x_center_px = x_center_norm * original_width
                    y_center_px = y_center_norm * original_height
                    width_px = raw_w * original_width
                    height_px = raw_h * original_height
 
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
 
cv2.imshow("x", img)
cv2.waitKey(0)
cv2.destroyAllWindows()