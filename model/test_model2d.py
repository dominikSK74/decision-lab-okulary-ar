import cv2
import numpy as np
import tensorflow as tf

model_path = "dl-model-alfa-20-07-26-epoch20-17-77931-2D.keras"

def dummy_loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'loss_function': dummy_loss}
)

img = cv2.imread("data/letterbox/dataset-train/images/val/0a0a7e6f66eb3910.jpg")
original_height = img.shape[0]
original_width = img.shape[1]
input_tensor = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

raw_predictions = model.predict(input_tensor)

outputs_to_check = [
    ("large", raw_predictions[0][0], 20),
    ("small", raw_predictions[1][0], 40),
]

for name, predictions, grid_size in outputs_to_check:
    for y in range(grid_size):
        for x in range(grid_size):
            for anchor_idx in range(3):
                data = predictions[y, x, anchor_idx]

                xx = np.clip(data[4], -87.0, 87.0)
                objectness = 1.0 / (1.0 + np.exp(-xx))

                if objectness > 0.1:
                    class_id = int(np.argmax(data[5:8]))

                    raw_x = data[0]
                    raw_y = data[1]
                    raw_w = data[2]
                    raw_h = data[3]

                    x_center_norm = (x + raw_x) / grid_size
                    y_center_norm = (y + raw_y) / grid_size

                    print(f"[{name}] ID: {class_id} | X: {x_center_norm:.4f} | Y: {y_center_norm:.4f} "
                          f"| W: {raw_w:.4f} | H: {raw_h:.4f} (Pewność: {objectness:.2f})")

                    x_center_px = x_center_norm * original_width
                    y_center_px = y_center_norm * original_height
                    width_px = raw_w * original_width
                    height_px = raw_h * original_height

                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)

                    color = (0, 255, 0) if name == "large" else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

cv2.imshow("x", img)
cv2.waitKey(0)
cv2.destroyAllWindows()