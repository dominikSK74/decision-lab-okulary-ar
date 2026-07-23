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


img = cv2.imread("data/letterbox/dataset-train/images/val/0517714716908733.jpg")
original_height = img.shape[0]
original_width = img.shape[1]
input_tensor = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
predictions = model.predict(input_tensor)[0]

for y in range(20):
    for x in range(20):
        for anchor_idx in range(3):
            data = predictions[y, x, anchor_idx]
            
            # objectness = 1.0 / (1.0 + np.exp(-data[4]))
            xx = np.clip(data[4], -87.0, 87.0)
            # print(xx)
            objectness = 1.0 / (1.0 + np.exp(-xx))
            if objectness > 0.1:
                class_id = int(np.argmax(data[5:8]))
                
                raw_x = data[0]
                raw_y = data[1]
                raw_w = data[2]
                raw_h = data[3]
                
                print(f"ID: {class_id} | X: {raw_x:.4f} | Y: {raw_y:.4f} | W: {raw_w:.4f} | H: {raw_h:.4f} (Pewność: {objectness:.2f})")

                #denormalization
                x_center_px = raw_x * original_width
                y_center_px = raw_y * original_height
                width_px = raw_w * original_width
                height_px = raw_h * original_height

                #calc corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.imshow("x", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                # print("Nie wykryto")
                pass