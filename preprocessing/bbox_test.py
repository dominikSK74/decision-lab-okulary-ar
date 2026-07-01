import cv2
import glob

def draw_bbox(image_path, label_path):
    img = cv2.imread(image_path)

    original_height = img.shape[0]
    original_width = img.shape[1]

    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    for line in lines:
        values = line.split(' ')
        id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])
            
        #denormalization
        x_center_px = x_center * original_width
        y_center_px = y_center * original_height
        width_px = width * original_width
        height_px = height * original_height

        #calc corners
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow("x", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


for image_path in glob.glob("preprocessing/test_data/images/*"):
    label_path = image_path.replace("images", "labels")
    label_path = label_path.replace(".jpg", ".txt")
    draw_bbox(image_path, label_path)

for image_path in glob.glob("testdest/i/*"):
    label_path = image_path.replace("i", "l")
    label_path = label_path.replace(".jpg", ".txt")
    draw_bbox(image_path, label_path)