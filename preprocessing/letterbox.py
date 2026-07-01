import cv2
import glob
import os

def letterbox(image_path, label_path, image_destination, label_destination):
    image_filename = os.path.basename(image_path)
    label_filename = os.path.basename(label_path) 
    img = cv2.imread(image_path)

    original_height = img.shape[0]
    original_width = img.shape[1]

    new_height = 640
    new_width = 640
    scale = min(new_width / original_width, new_height / original_height)

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    missing_width = new_width - resized_width
    missing_height = new_height - resized_height

    padding_left = missing_width // 2
    padding_right = missing_width - padding_left
    padding_top = missing_height // 2
    padding_bottom = missing_height - padding_top

    resized_image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    letterboxed_image = cv2.copyMakeBorder(
        resized_image,
        padding_top, padding_bottom, padding_left, padding_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    cv2.imwrite(f'{image_destination}/{image_filename}', letterboxed_image)

    with open(label_path, 'r') as label_file:
        lines = label_file.readlines()

    new_lines = []
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

        #scaling
        x_center_px = x_center_px * scale
        y_center_px = y_center_px * scale
        width_px = width_px * scale
        height_px = height_px * scale

        #offset by padding
        x_center_px = x_center_px + padding_left
        y_center_px = y_center_px + padding_top

        #normalization
        x_center_new = x_center_px / new_width
        y_center_new = y_center_px / new_height
        width_new = width_px / new_width
        height_new = height_px / new_height

        if line == lines[len(lines) - 1]:
            result = f'{id} {x_center_new} {y_center_new} {width_new} {height_new}'
        else:
            result = f'{id} {x_center_new} {y_center_new} {width_new} {height_new}\n'
        new_lines.append(result)
    
    with open(f'{label_destination}/{label_filename}', 'w') as file:
        for line in new_lines:
            file.write(line)