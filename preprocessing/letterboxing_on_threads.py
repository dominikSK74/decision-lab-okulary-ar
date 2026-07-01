from concurrent.futures import ThreadPoolExecutor
import glob
from letterbox import letterbox

paths = glob.glob("data/dataset-validation/images/val/*")
output_image = 'data/letterbox/dataset-validation/images/val'
output_label = 'data/letterbox/dataset-validation/labels/val'

def letterbox_try(image_path, label_path, output_image_path, output_label_path):
    try:
        letterbox(image_path, label_path, output_image_path, output_label_path)
    except Exception as e:
        print(f'Exception on {image_path}: {e}')

with ThreadPoolExecutor(max_workers=8) as executor:
    for image_path in paths:
        label_path = image_path.replace("images", "labels")
        label_path = label_path.replace(".jpg", ".txt")
        executor.submit(letterbox_try, image_path, label_path, output_image, output_label)