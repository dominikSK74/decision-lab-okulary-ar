# pip install fiftyone
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four
from fiftyone import ViewField as F

CLASSES = ["person", "car", "cell phone"]
TARGET = {"cell phone": None, "car": 10_000, "person": 10_000}  # None = bierz wszystko
EXPORT_DIR = "yolo_dataset"
LABEL_FIELD = "ground_truth"

merged = fo.Dataset("coco_3_classes", overwrite=True)
selected = set()


for cls in ["cell phone", "car", "person"]:
    limit = TARGET[cls]
    ds = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "validation"],
        label_types=["detections"],
        classes=[cls],
        max_samples=None if limit is None else limit * 2,
        shuffle=True,
        seed=51,
        dataset_name=f"coco_{cls.replace(' ', '_')}",
        drop_existing_dataset=True,
    )

    view = ds.match(~F("filepath").is_in(list(selected)))
    if limit is not None:
        view = view.take(limit, seed=51)

    selected.update(view.values("filepath"))
    merged.add_samples(view)
    print(f"{cls}: dodano {len(view)} zdjęć")

merged = merged.filter_labels(LABEL_FIELD, F("label").is_in(CLASSES))
print("Łącznie zdjęć:", len(merged))


four.random_split(merged, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=51)

for split in ["train", "val", "test"]:
    merged.match_tags(split).export(
        export_dir=EXPORT_DIR,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=LABEL_FIELD,
        classes=CLASSES,
        split=split,
    )

print(merged.count_values(f"{LABEL_FIELD}.detections.label"))