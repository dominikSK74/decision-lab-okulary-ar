import glob

labels_paths = glob.glob("data2/dataset-train/labels/*.txt")

class_0 = 0
class_1 = 0
class_2 = 0

for path in labels_paths:
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            value = int(line.split(" ")[0])
            match(value):
                case 0:
                    class_0 += 1
                    break
                case 1:
                    class_1 += 1
                    break
                case 2:
                    class_2 += 1


print(f'Klasa 0: {class_0}')
print(f'Klasa 1: {class_1}')
print(f'Klasa 2: {class_2}')