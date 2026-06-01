import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Путь к распакованному экспорту из CVAT
source_dir = "unsorted_dataset_2"
# Путь к итоговой папке датасета
target_dir = "dataset_2"
EXTRACT_DIR = "diplom" 
# Создаем папки
os.makedirs(f"{target_dir}/images/train", exist_ok=True)
os.makedirs(f"{target_dir}/images/val", exist_ok=True)
os.makedirs(f"{target_dir}/labels/train", exist_ok=True)
os.makedirs(f"{target_dir}/labels/val", exist_ok=True)

# Получаем все файлы изображений (поддерживаются .jpg, .png и др.)
images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Разделяем на train/val
train_images, val_images = train_test_split(images, test_size=0.5, random_state=42)

def copy_files(image_list, split):
    for img in image_list:
        base = os.path.splitext(img)[0]
        # копируем изображение
        shutil.copy(os.path.join(source_dir, img), os.path.join(target_dir, f"images/{split}", img))
        # копируем соответствующий txt
        txt_file = base + ".txt"
        if os.path.exists(os.path.join(source_dir, txt_file)):
            shutil.copy(os.path.join(source_dir, txt_file), os.path.join(target_dir, f"labels/{split}", txt_file))
        else:
            print(f"Warning: {txt_file} not found for {img}")

copy_files(train_images, "train")
copy_files(val_images, "val")

classes_file = os.path.join(EXTRACT_DIR, "obj.names")
if os.path.exists(classes_file):
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    # Если файла нет, зададим вручную. Уточните порядок, в котором вы размечали!
    class_names = ['kgo_full', 'kgo_empty']  # измените, если нужно

# Убедимся, что классов ровно 2
print(f"Имена классов: {class_names}")

# Создаём data.yaml
yaml_content = f"""
# Dataset for trash container detection
train: ./images/train
val: ./images/val

nc: {len(class_names)}
names: {class_names}
"""

with open(f"{target_dir}/data_2.yaml", 'w') as f:
    f.write(yaml_content)

print("data.yaml создан.")