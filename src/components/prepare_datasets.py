import os
import json
import shutil
import random
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_PATH = Path("vehicle-damage-assesment/data/raw")
PROCESSED_PATH = Path("vehicle-damage-assesment/data/processed")

DATASETS_CONFIG = {
    "parts": {
        "root_folder": BASE_PATH / "car_parts_raw",
        "img_folder_name": "img",   
        "json_folder_name": "ann", 
        "output_folder": PROCESSED_PATH / "parts_dataset"
    },
    "damages": {
        "root_folder": BASE_PATH / "car_damages_raw",
        "img_folder_name": "img",   
        "json_folder_name": "ann", 
        "output_folder": PROCESSED_PATH / "damages_dataset"
    }
}

def normalize_polygons(points, img_width, img_height):
    normalized = []
    for p in points:
        x = min(max(p[0] / img_width, 0.0), 1.0)
        y = min(max(p[1] / img_height, 0.0), 1.0)
        normalized.extend([x, y])
    return normalized


def find_coresponding_img(json_file, img_folder):
    base_name = json_file.stem
    if (img_folder / base_name).exists():
        return img_folder / base_name
    
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = img_folder / (base_name + ext)
        if img_path.exists():
            return img_path
    return None


def clean_filename(stem):
    return stem.strip().replace(" ", "_").replace(".", "_").lower()




def process_dataset(name, config):
    print(f"Processing dataset: {name}")

    root_folder = config["root_folder"]
    img_folder = root_folder / config["img_folder_name"]
    json_folder = root_folder / config["json_folder_name"]
    output_folder = config["output_folder"]

    if not img_folder.exists() or not json_folder.exists():
        print(f"Image or JSON folder does not exist for dataset: {name}")
        print(f"Searching {config['img_folder_name']} and {config['json_folder_name']}")
        return

    json_files = list(json_folder.glob("*.json"))
    valid_pairs = []
    all_classes = set()

    for jf in tqdm(json_files, desc="Validating files"):
        img_file = find_coresponding_img(jf, img_folder)
        if img_file:
            valid_pairs.append((jf, img_file))
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                for obj in data.get("objects", []):
                    all_classes.add(obj["classTitle"])
            except:
                pass
        else:
            pass

    sorted_classes = sorted(list(all_classes))
    class_map = {name: idx for idx, name in enumerate(sorted_classes)}
    print(f"Found {len(valid_pairs)} valid image-annotation pairs.")
    print(f"Classes found: {sorted_classes}")

    train, test = train_test_split(valid_pairs, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.15, random_state=42)
    splits = {"train": train, "val": val, "test": test}


    for split_name, pairs in splits.items():
        save_img_folder = output_folder / split_name / "images"
        save_ann_folder = output_folder / split_name / "labels"
        save_img_folder.mkdir(parents=True, exist_ok=True)
        save_ann_folder.mkdir(parents=True, exist_ok=True)

        for json_path, img_path in tqdm(pairs, desc=f"Processing {split_name} data"):

            original_stem = img_path.stem
            safe_stem = clean_filename(original_stem)
            new_img_name = f"{safe_stem}{img_path.suffix}" 
            new_txt_name = f"{safe_stem}.txt"


            shutil.copy(img_path, save_img_folder / new_img_name)
            with open(json_path, 'r') as f:
                data = json.load(f)

            img_width = data.get("size", {}).get("width", 1)
            img_height = data.get("size", {}).get("height", 1)

            yolo_lines = []
            for obj in data.get("objects", []):
                class_name = obj["classTitle"]
                if class_name not in class_map:
                    continue

                class_id = class_map[class_name]
                points = obj.get("points", {}).get("exterior", [])

                if points:
                    normalized_points = normalize_polygons(points, img_width, img_height)
                    yolo_line = f"{class_id} " + " ".join(map(str, normalized_points))
                    yolo_lines.append(yolo_line)

            with open(save_ann_folder / new_txt_name, 'w') as f_out:
                f_out.write("\n".join(yolo_lines))

    
    yaml_content = {
        "path": str(output_folder.absolute()), # Ruta absoluta para evitar líos
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {v: k for k, v in class_map.items()} # Invertir mapa para que sea ID: Nombre
    }
    with open(output_folder / "dataset.yaml", 'w') as f_yaml:
        yaml.dump(yaml_content, f_yaml, sort_keys=False)


if __name__ == "__main__":
    # Ejecutamos el proceso para ambos datasets
    process_dataset("parts", DATASETS_CONFIG["parts"])
    
    # Comenta esta línea si aún no tienes lista la carpeta de daños
    process_dataset("damages", DATASETS_CONFIG["damages"])