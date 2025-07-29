import shutil
from pathlib import Path
import random

folder_list = ["train", "val", "test"]
mushroom_list = ["Amanita muscaria", "Cerioporus squamosus", "Coprinus comatus"]
inbalance_mushroom = "Hypogymnia physodes"

for folder in folder_list:

    Path("inbalance_dataset", folder).mkdir(exist_ok=False, parents=True)

    for mushroom in mushroom_list:

        folder_path = Path("dataset", folder, mushroom)
        new_folder_path = Path("inbalance_dataset", folder, mushroom)

        shutil.copytree(folder_path, new_folder_path)

    
    if folder == "test":
        folder_path = Path("dataset", folder, inbalance_mushroom)
        new_folder_path = Path("inbalance_dataset", folder, inbalance_mushroom)
        shutil.copytree(folder_path, new_folder_path)

    else:
        file_path_list = list(Path("dataset", folder, inbalance_mushroom).glob("*"))

        selected_file_path_list = random.sample(file_path_list, len(file_path_list) // 10)

        save_folder = Path("inbalance_dataset", folder, inbalance_mushroom)

        save_folder.mkdir(exist_ok=False)

        for file_path in selected_file_path_list:

            shutil.copy2(file_path, save_folder / file_path.name)