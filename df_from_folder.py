from pathlib import Path
import pandas as pd
from pprint import pprint

def create_df(database, folder):
    """
    フォルダ構成からpandas DataFrameを作成する関数
    
    Args:
        database (str): 基本パス（各フォルダが含まれるディレクトリ）

        folder (str): train, val, or test
    
    Returns:
        pd.DataFrame: image列とlabel列を持つDataFrame
    """
    
    dir_path = Path(database, folder)

    label_file = Path(database, "labels.txt")

    if label_file.exists():
        print(f"\n{label_file.as_posix()}をもとにファルダ名とラベルのマッピングを行います")

        folder_labels = {}

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(":", 1)
                    folder_labels[key.strip()] = int(value.strip())
    else:
        print("\n新規にファルダ名とラベルのマッピングを行います")
        print(f"結果は{label_file.as_posix()}に保存しました")

        folders = [p.name for p in dir_path.iterdir() if p.is_dir()]

        # フォルダ名とラベルのマッピング
        folder_labels = {name: idx for idx, name in enumerate(folders)}
        
        with open(label_file, "w", encoding="utf-8") as f:
            for key, value in folder_labels.items():
                f.write(f"{key}: {value}\n")

    print()
    pprint(folder_labels, width=1, sort_dicts=False)

    # データを格納するリスト
    data = []
    
    # 各フォルダを処理
    for folder_name, label in folder_labels.items():
        folder_path = dir_path / folder_name
        
        image_files = folder_path.iterdir()
            
        for image_path in image_files:
            # フルパスを取得
            full_path = image_path.resolve()

            data.append({
                'image': full_path.as_posix(),
                'label': label
            })
                
    # DataFrameを作成
    df = pd.DataFrame(data)
   
    # 結果を表示
    print(f"\n{dir_path.as_posix()}から作成されたDataFrame")
    print(df)

    # ラベルの分布を確認
    print("\nラベルの分布:")
    print(df['label'].value_counts().sort_index())

    return df