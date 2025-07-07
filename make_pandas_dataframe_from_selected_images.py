import pandas as pd
import os

def create_dataframe_from_folders(base_path):
    """
    フォルダ構成からpandas DataFrameを作成する関数
    
    Args:
        base_path (str): 基本パス（各フォルダが含まれるディレクトリ）
    
    Returns:
        pd.DataFrame: image列とlabel列を持つDataFrame
    """
    
    # フォルダ名とラベルのマッピング
    folder_labels = {
        'mixed_small_folders': 0,
        'Amanita muscaria': 1,
        'Cerioporus squamosus': 2,
        'Coprinus comatus': 3,
        'Fomes fomentarius': 4,
        'Fomitopsis betulina': 5,
        'Fomitopsis pinicola': 6,
        'Hypogymnia physodes': 7,
        'Laetiporus sulphureus': 8,
        'Leccinum scabrum': 9,
        'Parmelia sulcata': 10,
        'Xanthoria parietina': 11
    }
    
    # データを格納するリスト
    data = []
    
    # 各フォルダを処理
    for folder_name, label in folder_labels.items():
        folder_path = os.path.join(base_path, folder_name)
        
        # フォルダが存在するかチェック
        if os.path.exists(folder_path):
            # 画像ファイルを検索（重複を避けるため、os.listdirを使用）
            image_files = []
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    # 画像拡張子をチェック（大文字小文字を区別しない）
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
                        image_files.append(os.path.join(folder_path, filename))
            
            # 各画像ファイルに対してデータを追加
            for image_path in image_files:
                # フルパスを取得
                full_path = os.path.abspath(image_path)
                data.append({
                    'image': full_path,
                    'label': label
                })
            
            print(f"フォルダ '{folder_name}' から {len(image_files)} 枚の画像を読み込みました（ラベル: {label}）")
        else:
            print(f"警告: フォルダ '{folder_path}' が見つかりません")
    
    # DataFrameを作成
    df = pd.DataFrame(data)
    
    # データをシャッフル（オプション）
    # df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def split_dataframe_by_labels(df, train_size=1000, test_size=500):
    """
    各ラベルごとにDataFrameを学習用とテスト用に分割する関数
    
    Args:
        df (pd.DataFrame): 分割するDataFrame
        train_size (int): 各ラベルの学習用データ数
        test_size (int): 各ラベルのテスト用データ数
    
    Returns:
        tuple: (train_df, test_df)
    """
    train_data = []
    test_data = []
    
    # 各ラベルごとに処理
    for label in df['label'].unique():
        # 該当ラベルのデータを取得
        label_data = df[df['label'] == label].copy()
        
        # データをシャッフル
        label_data = label_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 必要なデータ数をチェック
        total_needed = train_size + test_size
        if len(label_data) < total_needed:
            print(f"警告: ラベル {label} のデータが不足しています。必要: {total_needed}, 実際: {len(label_data)}")
            # 可能な限り分割
            actual_train_size = min(train_size, len(label_data))
            actual_test_size = min(test_size, len(label_data) - actual_train_size)
        else:
            actual_train_size = train_size
            actual_test_size = test_size
        
        # 学習用とテスト用に分割
        train_subset = label_data.iloc[:actual_train_size]
        test_subset = label_data.iloc[actual_train_size:actual_train_size + actual_test_size]
        
        train_data.append(train_subset)
        test_data.append(test_subset)
        
        print(f"ラベル {label}: 学習用 {len(train_subset)} 枚, テスト用 {len(test_subset)} 枚")
    
    # 結果を結合
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    # 再度シャッフル
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, test_df

# 使用例
if __name__ == "__main__":
    # 基本パスを設定（実際のパスに変更してください）
    base_path = "sampled_dataset"  # 実際のパスに変更
    
    # DataFrameを作成
    df = create_dataframe_from_folders(base_path)
    
    # 結果を表示
    print(f"\n作成されたDataFrame:")
    print(f"形状: {df.shape}")
    print(f"\n最初の5行:")
    print(df.head())
    print(f"\n最後の5行:")
    print(df.tail())
    
    # ラベルの分布を確認
    print(f"\nラベルの分布:")
    print(df['label'].value_counts().sort_index())
    
    # 学習用とテスト用に分割
    print(f"\n=== データ分割 ===")
    train_df, test_df = split_dataframe_by_labels(df, train_size=1000, test_size=500)
    
    print(f"\n学習用データ:")
    print(f"形状: {train_df.shape}")
    print(f"ラベル分布:")
    print(train_df['label'].value_counts().sort_index())
    
    print(f"\nテスト用データ:")
    print(f"形状: {test_df.shape}")
    print(f"ラベル分布:")
    print(test_df['label'].value_counts().sort_index())
    
    # Pickleファイルとして保存
    df.to_pickle('image_dataset_full.pkl')
    train_df.to_pickle('image_dataset_train.pkl')
    test_df.to_pickle('image_dataset_test.pkl')
    
    print("\n=== 保存完了 ===")
    print("全データ: 'image_dataset_full.pkl'")
    print("学習用データ: 'image_dataset_train.pkl'")
    print("テスト用データ: 'image_dataset_test.pkl'")
