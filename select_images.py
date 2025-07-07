import random
import shutil
from pathlib import Path

def sample_images_from_dataset(source_folder, output_folder, target_samples=1000):
    """
    画像データセットから条件に応じてサンプリングを行う
    
    Args:
        source_folder (str): 元のデータセットフォルダパス
        output_folder (str): 出力先フォルダパス
        target_samples (int): 抽出する画像数（デフォルト1000枚）
    """
    # 一般的な画像ファイルの拡張子
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}
    
    source_path = Path(source_folder)
    output_path = Path(output_folder)
    
    # 出力フォルダを作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 各フォルダの画像ファイルを収集
    folder_images = {}
    large_folders = []  # 1000枚以上のフォルダ
    small_folders = []  # 1000枚未満のフォルダ
    
    print("フォルダ情報を収集中...")
    
    for folder_path in source_path.iterdir():
        if folder_path.is_dir():
            # フォルダ内の画像ファイルを取得
            image_files = []
            for file_path in folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
            
            folder_images[folder_path.name] = image_files
            image_count = len(image_files)
            
            if image_count >= target_samples:
                large_folders.append(folder_path.name)
                print(f"大容量フォルダ: {folder_path.name} ({image_count}枚)")
            else:
                small_folders.append(folder_path.name)
                print(f"小容量フォルダ: {folder_path.name} ({image_count}枚)")
    
    print(f"\n大容量フォルダ数: {len(large_folders)}")
    print(f"小容量フォルダ数: {len(small_folders)}")
    print("-" * 50)
    
    # 1. 大容量フォルダから各1000枚をランダム抽出
    for folder_name in large_folders:
        print(f"{folder_name}から{target_samples}枚を抽出中...")
        
        # 出力フォルダを作成
        folder_output_path = output_path / folder_name
        folder_output_path.mkdir(exist_ok=True)
        
        # ランダムサンプリング
        selected_images = random.sample(folder_images[folder_name], target_samples)
        
        # ファイルをコピー
        for i, image_file in enumerate(selected_images):
            destination = folder_output_path / image_file.name
            shutil.copy2(image_file, destination)
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{target_samples}枚コピー完了")
        
        print(f"  {folder_name}: {target_samples}枚の抽出完了")
    
    # 2. 小容量フォルダから合計1000枚をランダム抽出
    if small_folders:
        print(f"\n小容量フォルダから合計{target_samples}枚を抽出中...")
        
        # 小容量フォルダ用の出力フォルダを作成
        mixed_output_path = output_path / "mixed_small_folders"
        mixed_output_path.mkdir(exist_ok=True)
        
        # 全ての小容量フォルダの画像を集める
        all_small_images = []
        for folder_name in small_folders:
            for image_file in folder_images[folder_name]:
                all_small_images.append((image_file, folder_name))
        
        total_small_images = len(all_small_images)
        print(f"  小容量フォルダ合計画像数: {total_small_images}枚")
        
        if total_small_images >= target_samples:
            # ランダムサンプリング
            selected_mixed_images = random.sample(all_small_images, target_samples)
            
            # ファイルをコピー
            for i, (image_file, original_folder) in enumerate(selected_mixed_images):
                # ファイル名に元のフォルダ名を付加
                new_filename = f"{original_folder}_{image_file.name}"
                destination = mixed_output_path / new_filename
                shutil.copy2(image_file, destination)
                
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{target_samples}枚コピー完了")
            
            print(f"  混合フォルダ: {target_samples}枚の抽出完了")
        else:
            print(f"  警告: 小容量フォルダの総画像数({total_small_images})が目標数({target_samples})より少ないため、全て抽出します")
            
            # 全ての画像をコピー
            for i, (image_file, original_folder) in enumerate(all_small_images):
                new_filename = f"{original_folder}_{image_file.name}"
                destination = mixed_output_path / new_filename
                shutil.copy2(image_file, destination)
                
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{total_small_images}枚コピー完了")
            
            print(f"  混合フォルダ: {total_small_images}枚の抽出完了")
    
    # 結果のサマリー
    print("\n" + "=" * 50)
    print("抽出結果サマリー:")
    print(f"大容量フォルダ数: {len(large_folders)}")
    print(f"各大容量フォルダから抽出: {target_samples}枚")
    print(f"小容量フォルダ数: {len(small_folders)}")
    if small_folders:
        mixed_count = min(target_samples, sum(len(folder_images[name]) for name in small_folders))
        print(f"混合フォルダから抽出: {mixed_count}枚")
    print(f"出力先: {output_folder}")

def main():
    # 設定
    source_folder = "merged_dataset"
    output_folder = "sampled_dataset"
    target_samples = 1500
    
    # シード値を設定（再現性のため）
    random.seed(42)
    
    print(f"画像サンプリング開始")
    print(f"元フォルダ: {source_folder}")
    print(f"出力フォルダ: {output_folder}")
    print(f"抽出枚数: {target_samples}枚")
    print("=" * 50)
    
    sample_images_from_dataset(source_folder, output_folder, target_samples)
    
    print("\n処理完了！")

if __name__ == "__main__":
    main()