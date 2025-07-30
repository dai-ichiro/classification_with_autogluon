import warnings
warnings.filterwarnings('ignore')

from df_from_folder import create_df
from auto_file_copy import FileWatcher
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
from pathlib import Path
import time

def main():
    save_path = "swin_large_trial1"
    clone_folder = f"{save_path}_all_checkpoint"

    # ディレクトリ作成
    Path(save_path).mkdir(exist_ok=True)
    
    # ファイル監視を開始
    watcher = FileWatcher(
        watch_dir=save_path,
        dest_dir=clone_folder,
        file_patterns=["epoch=*.ckpt", "last.ckpt", "*.yaml", "*.json", "*.pkl"]
    )
    watcher.start()
    
    try:
        # トレーニング実行
        run_training(save_path)
        
        # 最後のファイル処理を待つ
        print("⏳ Waiting for final file operations...")
        time.sleep(10)
        
    finally:
        # 監視を停止
        watcher.stop()
        print("🏁 All operations completed!")

def run_training(save_path):
    train_df = create_df("dataset", "train")
    val_df = create_df("dataset", "val")
    new_train = pd.concat([train_df, val_df], ignore_index=True)

    predictor = MultiModalPredictor(
        problem_type="multiclass",
        label="label"
    )

    hyperparameters = {
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        "optim.optim_type": "adamw",
        "env.per_gpu_batch_size": 64,
        "env.batch_size": 128
    }

    predictor.fit(
        train_data=new_train,
        hyperparameters=hyperparameters,
        presets="best_quality",
        save_path=save_path,
        clean_ckpts=False
    )

    predictor.dump_model(save_path)

if __name__ == "__main__":
    main()