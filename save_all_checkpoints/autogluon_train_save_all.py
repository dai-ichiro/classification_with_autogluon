import warnings
warnings.filterwarnings('ignore')

from df_from_folder import create_df
from filewatch import FileWatcherConfig, watch_files
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
from pathlib import Path
import time

save_path = "swin_large_trial3"
clone_folder = f"{save_path}_all_checkpoint"

Path(save_path).mkdir(exist_ok=False)
Path(clone_folder).mkdir(exist_ok=False)

config = FileWatcherConfig(
    watch_dir=save_path,
    dest_dir=clone_folder,
    recursive=False,
    file_patterns=["*.ckpt", "*.yaml", "*.json"],
    copy_files=True,
    daemon=True
)

@watch_files(config)
def file_monitor(src_path=None, dest_path=None):

    # デーモンモードで引数なしで呼ばれた場合は何もしない（監視のみ開始）
    if src_path is None or dest_path is None:
        return
    
    """ファイル監視のコールバック関数"""
    print(f"🔍 File watcher callback triggered: {src_path}")
    print(f"📁 File backup: {dest_path}")

def run_training():
    
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
        "env.per_gpu_batch_size" : 32,
        "env.batch_size": 64
    }

    predictor.fit(
        train_data = new_train,
        hyperparameters=hyperparameters,
        presets="best_quality",
        save_path=save_path,
        clean_ckpts=False
    )

    predictor.dump_model(save_path)
    print("✅ Training completed!")

if __name__ == "__main__":

    print("📂 Starting file monitoring in daemon mode...")
    
    # ファイル監視を開始（デーモンモードなので即座に戻る）
    file_monitor()
    
    print("📁 File monitoring started in background")
    
    # 訓練を実行
    run_training()
    
    # 訓練完了後も少し待機（最後のファイルをコピーするため）
    print("⏳ Waiting for final file operations...")
    time.sleep(10)
    print("🏁 All operations completed!")