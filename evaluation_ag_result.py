import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import typer
from typer import Option

def extract_values_from_yaml(yaml_file_path) -> dict | None:
    """
    YAMLファイルから指定された値を抽出する関数
    
    Args:
        yaml_file_path (str | Path): YAMLファイルのパス
        
    Returns:
        dict: 抽出された値の辞書
    """
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # 抽出したい値を取得
        extracted_values = {
            'checkpoint_name': data.get('model', {}).get('timm_image', {}).get('checkpoint_name'),
            'optim_type': data.get('optim', {}).get('optim_type'),
            'lr': data.get('optim', {}).get('lr'),
            'weight_decay': data.get('optim', {}).get('weight_decay'),
            'batch_size': data.get('env', {}).get('batch_size'),
            'per_gpu_batch_size': data.get('env', {}).get('per_gpu_batch_size')
        }
        
        return extracted_values
    
    except FileNotFoundError:
        print(f"エラー: ファイル '{yaml_file_path}' が見つかりません。")
        return None
    except yaml.YAMLError as e:
        print(f"YAML解析エラー: {e}")
        return None
    except Exception as e:
        print(f"予期しないエラー: {e}")
        return None

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6), save_path=None):
    """
    混同行列を可視化（修正版）
    
    Parameters:
    -----------
    y_true : array-like
        実際のラベル
    y_pred : array-like
        予測ラベル
    figsize : tuple
        図のサイズ
    save_path : str, optional
        保存先パス（指定すれば画像として保存）
    """
    
    # 混同行列の計算
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(y_true.unique())
    
    print(f"クラス数: {len(unique_labels)}")
    print(f"クラス: {unique_labels}")
    print(f"混同行列の形状: {cm.shape}")
    
    # matplotlib/seabornの設定
    plt.style.use('default')  # スタイルをリセット
    fig, ax = plt.subplots(figsize=figsize)
    
    # ヒートマップの作成
    sns.heatmap(cm, 
                annot=True,           # 数値を表示
                fmt='d',              # 整数フォーマット
                cmap='Blues',         # カラーマップ
                xticklabels=unique_labels, 
                yticklabels=unique_labels,
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    # タイトルとラベルの設定
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混同行列を {save_path} に保存しました")
    
    return cm

def alternative_visualization(y_true, y_pred):
    """
    代替的な可視化方法
    """
    print("\n" + "="*60)
    print("代替可視化：詳細分析")
    print("="*60)
    
    # 基本統計
    unique_labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"総サンプル数: {len(y_true)}")
    print(f"クラス数: {len(unique_labels)}")
    print(f"各クラスの分布:")
    
    for label in unique_labels:
        actual_count = np.sum(y_true == label)
        predicted_count = np.sum(y_pred == label)
        print(f"  {label}: 実際={actual_count}, 予測={predicted_count}")
    
    # 分類レポート
    print("\n詳細分類レポート:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))
    
    # 誤分類の詳細
    print("\n誤分類の詳細:")
    misclassified = y_true != y_pred
    if np.any(misclassified):
        print(f"誤分類数: {np.sum(misclassified)} / {len(y_true)} ({np.sum(misclassified)/len(y_true)*100:.2f}%)")
        
        # 誤分類のパターン
        misclass_df = pd.DataFrame({
            'actual': y_true[misclassified],
            'predicted': y_pred[misclassified]
        })
        
        print("\n誤分類パターンの頻度:")
        misclass_patterns = misclass_df.groupby(['actual', 'predicted']).size()
        for (actual, predicted), count in misclass_patterns.items():
            print(f"  {actual} → {predicted}: {count}回")
    else:
        print("誤分類はありません（完璧な分類）")

def evaluate_model(foldername: str, test_data: str, predictor=None, test_df=None, y_pred=None) -> None:
    """
    基本的なモデル評価を実行
    
    Args:
        foldername: 学習済みモデルのフォルダ
        test_data: テストデータ
        predictor: 既に読み込み済みの予測器（オプション）
        test_df: 既に読み込み済みのテストデータ（オプション）
        y_pred: 既に実行済みの予測結果（オプション）
    """
    # データの読み込み（まだ読み込まれていない場合のみ）
    if test_df is None:
        dataset_path = Path(test_data)
        dataset_parent, dataset_name = dataset_path.parent, dataset_path.name
        test_df = create_df(dataset_parent, dataset_name)

    # 予測器の読み込み（まだ読み込まれていない場合のみ）
    if predictor is None:
        predictor = MultiModalPredictor.load(foldername)

    # 予測実行（まだ実行されていない場合のみ）
    if y_pred is None:
        print("予測実行中...")
        y_pred = predictor.predict(test_df)

    # accuracyを手動計算
    from sklearn.metrics import accuracy_score
    y_true = test_df['label']
    accuracy = accuracy_score(y_true, y_pred)

    total_parameters = predictor.total_parameters
    trainable_parameters = predictor.trainable_parameters

    yaml_path = Path(foldername) / "config.yaml"
    result = extract_values_from_yaml(yaml_path)

    print(f"{foldername}: {{'accuracy': {accuracy:.6f}}}")
    if result:
        print(f"checkpoint_name: {result['checkpoint_name']}")
        print(f"total_parameters: {int(total_parameters/1000000)}MB")
        print(f"trainable_parameters: {int(trainable_parameters/1000000)}MB")
        print(f"optim_type: {result['optim_type']}")
        print(f"lr: {result['lr']}")
        print(f"weight_decay: {result['weight_decay']}")
        print(f"batch_size: {result['batch_size']}")
        print(f"per_gpu_batch_size: {result['per_gpu_batch_size']}")
    else:
        print("値の抽出に失敗しました。")

def visualize_confusion_matrix(foldername: str, test_data: str, predictor=None, test_df=None, y_pred=None) -> None:
    """
    混同行列を可視化
    
    Args:
        foldername: 学習済みモデルのフォルダ
        test_data: テストデータ
        predictor: 既に読み込み済みの予測器（オプション）
        test_df: 既に読み込み済みのテストデータ（オプション）
        y_pred: 既に実行済みの予測結果（オプション）
    """
    try:
        # データの読み込み（まだ読み込まれていない場合のみ）
        if test_df is None:
            dataset_path = Path(test_data)
            dataset_parent, dataset_name = dataset_path.parent, dataset_path.name
            test_df = create_df(dataset_parent, dataset_name)
        
        # 予測器の読み込み（まだ読み込まれていない場合のみ）
        if predictor is None:
            predictor = MultiModalPredictor.load(foldername)
        
        # 予測実行（まだ実行されていない場合のみ）
        if y_pred is None:
            print("予測実行中...")
            y_pred = predictor.predict(test_df)
        
        print("混同行列の可視化を開始...")
        
        y_true = test_df['label']
        
        # 混同行列の可視化（画像保存も含む）
        cm = plot_confusion_matrix(y_true, y_pred, 
                                 figsize=(10, 8), 
                                 save_path="confusion_matrix.png")
        
        # 代替可視化
        alternative_visualization(y_true, y_pred)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("サンプルデータで動作確認します...")
        
        # サンプルデータで動作確認
        np.random.seed(42)
        sample_true = np.random.choice(['A', 'B', 'C'], 100)
        sample_pred = sample_true.copy()
        # 10%の誤分類を意図的に作成
        error_indices = np.random.choice(100, 10, replace=False)
        sample_pred[error_indices] = np.random.choice(['A', 'B', 'C'], 10)
        
        print("サンプルデータでの可視化:")
        plot_confusion_matrix(sample_true, sample_pred)

def main(
    foldername: str = Option(..., "-m", "--model", help="学習済みモデルのフォルダ"),
    test_data: str = Option(..., "-d", "--data", help="テストデータ"),
    mode: str = Option("both", "-t", "--type", help="実行モード: 'simple', 'visualize', または 'both'")
) -> None:
    """
    統合評価スクリプト
    
    Args:
        foldername: 学習済みモデルのフォルダ
        test_data: テストデータ
        mode: 実行モード ('simple', 'visualize', または 'both')
    """
    
    if mode == "simple":
        print("=== モデル評価モード ===")
        evaluate_model(foldername, test_data)
    elif mode == "visualize":
        print("=== 混同行列可視化モード ===")
        visualize_confusion_matrix(foldername, test_data)
    elif mode == "both":
        print("=== 両方のモード実行（効率化版） ===")
        print("データとモデルの読み込み中...")
        
        # データの読み込み（1回のみ）
        dataset_path = Path(test_data)
        dataset_parent, dataset_name = dataset_path.parent, dataset_path.name
        test_df = create_df(dataset_parent, dataset_name)
        print(f"テストデータを読み込みました: {len(test_df)} サンプル")
        
        # 予測器の読み込み（1回のみ）
        predictor = MultiModalPredictor.load(foldername)
        print(f"モデルを読み込みました: {foldername}")
        
        # 予測実行（1回のみ）
        print("予測実行中...")
        y_pred = predictor.predict(test_df)
        print("予測が完了しました")
        
        print("\n" + "="*60)
        print("1. モデル評価を実行中...")
        print("="*60)
        evaluate_model(foldername, test_data, predictor, test_df, y_pred)
        
        print("\n" + "="*60)
        print("2. 混同行列可視化を実行中...")
        print("="*60)
        visualize_confusion_matrix(foldername, test_data, predictor, test_df, y_pred)

    else:
        print(f"エラー: 無効なモード '{mode}' が指定されました。")
        print("利用可能なモード: 'evaluate', 'visualize', 'both'")
        raise typer.Exit(1)

if __name__ == "__main__":
    typer.run(main)