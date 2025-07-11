import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import typer
from typer import Option

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
    ax.set_title('混同行列 (Confusion Matrix)', fontsize=14, pad=20)
    ax.set_ylabel('実際のラベル (True Label)', fontsize=12)
    ax.set_xlabel('予測ラベル (Predicted Label)', fontsize=12)
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 保存または表示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混同行列を {save_path} に保存しました")
    
    # 表示を試行
    try:
        plt.show()
    except Exception as e:
        print(f"表示エラー: {e}")
        print("代替として、混同行列をテキストで表示します：")
        
        # テキスト形式で表示
        print("\n" + "="*50)
        print("混同行列（テキスト形式）")
        print("="*50)
        
        # pandas DataFrameとして表示
        cm_df = pd.DataFrame(cm, 
                           index=[f"実際_{label}" for label in unique_labels],
                           columns=[f"予測_{label}" for label in unique_labels])
        print(cm_df)
        
        # 正解率の計算
        diagonal_sum = np.trace(cm)
        total_sum = np.sum(cm)
        accuracy = diagonal_sum / total_sum
        print(f"\n全体の正解率: {accuracy:.4f} ({diagonal_sum}/{total_sum})")
        
        # クラス別の正解率
        print("\nクラス別の正解率:")
        for i, label in enumerate(unique_labels):
            class_correct = cm[i, i]
            class_total = np.sum(cm[i, :])
            class_accuracy = class_correct / class_total if class_total > 0 else 0
            print(f"  {label}: {class_accuracy:.4f} ({class_correct}/{class_total})")
    
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
    print(classification_report(y_true, y_pred, zero_division=0))
    
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

# 使用例
def main_visualization(
    foldername: str = Option(..., "-m", "--model", help="学習済みモデルのフォルダ"),
    test_data: str = Option(..., "-d", "--data", help="テストデータ")
    ):
    """
    メイン関数（可視化専用）

    Args:
        foldername: 学習済みモデルのフォルダ
        
        test_data: テストデータ
    """
    try:
        # テストデータの読み込み
        test_df = pd.read_pickle(test_data)
        
        # 予測器の読み込み
        from autogluon.multimodal import MultiModalPredictor
        predictor = MultiModalPredictor.load(foldername)
        
        # 予測実行
        print("予測実行中...")
        y_true = test_df['label']
        y_pred = predictor.predict(test_df)
        
        print("混同行列の可視化を開始...")
        
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

if __name__ == "__main__":
    typer.run(main_visualization)