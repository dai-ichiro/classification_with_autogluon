import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from multiprocessing import freeze_support
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_detailed_metrics(predictor, test_df, label_col='label'):
    """
    詳細な評価指標を計算・表示する関数
    
    Parameters:
    -----------
    predictor : MultiModalPredictor
        学習済みの予測器
    test_df : pd.DataFrame
        テストデータ
    label_col : str
        ラベル列名（デフォルト: 'label'）
    """
    
    print("=" * 60)
    print("詳細評価結果")
    print("=" * 60)
    
    # AutoGluonの標準評価
    print("1. AutoGluon標準評価:")
    score = predictor.evaluate(test_df, metrics=["accuracy"])
    print(f"   Accuracy: {score['accuracy']:.4f}")
    print()
    
    # 予測実行
    print("2. 予測実行中...")
    predictions = predictor.predict(test_df)
    
    # 確率予測も取得（可能な場合）
    try:
        prediction_probs = predictor.predict_proba(test_df)
        has_probs = True
    except:
        prediction_probs = None
        has_probs = False
        print("   注意: 確率予測が利用できません")
    
    # 真のラベル取得
    y_true = test_df[label_col]
    y_pred = predictions
    
    print("3. 詳細指標計算:")
    
    # 基本指標（weighted average）
    accuracy = accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"   Accuracy:           {accuracy:.4f}")
    print(f"   Precision (weighted): {precision_weighted:.4f}")
    print(f"   Recall (weighted):    {recall_weighted:.4f}")
    print(f"   F1-Score (weighted):  {f1_weighted:.4f}")
    print()
    
    # 各平均方法での指標
    print("4. 各平均方法での指標:")
    for avg_method in ['micro', 'macro', 'weighted']:
        prec = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
        f1_avg = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
        print(f"   {avg_method:8s}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1_avg:.4f}")
    print()
    
    # クラス別詳細レポート
    print("5. クラス別詳細レポート:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 混同行列
    print("6. 混同行列:")
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(y_true.unique())
    cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    print(cm_df)
    print()
    
    # ROC-AUC（確率予測がある場合）
    if has_probs:
        print("7. ROC-AUC:")
        try:
            if len(unique_labels) == 2:  # 二値分類
                roc_auc = roc_auc_score(y_true, prediction_probs.iloc[:, 1])
                print(f"   ROC-AUC: {roc_auc:.4f}")
            else:  # 多クラス分類
                roc_auc = roc_auc_score(y_true, prediction_probs, multi_class='ovr')
                print(f"   ROC-AUC (OvR): {roc_auc:.4f}")
        except Exception as e:
            print(f"   ROC-AUC計算エラー: {e}")
    else:
        print("7. ROC-AUC: 確率予測が利用できないため計算できません")
    
    print()
    
    # 結果の辞書として返す
    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }
    
    return results, y_true, y_pred, prediction_probs

def plot_confusion_matrix(y_true, y_pred, figsize=(8, 6)):
    """
    混同行列を可視化
    """
    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(y_true.unique())
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('混同行列')
    plt.ylabel('実際のラベル')
    plt.xlabel('予測ラベル')
    plt.tight_layout()
    plt.show()

def save_results_to_csv(results, filename="evaluation_results.csv"):
    """
    評価結果をCSVファイルに保存
    """
    # 基本指標のデータフレーム作成
    basic_metrics = {
        'Metric': ['Accuracy', 'Precision (weighted)', 'Recall (weighted)', 'F1-Score (weighted)'],
        'Score': [results['accuracy'], results['precision_weighted'], 
                 results['recall_weighted'], results['f1_weighted']]
    }
    
    df_basic = pd.DataFrame(basic_metrics)
    df_basic.to_csv(filename, index=False)
    print(f"評価結果を {filename} に保存しました")

def main():
    test_df = pd.read_pickle("test_df.pkl")
    
    predictor = MultiModalPredictor.load("step1_high_quality")
    #predictor = MultiModalPredictor.load("high_quality_hpo")
    
    # 詳細評価の実行
    results, y_true, y_pred, prediction_probs = evaluate_detailed_metrics(
        predictor, test_df, label_col='label'  # ラベル列名を適切に設定してください
    )
    
    # 結果をCSVに保存
    save_results_to_csv(results)
    
    # 混同行列の可視化（オプション）
    print("混同行列を可視化しますか？ (y/n): ", end="")
    response = input().lower()
    if response == 'y':
        plot_confusion_matrix(y_true, y_pred)
    
    print("\n評価完了！")

if __name__ == "__main__":
    freeze_support()  # Windows用
    main()