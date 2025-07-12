import json
import pandas as pd
import numpy as np
from collections import defaultdict
import statistics

class HPODataExtractor:
    """HPO実験結果からデータを抽出・分析するクラス"""
    
    def __init__(self, json_file_path):
        """
        Args:
            json_file_path (str): 実験結果JSONファイルのパス
        """
        self.json_file_path = json_file_path
        self.data = None
        self.trials_df = None
        
    def load_data(self):
        """JSONファイルを読み込み"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ ファイル読み込み成功: {self.json_file_path}")
            return True
        except Exception as e:
            print(f"✗ ファイル読み込み失敗: {e}")
            return False
    
    def extract_trials_data(self):
        """全試行のデータを抽出してDataFrameに変換"""
        if not self.data:
            print("まずload_data()を実行してください")
            return None
        
        trials = []
        
        for i, trial_data in enumerate(self.data['trial_data']):
            try:
                # 設定データと結果データを個別にパース
                trial_config = json.loads(trial_data[0])
                trial_results = json.loads(trial_data[1])
                
                # メトリクス名を確認（val_roc_aucまたはval_accuracyのどちらが使われているか）
                last_result = trial_results['last_result']
                metric_name = 'val_roc_auc' if 'val_roc_auc' in last_result else 'val_accuracy'
                
                # 必要なデータを抽出
                trial_info = {
                    'trial_id': trial_config['trial_id'],
                    'status': trial_config['status'],
                    
                    # ハイパーパラメータ
                    'learning_rate': trial_config['config']['optim.lr'],
                    'optimizer': trial_config['config']['optim.optim_type'],
                    'batch_size': trial_config['config']['env.batch_size'],
                    'max_epochs': trial_config['config']['optim.max_epochs'],
                    'model_name': trial_config['config']['model.timm_image.checkpoint_name'],
                    
                    # 結果（メトリクス名に応じて動的に設定）
                    'final_metric': last_result[metric_name],
                    'training_time_s': last_result['time_total_s'],
                    'final_iteration': last_result['training_iteration'],
                    'metric_name': metric_name,
                    
                    # メトリクス統計
                    'avg_metric': trial_results['metric_analysis'][metric_name]['avg'],
                    'max_metric': trial_results['metric_analysis'][metric_name]['max'],
                    'min_metric': trial_results['metric_analysis'][metric_name]['min'],
                    
                    # 時間統計
                    'avg_time_per_iter': trial_results['metric_analysis']['time_this_iter_s']['avg'],
                    'max_time_per_iter': trial_results['metric_analysis']['time_this_iter_s']['max'],
                    'min_time_per_iter': trial_results['metric_analysis']['time_this_iter_s']['min'],
                }
                
                trials.append(trial_info)
                
            except Exception as e:
                print(f"試行 {i} の処理でエラー: {e}")
                continue
        
        self.trials_df = pd.DataFrame(trials)
        print(f"✓ {len(trials)} 試行のデータを抽出しました")
        return self.trials_df
    
    def get_summary(self):
        """実験の要約統計を表示"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        # メトリクス名を取得
        metric_name = self.trials_df['metric_name'].iloc[0] if len(self.trials_df) > 0 else 'unknown'
        
        print("\n" + "="*60)
        print("HPO実験結果サマリー")
        print("="*60)
        print(f"試行数: {len(self.trials_df)}")
        print(f"評価メトリクス: {metric_name}")
        
        if 'start_time' in self.data.get('stats', {}):
            print(f"実験開始時間: {self.data['stats']['start_time']}")
        
        if '_total_time' in self.data.get('runner_data', {}):
            total_time = self.data['runner_data']['_total_time']
            print(f"総実行時間: {total_time:.2f}秒 ({total_time/60:.1f}分)")
        
        print(f"\n📊 {metric_name}スコア統計:")
        print(f"  最高: {self.trials_df['final_metric'].max():.4f}")
        print(f"  最低: {self.trials_df['final_metric'].min():.4f}")
        print(f"  平均: {self.trials_df['final_metric'].mean():.4f}")
        print(f"  標準偏差: {self.trials_df['final_metric'].std():.4f}")
        print(f"  中央値: {self.trials_df['final_metric'].median():.4f}")
        
        print("\n⏱️  学習時間統計:")
        print(f"  最長: {self.trials_df['training_time_s'].max():.2f}秒")
        print(f"  最短: {self.trials_df['training_time_s'].min():.2f}秒")
        print(f"  平均: {self.trials_df['training_time_s'].mean():.2f}秒")
        
        print("\n🔧 ハイパーパラメータ分布:")
        print(f"  オプティマイザ: {self.trials_df['optimizer'].value_counts().to_dict()}")
        print(f"  バッチサイズ: {sorted(self.trials_df['batch_size'].unique())}")
        print(f"  学習率範囲: {self.trials_df['learning_rate'].min():.6f} - {self.trials_df['learning_rate'].max():.6f}")
        print(f"  エポック数範囲: {self.trials_df['max_epochs'].min()} - {self.trials_df['max_epochs'].max()}")
        
        # モデル別統計
        print("\n🤖 モデル別統計:")
        for model in self.trials_df['model_name'].unique():
            model_data = self.trials_df[self.trials_df['model_name'] == model]
            print(f"  {model}:")
            print(f"    試行数: {len(model_data)}")
            print(f"    平均{metric_name}: {model_data['final_metric'].mean():.4f}")
            print(f"    最高{metric_name}: {model_data['final_metric'].max():.4f}")
    
    def get_best_trial(self):
        """最高性能の試行を表示"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        metric_name = self.trials_df['metric_name'].iloc[0]
        best_idx = self.trials_df['final_metric'].idxmax()
        best_trial = self.trials_df.iloc[best_idx]
        
        print("\n🏆 最高性能の試行:")
        print(f"Trial ID: {best_trial['trial_id']}")
        print(f"{metric_name}: {best_trial['final_metric']:.4f}")
        print(f"学習率: {best_trial['learning_rate']:.6f}")
        print(f"オプティマイザ: {best_trial['optimizer']}")
        print(f"バッチサイズ: {best_trial['batch_size']}")
        print(f"エポック数: {best_trial['max_epochs']}")
        print(f"モデル: {best_trial['model_name']}")
        print(f"学習時間: {best_trial['training_time_s']:.2f}秒")
        print(f"イテレーション数: {best_trial['final_iteration']}")
        
        return best_trial
    
    def get_worst_trial(self):
        """最低性能の試行を表示"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        metric_name = self.trials_df['metric_name'].iloc[0]
        worst_idx = self.trials_df['final_metric'].idxmin()
        worst_trial = self.trials_df.iloc[worst_idx]
        
        print("\n💔 最低性能の試行:")
        print(f"Trial ID: {worst_trial['trial_id']}")
        print(f"{metric_name}: {worst_trial['final_metric']:.4f}")
        print(f"学習率: {worst_trial['learning_rate']:.6f}")
        print(f"オプティマイザ: {worst_trial['optimizer']}")
        print(f"バッチサイズ: {worst_trial['batch_size']}")
        print(f"モデル: {worst_trial['model_name']}")
        
        return worst_trial
    
    def analyze_hyperparameters(self):
        """ハイパーパラメータの効果を分析"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        print("\n📈 ハイパーパラメータ効果分析:")
        
        # オプティマイザ別分析
        print("\n🔄 オプティマイザ別性能:")
        optimizer_stats = self.trials_df.groupby('optimizer')['final_metric'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(optimizer_stats)
        
        # バッチサイズ別分析
        print("\n📦 バッチサイズ別性能:")
        batch_stats = self.trials_df.groupby('batch_size')['final_metric'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(batch_stats)
        
        # モデル別分析
        print("\n🤖 モデル別性能:")
        model_stats = self.trials_df.groupby('model_name')['final_metric'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(model_stats)
        
        # 学習率と性能の相関
        correlation = self.trials_df['learning_rate'].corr(self.trials_df['final_metric'])
        print(f"\n📊 学習率とメトリクスの相関係数: {correlation:.4f}")
        
        # エポック数と性能の相関
        correlation_epochs = self.trials_df['max_epochs'].corr(self.trials_df['final_metric'])
        print(f"📊 エポック数とメトリクスの相関係数: {correlation_epochs:.4f}")
    
    def get_top_trials(self, n=5):
        """上位n個の試行を表示"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        metric_name = self.trials_df['metric_name'].iloc[0]
        top_trials = self.trials_df.nlargest(n, 'final_metric')
        
        print(f"\n🏅 上位{n}位の試行:")
        for i, (_, trial) in enumerate(top_trials.iterrows(), 1):
            print(f"\n{i}位: Trial {trial['trial_id']}")
            print(f"  {metric_name}: {trial['final_metric']:.4f}")
            print(f"  学習率: {trial['learning_rate']:.6f}")
            print(f"  オプティマイザ: {trial['optimizer']}")
            print(f"  バッチサイズ: {trial['batch_size']}")
            print(f"  モデル: {trial['model_name']}")
        
        return top_trials
    
    def analyze_convergence(self):
        """収束性を分析"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        print("\n📉 収束性分析:")
        
        # 早期終了した試行の分析
        completed_trials = self.trials_df[self.trials_df['status'] == 'TERMINATED']
        print(f"完了した試行数: {len(completed_trials)}")
        
        # イテレーション数の統計
        print(f"\nイテレーション数統計:")
        print(f"  平均: {self.trials_df['final_iteration'].mean():.1f}")
        print(f"  最大: {self.trials_df['final_iteration'].max()}")
        print(f"  最小: {self.trials_df['final_iteration'].min()}")
        
        # 性能向上の傾向
        trials_by_time = self.trials_df.sort_values('training_time_s')
        running_max = trials_by_time['final_metric'].expanding().max()
        improvement_rate = (running_max.iloc[-1] - running_max.iloc[0]) / len(trials_by_time)
        print(f"\n性能向上率（試行あたり）: {improvement_rate:.6f}")
    
    def save_to_csv(self, output_path="hpo_results.csv"):
        """結果をCSVファイルに保存"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        self.trials_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ 結果を {output_path} に保存しました")
    
    def get_all_trials(self):
        """全試行のDataFrameを表示"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        print("\n📋 全試行データ:")
        # 重要な列のみを表示
        display_cols = ['trial_id', 'final_metric', 'learning_rate', 'optimizer', 
                       'batch_size', 'model_name', 'training_time_s']
        print(self.trials_df[display_cols].to_string(index=False))
        
        return self.trials_df
    
    def create_insights_report(self):
        """洞察レポートを生成"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        metric_name = self.trials_df['metric_name'].iloc[0]
        
        print("\n" + "="*60)
        print("🔍 HPO実験洞察レポート")
        print("="*60)
        
        # 1. 最適なハイパーパラメータの組み合わせ
        best_trial = self.trials_df.loc[self.trials_df['final_metric'].idxmax()]
        print(f"\n✨ 推奨ハイパーパラメータ:")
        print(f"  学習率: {best_trial['learning_rate']:.6f}")
        print(f"  オプティマイザ: {best_trial['optimizer']}")
        print(f"  バッチサイズ: {best_trial['batch_size']}")
        print(f"  モデル: {best_trial['model_name']}")
        print(f"  → 期待性能: {best_trial['final_metric']:.4f} {metric_name}")
        
        # 2. パフォーマンス効率性
        efficiency = self.trials_df['final_metric'] / self.trials_df['training_time_s']
        best_efficiency_idx = efficiency.idxmax()
        efficient_trial = self.trials_df.loc[best_efficiency_idx]
        
        print(f"\n⚡ 最も効率的な設定:")
        print(f"  Trial ID: {efficient_trial['trial_id']}")
        print(f"  {metric_name}: {efficient_trial['final_metric']:.4f}")
        print(f"  学習時間: {efficient_trial['training_time_s']:.1f}秒")
        print(f"  効率性スコア: {efficiency[best_efficiency_idx]:.6f}")
        
        # 3. 注意すべき設定
        worst_trials = self.trials_df.nsmallest(3, 'final_metric')
        print(f"\n⚠️  避けるべき設定パターン:")
        common_bad_optimizers = worst_trials['optimizer'].mode()
        if len(common_bad_optimizers) > 0:
            print(f"  低性能でよく見られるオプティマイザ: {common_bad_optimizers[0]}")
        
        # 4. 安定性分析
        optimizer_stability = self.trials_df.groupby('optimizer')['final_metric'].std()
        most_stable = optimizer_stability.idxmin()
        print(f"\n🎯 最も安定したオプティマイザ: {most_stable} (標準偏差: {optimizer_stability[most_stable]:.4f})")
    
    def analyze_learning_rate_ranges(self):
        """学習率範囲別の性能分析"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        print("\n📈 学習率範囲別性能分析:")
        
        # 学習率を範囲別に分類
        lr_ranges = [
            (0, 0.001, "低学習率 (< 0.001)"),
            (0.001, 0.003, "中学習率 (0.001-0.003)"),
            (0.003, 0.01, "高学習率 (> 0.003)")
        ]
        
        for min_lr, max_lr, label in lr_ranges:
            range_trials = self.trials_df[
                (self.trials_df['learning_rate'] >= min_lr) & 
                (self.trials_df['learning_rate'] < max_lr)
            ]
            
            if len(range_trials) > 0:
                print(f"\n{label}:")
                print(f"  試行数: {len(range_trials)}")
                print(f"  平均メトリクス: {range_trials['final_metric'].mean():.4f}")
                print(f"  最高メトリクス: {range_trials['final_metric'].max():.4f}")
                print(f"  最低メトリクス: {range_trials['final_metric'].min():.4f}")
    
    def compare_model_performance(self):
        """モデル間の詳細な性能比較"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        print("\n🤖 モデル詳細比較:")
        
        for model in self.trials_df['model_name'].unique():
            model_trials = self.trials_df[self.trials_df['model_name'] == model]
            model_short = model.split('.')[0]  # モデル名を短縮
            
            print(f"\n{model_short}:")
            print(f"  試行数: {len(model_trials)}")
            print(f"  平均メトリクス: {model_trials['final_metric'].mean():.4f}")
            print(f"  最高メトリクス: {model_trials['final_metric'].max():.4f}")
            print(f"  平均学習時間: {model_trials['training_time_s'].mean():.2f}秒")
            print(f"  最適学習率: {model_trials.loc[model_trials['final_metric'].idxmax(), 'learning_rate']:.6f}")
    
    def generate_executive_summary(self):
        """エグゼクティブサマリーを生成"""
        if self.trials_df is None:
            print("まずextract_trials_data()を実行してください")
            return
        
        metric_name = self.trials_df['metric_name'].iloc[0]
        
        print("\n" + "="*60)
        print("📋 エグゼクティブサマリー")
        print("="*60)
        
        best_trial = self.trials_df.loc[self.trials_df['final_metric'].idxmax()]
        
        print(f"\n🎯 キーファインディング:")
        print(f"  • 最高性能: {best_trial['final_metric']:.4f} {metric_name}")
        print(f"  • 最適学習率: {best_trial['learning_rate']:.6f}")
        print(f"  • 推奨オプティマイザ: {best_trial['optimizer']}")
        print(f"  • 推奨バッチサイズ: {best_trial['batch_size']}")
        print(f"  • 最適モデル: {best_trial['model_name'].split('.')[0]}")
        
        # パフォーマンス分布
        high_perf_threshold = 0.9 if 'roc_auc' in metric_name else 0.9
        low_perf_threshold = 0.5 if 'roc_auc' in metric_name else 0.5
        
        high_perf_count = len(self.trials_df[self.trials_df['final_metric'] > high_perf_threshold])
        low_perf_count = len(self.trials_df[self.trials_df['final_metric'] < low_perf_threshold])
        
        print(f"\n📊 性能分布:")
        print(f"  • 高性能試行 ({metric_name} > {high_perf_threshold}): {high_perf_count}/{len(self.trials_df)} ({high_perf_count/len(self.trials_df)*100:.1f}%)")
        print(f"  • 低性能試行 ({metric_name} < {low_perf_threshold}): {low_perf_count}/{len(self.trials_df)} ({low_perf_count/len(self.trials_df)*100:.1f}%)")
        
        # 効率性
        efficiency = self.trials_df['final_metric'] / self.trials_df['training_time_s']
        best_efficiency_trial = self.trials_df.loc[efficiency.idxmax()]
        
        print(f"\n⚡ 効率性:")
        print(f"  • 最効率試行: {best_efficiency_trial['trial_id']}")
        print(f"  • 効率性スコア: {efficiency.max():.6f}")
        print(f"  • 平均学習時間: {self.trials_df['training_time_s'].mean():.2f}秒")
        
        print(f"\n💡 推奨アクション:")
        print(f"  1. 学習率 {best_trial['learning_rate']:.6f} を基準値として使用")
        print(f"  2. {best_trial['optimizer']} オプティマイザーを採用推奨")
        print(f"  3. バッチサイズ {best_trial['batch_size']} で本番実装")
        print(f"  4. {best_trial['model_name'].split('.')[0]} モデルを選択")
        print(f"  5. 学習時間効率を重視する場合は Trial {best_efficiency_trial['trial_id']} の設定を検討")


# 使用例
if __name__ == "__main__":
    import argparse
    import sys
    import os
    import glob
    
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='HPO実験結果を分析するツール')
    parser.add_argument('-m', '--model-dir', 
                       required=True,
                       help='実験結果が格納されているディレクトリのパス')
    parser.add_argument('--output', '-o', 
                       default='hpo_analysis_results.csv',
                       help='出力CSVファイルのパス (デフォルト: hpo_analysis_results.csv)')
    parser.add_argument('--top-n', '-n', 
                       type=int, default=5,
                       help='上位何位まで表示するか (デフォルト: 5)')
    parser.add_argument('--quick', '-q', 
                       action='store_true',
                       help='簡易分析モード（サマリーとベスト試行のみ表示）')
    
    args = parser.parse_args()
    
    # ディレクトリ存在チェック
    if not os.path.exists(args.model_dir):
        print(f"❌ エラー: ディレクトリ '{args.model_dir}' が見つかりません")
        sys.exit(1)
    
    if not os.path.isdir(args.model_dir):
        print(f"❌ エラー: '{args.model_dir}' はディレクトリではありません")
        sys.exit(1)
    
    # experiment_state-*.jsonファイルを検索
    json_pattern = os.path.join(args.model_dir, "experiment_state-*.json")
    json_files = glob.glob(json_pattern)
    
    if len(json_files) == 0:
        print(f"❌ エラー: '{args.model_dir}' 内にexperiment_state-*.jsonファイルが見つかりません")
        sys.exit(1)
    elif len(json_files) > 1:
        print(f"⚠️  警告: 複数のJSONファイルが見つかりました:")
        for f in json_files:
            print(f"  - {f}")
        print(f"最新のファイルを使用します: {max(json_files, key=os.path.getmtime)}")
        json_file = max(json_files, key=os.path.getmtime)
    else:
        json_file = json_files[0]
    
    print(f"📁 分析対象ディレクトリ: {args.model_dir}")
    print(f"📄 分析対象ファイル: {json_file}")
    print(f"📊 出力ファイル: {args.output}")
    print("-" * 50)
    
    # HPODataExtractorの実行
    extractor = HPODataExtractor(json_file)
    
    # データ読み込みと抽出
    if extractor.load_data():
        df = extractor.extract_trials_data()
        
        if df is None or len(df) == 0:
            print("❌ エラー: データの抽出に失敗しました")
            sys.exit(1)
        
        # 分析の実行
        if args.quick:
            # 簡易分析モード
            print("\n🚀 簡易分析モード")
            extractor.get_summary()
            extractor.get_best_trial()
        else:
            # 完全分析モード
            print("\n🔬 完全分析モード")
            extractor.get_summary()
            extractor.get_best_trial()
            extractor.get_worst_trial()
            extractor.analyze_hyperparameters()
            extractor.analyze_convergence()
            extractor.get_top_trials(args.top_n)
            extractor.analyze_learning_rate_ranges()
            extractor.compare_model_performance()
            extractor.create_insights_report()
            extractor.generate_executive_summary()
        
        # 結果をCSVに保存
        extractor.save_to_csv(args.output)
        
        print(f"\n✅ 分析完了！抽出されたデータは {len(df)} 行です。")
        print(f"📁 詳細データは {args.output} に保存されました。")
    else:
        print("❌ エラー: ファイルの読み込みに失敗しました")
        sys.exit(1)