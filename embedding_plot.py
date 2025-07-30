from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd
import typer
from typer import Option
from pathlib import Path

def main(
    foldername: str = Option(..., "-m", "--model", help="学習済みモデルのフォルダ"),
    test_data: str = Option(..., "-d", "--data", help="テストデータ")
) -> None:

    # データの読み込み（1回のみ）
    dataset_path = Path(test_data)
    dataset_parent, dataset_name = dataset_path.parent, dataset_path.name
    test_df = create_df(dataset_parent, dataset_name)
    print(f"テストデータを読み込みました: {len(test_df)} サンプル")
        
    # 予測器の読み込み（1回のみ）
    predictor = MultiModalPredictor.load(foldername)
    print(f"モデルを読み込みました: {foldername}")
    
    result = predictor.extract_embedding(test_df)

    X_pca = PCA(n_components=50).fit_transform(result)
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        learning_rate="auto",
        random_state=0
    ).fit_transform(X_pca)

    category = list(test_df["label"])

    df_plot = pd.DataFrame({'x':X_tsne[:,0], 'y':X_tsne[:,1], 'c':category})
    df_plot.plot(kind='scatter', x='x', y='y', c=category, colormap='gnuplot')
    plt.savefig("tsne_result.png")

if __name__ == "__main__":
    typer.run(main)
