from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import pandas as pd

predictor = MultiModalPredictor.load("swinT_inbalance")

test_df = create_df("inbalance_dataset", "test")

result = predictor.extract_embedding(test_df)

X_pca = PCA(n_components=50).fit_transform(result)
X_tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    learning_rate="auto"
).fit_transform(X_pca)

category = list(test_df["label"])

df_plot = pd.DataFrame({'x':X_tsne[:,0], 'y':X_tsne[:,1], 'c':category})
df_plot.plot(kind='scatter', x='x', y='y', c=category, colormap='gnuplot')
plt.savefig("tsne_result.png")