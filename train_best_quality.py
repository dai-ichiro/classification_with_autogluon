import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

def main():

    train_df = pd.read_pickle("image_dataset_train.pkl")

    predictor = MultiModalPredictor(
        problem_type="multiclass",
        label="label"
    )

    predictor.fit(
        train_data = train_df,
        presets="best_quality",
        save_path="simple_best_quality"
    )

if __name__ == "__main__":
    main()