import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from autogluon.multimodal import MultiModalPredictor

def main():

    train_df = pd.read_pickle("train.pkl")

    predictor = MultiModalPredictor(
        problem_type="multiclass",
        label="label"
    )

    hyperparameters = {
        #"optim.lr": tune.uniform(0.00005, 0.005),
        #"optim.optim_type": tune.choice(["adamw", "sgd"]),
        #"optim.max_epochs": tune.choice(["10", "20"]), 
        #"model.timm_image.checkpoint_name": "deit_base_patch16_224.fb_in1k",
        "model.timm_image.checkpoint_name": "deit3_base_patch16_224.fb_in1k",
        "optim.optim_type": "adamw"
    }

    hyperparameter_tune_kwargs = {
        "num_trials": 20,
    }
    predictor.fit(
        train_data = train_df,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        presets="best_quality_hpo",
        save_path="results/deit3_base_patch16_224_best_hpo"
    )

if __name__ == "__main__":
    main()