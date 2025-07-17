import warnings
warnings.filterwarnings('ignore')

from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor
from ray import tune

def main():

    train_df = create_df("dataset", "train")

    predictor = MultiModalPredictor(
        problem_type="multiclass",
        label="label"
    )

    hyperparameters = {
        "optim.lr": tune.uniform(0.00005, 0.005),
        "optim.max_epochs": tune.choice(["10", "20"]), 
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        "optim.optim_type": "adamw"
    }

    hyperparameter_tune_kwargs = {
        "num_trials": 20,
    }

    save_path = "swin_large_patch4_window7_224_high_quality_hpo"

    predictor.fit(
        train_data = train_df,
        hyperparameters=hyperparameters,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        presets="high_quality_hpo",
        save_path=save_path
    )

    predictor.dump_model(save_path)

if __name__ == "__main__":
    main()