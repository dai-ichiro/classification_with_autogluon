import warnings
warnings.filterwarnings('ignore')

from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor

def main():

    train_df = create_df("dataset", "train")

    predictor = MultiModalPredictor(
        problem_type="multiclass",
        label="label"
    )

    hyperparameters = {
        "model.timm_image.checkpoint_name": "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
        "optim.optim_type": "adamw"
    }

    save_path = "swin_large_patch4_window7_224_high_quality"

    predictor.fit(
        train_data = train_df,
        hyperparameters=hyperparameters,
        presets="high_quality",
        save_path=save_path
    )

    predictor.dump_model(save_path)

if __name__ == "__main__":
    main()