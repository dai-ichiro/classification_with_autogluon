import warnings
warnings.filterwarnings("ignore")

from df_from_folder import create_df
from autogluon.multimodal import MultiModalPredictor
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
    
def main(
    foldername: str = Option(..., "-m", "--model", help="学習済みモデルのフォルダ"),
    test_data: str = Option(..., "-d", "--data", help="テストデータ")
    ) -> None:
    """
    Args:
        foldername: 学習済みモデルのフォルダ
        
        test_data: テストデータ
    """

    dataset_path = Path(test_data)
    dataset_parent, dataset_name= dataset_path.parent, dataset_path.name

    test_df = create_df(dataset_parent, dataset_name)

    predictor = MultiModalPredictor.load(foldername)
    total_parameters = predictor.total_parameters
    trainable_parameters = predictor.trainable_parameters

    score = predictor.evaluate(test_df, metrics=["accuracy"])

    yaml_path = Path(foldername) / "config.yaml"
    result = extract_values_from_yaml(yaml_path)

    print(f"{foldername}: {score}")
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

if __name__ == "__main__":
    typer.run(main)