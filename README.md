# AutoGluon Image Classification Project

## Overview

This project provides a set of tools to train, tune, and evaluate image classification models using [AutoGluon](https://auto.gluon.ai/stable/index.html). It is designed to work with a standard image folder structure and provides scripts to streamline the end-to-end machine learning workflow.

## Features

- **Automated Data Preparation**: Automatically creates DataFrames and label mappings from a simple folder structure.
- **Flexible Training**: Easily train models with different configurations using a single script.
- **Hyperparameter Optimization**: Integrated support for hyperparameter tuning with Ray Tune.
- **Comprehensive Evaluation**: Generate detailed evaluation reports, including accuracy, classification reports, and confusion matrices.
- **Command-Line Interface**: Scripts use `typer` for easy-to-use command-line arguments.

## Directory Structure

```
.
├── dataset/
│   ├── train/
│   │   ├── class_a/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── class_b/
│   │       └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── autogluon_train.py              # Main training script
├── autogluon_hpo_train.py          # Hyperparameter optimization script
├── evaluation_ag_result.py         # Model evaluation script
├── df_from_folder.py               # Data preparation utility
├── embedding_plot.py               # Script for visualizing embeddings
├── requirements_ubuntu.txt         # Main Python dependencies
├── tensorboard_requirements.txt    # Dependencies for TensorBoard
└── ...                             # Other utility scripts
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements_ubuntu.txt
    ```
    If you plan to use the embedding visualization script, install the TensorBoard requirements as well:
    ```bash
    pip install -r tensorboard_requirements.txt
    ```

## Usage

### 1. Prepare Your Dataset

Organize your images in the `dataset` directory following the structure described above. The `df_from_folder.py` script will automatically handle the creation of a `labels.txt` file, which maps your class folder names to integer labels.

### 2. Training a Model

The `autogluon_train.py` script trains a model using the settings defined within the script.

**To run the training:**
```bash
python autogluon_train.py
```

You can modify the `hyperparameters`, `save_path`, and other settings directly in the script to customize your training run.

### 3. Hyperparameter Optimization

The `autogluon_hpo_train.py` script performs hyperparameter optimization to find the best model configuration.

**To run HPO:**
```bash
python autogluon_hpo_train.py
```

The search space for hyperparameters and the number of trials can be adjusted within the script.

### 4. Evaluating a Model

The `evaluation_ag_result.py` script evaluates a trained model on a test set. It provides a simple evaluation, a confusion matrix visualization, or both.

**Command-Line Arguments:**
-   `-m`, `--model`: Path to the saved model directory.
-   `-d`, `--data`: Path to the test data directory (e.g., `dataset/test`).
-   `-t`, `--type`: The evaluation mode. Can be `simple`, `visualize`, or `both`.

**Example:**
To run a full evaluation (metrics and confusion matrix) on a model saved in `swin_large_patch4_window7_224_high_quality`, using the test data in `dataset/test`:
```bash
python evaluation_ag_result.py --model swin_large_patch4_window7_224_high_quality --data dataset/test --type both
```

This will print the evaluation metrics to the console and save the confusion matrix as `confusion_matrix.png`.
