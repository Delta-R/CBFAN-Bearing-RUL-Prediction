# CBFAN: Cross-Dimensional Dual-Temporal Attention Fusion Network for Bearing RUL Prediction

> Bearing remaining useful-life prediction method based on cross-dimensional dual-temporal attention fusion (CBFAN)

This repository contains the experimental code and data for the paper **CBFAN (Cross-Dimensional Dual-Temporal Attention Fusion Network)**, which focuses on remaining useful life (RUL) prediction of rolling bearings and related comparison / ablation experiments.

---

## 🔍 Overview

CBFAN targets bearing RUL prediction by jointly exploiting time-domain, frequency-domain and multi-scale statistical features. It builds a cross-dimensional, dual-temporal attention fusion network to improve the accuracy and stability of RUL estimation.

The figure below shows the overall framework of CBFAN. Starting from raw vibration signals, the data passes through a global cross-dimensional statistical attention module, a multi-scale dual-temporal attention fusion module, and a multi-head feature-fusion Transformer encoder, and finally outputs the bearing RUL prediction curve.

<img src=structure.png  width=70%  />

This repository includes:

- 🧠 Training and evaluation code for the main CBFAN model, together with example model checkpoints.
- 📊 Comparison experiments against baselines such as Transformer‑BiLSTM and CNN‑Transformer.
- 🗂️ Dataset preparation and splitting scripts for different working conditions (e.g., Bearing1_1 ~ Bearing1_7).
- 📈 Ablation studies and various visualization / plotting scripts (comparison experiments, ablations, RUL curves, etc.).

> Note: This README is written based only on file / folder names and the paper abstract. It does not dig into implementation details, but aims to help you understand the overall structure without reading all the source code.

---

## ⚙️ Environment

We recommend **Python 3.8+**. Core dependencies are listed in the root `requirements.txt`, for example:

- Deep learning framework (e.g., PyTorch)
- Data processing: NumPy, Pandas
- Visualization: Matplotlib

You can create a virtual environment and install dependencies as follows (Windows example):

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📁 Repository Structure (by module)

The following descriptions are based on file / folder names, to help you quickly locate code and data.

### Root

- `requirements.txt`: Python dependency list.
- `setup.py`: Installation / packaging configuration.
- `utils.py`: Common utility functions, such as data loading, metric computation and logging.
- `__init__.py`: Marks the root directory as a Python package.
- `paper_content.png`: Screenshot of the paper abstract or a related illustration.

### 1. `RemainingLifePredictionModel/`

Main CBFAN model and related experiments.

- `CBFAN-model/`
  - `CBFAN.ipynb`: Jupyter Notebook for building, training and evaluating the CBFAN model.
  - `best_model_transformer_bilstm.pt`: Example / best checkpoint, possibly for a Transformer‑BiLSTM submodule or baseline.
- `DataVisualization.ipynb`: Dataset construction and visualization (feature extraction, sample splitting, statistics, etc.).
- `plot_compare/`
  - `cnn_transformer_origin/`: Results and visualizations for the original CNN‑Transformer model.
  - `cnn_transformer_pre/`: Results and visualizations for a preprocessed / improved CNN‑Transformer variant.

### 2. `ComparisonExperiments/`

Comparison experiments with other models.

- `Transformer-BiLSTM-serial-model/`
  - `model/`
    - `Transformer-BiLSTM.ipynb`: Transformer‑BiLSTM serial architecture for RUL prediction.
    - `best_model_transformer_bilstm.pt`: Best checkpoint for this model.
  - `bearing_1-7/`
    - `Transformer-BiLSTM.ipynb`: Experiments for a specific bearing condition (e.g., Bearing1_7).
    - `best_model_transformer_bilstm.pt`: Checkpoint trained on that condition.
  - `dataresult/`
    - `train_set/`, `train_label/`, `test_set/`, `test_label/`: Data splits and labels for training and evaluation.
  - `plot_compare/`: Visualizations of the Transformer‑BiLSTM serial model results (if present).

### 3. `DatasetPreprocess/`

Preprocessing for raw bearing data.

- `data_preprocess.py`: Main preprocessing script, typically including:
  - Cropping segments from raw vibration or monitoring signals.
  - Extracting time-domain / frequency-domain features.
  - Generating sliding-window samples with RUL labels.
- `Bearing1_2_features_df/`: Feature data for the Bearing1_2 condition.
- `FUll_Bearing1_3_all_data/`: Full raw data for Bearing1_3.
- `FUll_Bearing1_3_features_df/`: Extracted feature dataset for Bearing1_3.

These files are commonly used as data sources for CBFAN and the baseline models.

### 4. `AblationExperiments/`

Ablation studies on different modules, such as cross-dimensional attention, dual-temporal attention and the multi-head feature-fusion Transformer encoder.

- `cnn-transform-bfm/`
  - `CNN-Transformer-model.ipynb`: CNN‑Transformer implementation and experiments.
  - `best_model_cnn_transformer.pt`: Best checkpoint for the CNN‑Transformer model.
  - `dataresult/`
    - `samples_data_Bearing1_1.csv`, `samples_data_Bearing1_2.csv`, `samples_data_FUll_Bearing1_3.csv`: Sample datasets for several bearing conditions.
    - `scaler/`: Stored scalers or normalizers (e.g., `StandardScaler`).
    - `train_set/`, `train_label/`, `test_set/`, `test_label/`: Dataset splits used in the ablation experiments.
  - `bearing_1-4/` ~ `bearing_1-7/`
    - Notebooks (`CNN-Transformer-model.ipynb`) and best checkpoints `best_model_cnn_transformer.pt` for each condition.
  - `original/`
    - `30epoch/`, `50epoch/`: Original CNN‑Transformer experiments with different training epochs.
      - `CNN-Transformer-model.ipynb`
      - `best_model_cnn_transformer.pt`
  - `plot_compare/`: Reserved or empty directory for additional plots.
- `plot_compare/`
  - `cnn_transformer_origin/`, `cnn_transformer_pre/`: Visualizations for different CNN‑Transformer variants used in the ablation studies.

### 5. `PlotCompare/`

Centralized plotting code and results for both comparison and ablation experiments.

- `ComparisonExperiments/`
  - `cnn_origin/`, `cnn_pre/`, `cnn_lstm_origin/`, `cnn_lstm_pre/`, `cnn_transformer_origin/`, `cnn_transformer_pre/`, `gru_origin/`, `gru_pre/`, `transformer_origin/`, `transformer_pre/`, `transformer_bilstm_serial_pre/`:
    - Visualization outputs (e.g., RUL curves, error distributions) for various architectures and settings.
  - `ComparisonPlot.py`: Python script that generates comparison plots.
  - `ComparisonPlot.ipynb`: Notebook version of the comparison plotting workflow.
  - `ComparisonPlot_backup.ipynb`: Backup of the comparison plotting notebook.
  - `ComparisonPlot.png`: Example comparison result figure.
- `AblationExperiments/`
  - `A_cnn_transformer_origin/`, `A_cnn_transformer_pre/`: Visualizations for CNN‑Transformer variants in the ablation experiments.
  - `AblationPlot.ipynb`: Notebook for generating ablation plots.
  - `AblationPlot.py`: Python script for ablation visualization.

---

## 🚀 Recommended Workflow

Below is a typical workflow to reproduce CBFAN-related experiments (refer to the notebooks / scripts for exact settings):

1. **Environment setup**  
   Install Python dependencies and ensure Jupyter Notebook runs correctly.

2. **Data preprocessing**  
   Run `data_preprocess.py` in `DatasetPreprocess/`, or follow `DataVisualization.ipynb` to generate feature files and dataset splits.

3. **Train the CBFAN model**  
   Open `CBFAN.ipynb` in `RemainingLifePredictionModel/CBFAN-model/` and:
   - Load and split the data (from `dataresult/` or the outputs of `DatasetPreprocess/`).
   - Configure the model (cross-dimensional attention, dual-temporal attention, multi-head feature-fusion Transformer encoder, etc.).
   - Train and validate the model, saving the best checkpoint as a `.pt` file.

4. **Comparison experiments**  
   In `ComparisonExperiments/` and `PlotCompare/ComparisonExperiments/`, run notebooks and plotting scripts for Transformer‑BiLSTM, CNN‑Transformer and other baselines to obtain performance curves and metrics (MAE, RMSE, Score, etc.).

5. **Ablation experiments**  
   In `AblationExperiments/`, selectively remove or replace key CBFAN components (cross-dimensional attention, dual-temporal attention, feature-fusion encoder, etc.), run `CNN-Transformer-model.ipynb` and related plotting scripts, and analyze each module's contribution.

6. **Visualization and analysis**  
   Use scripts and notebooks under `PlotCompare/` to generate comparison and ablation figures, and analyze the behavior of CBFAN versus the baselines.

---

## 📚 Reference

If you use this repository in your research, please cite the original paper:

> Bearing remaining useful-life prediction method based on cross-dimensional dual-temporal attention fusion.

This README is intended to help you understand the repository layout and typical usage. For detailed implementation and experimental settings, please refer to the actual code and paper.
