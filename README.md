# 🌡️ Hackathon Task 3: Statistical Global Forecasting

This repository contains an advanced forecasting pipeline built on classical statistical principles and deep learning. It is optimized to predict **monthly average energy consumption (`x2`)** for industrial heat pumps.

## 🚀 Key Features

- **Statistical Feature Engineering**:
  - Stationarity via log-returns and differencing.
  - Significance-based lags using PACF (Partial Autocorrelation Function).
  - Noise reduction via Butterworth low-pass filtering.
  - Volatility modeling to capture clustering of variance.
- **Advanced Architecture**: Probabilistic-ready LSTM and Transformer models focused on scalar monthly averages.
- **Scalable Pipeline**:
  - Fuzzy coordinate matching with Open-Meteo daily weather updates.
  - High-performance data processing for multi-GB datasets.
- **HPC Support**: SLURM scripts and automated environment setup for the Athena supercomputer.

## 📂 Project Structure

- `📂 src/`: Core logic (`features.py`, `model.py`, `trainer.py`).
- `📂 scripts/`: Operational scripts.
  - `prepare_data.py`: Preprocesses raw `data.csv` and merges weather.
  - `inference.py`: Generates batch predictions for a given dataset.
- `📂 notebooks/`: EDA, feature research, and training experiments.
- `📂 data/`: Data storage (support for CSV and Parquet).
- `📂 models/`: Persistent model weights.

## 🔧 Workflow

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation** (Processes `data.csv` and calculates statistics):
   ```bash
   python scripts/prepare_data.py
   ```

3. **Training**:
   ```bash
   python train.py --model transformer --epochs 50
   ```

4. **Inference** (Generates vector for `task3.pdf` compliance):
   ```bash
   python scripts/inference.py --input data/new_data.csv --model_path models/transformer_best.pth
   ```

---
*Optimized for accurately capturing energy trends across large device fleets.*