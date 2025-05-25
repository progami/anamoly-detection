# RLNAS for Autoencoder-based Anomaly Detection in Time Series Data

Implementation of Neural Architecture Search for automated anomaly detection in IoT and smart building time series data.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This project implements a Reinforcement Learning-based Neural Architecture Search (RLNAS) framework that automatically discovers optimal autoencoder architectures for detecting anomalies in time series data.

### Key Features
- 🤖 **Automated Architecture Design**: Uses RL to automatically design neural networks
- 📊 **Time Series Anomaly Detection**: Specialized for IoT and smart building data
- 🎯 **High Performance**: Achieves 97.6% precision with minimal false alarms
- 🔧 **Flexible Framework**: Works with various time series datasets

## 📈 Results

Our NAS framework discovered an architecture achieving:
- **F1-Score**: 87.86%
- **Precision**: 97.62% (very few false alarms!)
- **Recall**: 79.87%

### Discovered Architecture
```
Block 1: Conv(16 filters, kernel=15) → AvgPool(7) → BatchNorm
Block 2: Conv(16 filters, kernel=15) → BatchNorm
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone git@github.com:progami/anamoly-detection.git
cd anamoly-detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Preparation

1. Place your time series data in `dataset/weather_data/` 
2. Generate synthetic anomalies:
```bash
python prepare_data.py
```

This creates:
- Training data (clean, no anomalies)
- Test data (20% contaminated with anomalies)
- Separate datasets for each feature (Temperature, Humidity, Pressure)

### Running NAS

Train the Neural Architecture Search to find optimal architectures:

```bash
# For Temperature anomaly detection
python main.py --train_dataset_path dataset/synth_ts_data/train_IHAMPS1_Temperature_C.csv \
               --test_dataset_path dataset/synth_ts_data/test_IHAMPS1_Temperature_C.csv \
               --output_file results/nas_results.txt
```

The NAS will:
- Evaluate 200 different architectures
- Use reinforcement learning to guide the search
- Output the best architecture and its performance

### Visualizing Results

Use the provided Jupyter notebooks for visualization and analysis:

```bash
jupyter notebook notebooks/autoencoder_based_time_series_anomaly_detection.ipynb
```

## 📁 Project Structure

```
.
├── config/                          # Configuration files
│   ├── autoencoder_params.py       # Training parameters
│   ├── controller_params.py        # RL controller settings
│   └── search_space.py             # Architecture search space
├── dataset/                         # Data directory
│   ├── weather_data/               # Raw weather data
│   ├── gen_anomalies/              # Generated anomalies
│   └── synth_ts_data/              # Train/test datasets
├── models/                          # Model definitions
│   ├── autoencoder.py              # Autoencoder architecture
│   └── controller.py               # RL controller network
├── notebooks/                       # Jupyter notebooks
│   └── autoencoder_based_time_series_anomaly_detection.ipynb
├── utils/                           # Utility functions
│   ├── data_loaders.py             # Data loading utilities
│   └── utils.py                    # Helper functions
├── main.py                         # Main NAS training script
├── prepare_data.py                 # Data preparation script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Configuration

### Search Space Configuration (`config/search_space.py`)
```python
n_episodes = 200        # Number of architectures to try
n_blocks = 2           # Number of blocks in architecture
```

### Training Configuration (`config/autoencoder_params.py`)
```python
epochs = 100           # Training epochs per architecture
batch_size = 32       
learning_rate = 0.001
```

## 📊 Datasets

The framework works with time series data containing:
- **Features**: Temperature, Humidity, Pressure (or any univariate time series)
- **Window Size**: 100 timesteps
- **Anomaly Types**: Point anomalies, contextual anomalies

### Using Your Own Data

1. Format your data as CSV with columns matching the expected format
2. Place in `dataset/weather_data/`
3. Update `dataset_names` in `prepare_data.py`
4. Run data preparation script

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.