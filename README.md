# DualMFE

This repository contains the implementation of a qualification test from Professor Sun, focusing on time series forecasting with multiple model architectures.

## 📊 Task Analysis

This project addresses a dual-perspective financial modeling challenge:
1. **Time Series Forecasting**: Predicting future returns based on historical factor data
2. **Cross-sectional Regression**: It seems more like a masked regression task, where we are trying to predict the label for each timestamp based on the features at that timestamp and before.

We've chosen to approach this as a regression problem rather than a traditional forecasting task due to the high-frequency nature of the minute-level data and the absence of clearly defined prediction windows.

### Data Description
- **Long time series**: ~530,000 timestamps, 1 year of minute-level data in both train and test datasets
- **High dimensional features**: ~900 features, divided into:
  - **Known features**: ~6 items, mainly LOB features
  - **Anonymized features**: ~890 items, potentially constructed factors
- **Data quality**: Infinite values exist in both train and test datasets (~20 items)

## 🏗️ Model Architecture

### 1. GRU
A specialized GRU architecture with the following structure:
- **Data Flow**: Input → Feature Embedding → GRU Blocks → LayerNorm → Linear Output
- **GRU Block Design**:
  1. LayerNorm1 → GRU Layer → Residual Connection
  2. LayerNorm2 → Feed Forward Network → Residual Connection
- **Implementation Details**:
  - Bidirectional=False with batch_first=True
  - Weight initialization uses Kaiming normal for weights and zero for biases

### 2. vTransformer
Encoder-only Transformer architecture with causal attention mask, implemented using PyTorch's native TransformerEncoder. Features include:
- **Data Flow**: Input → Feature Linear Embedding → Sinusoidal Position Embedding → TransformerEncoder → Linear Output
- **Attention Mechanism**: Multi-head self-attention with causal masking
- **Position Encoding**: Sinusoidal position embedding added to feature embeddings
- **Implementation Details**:
  - Uses PyTorch's TransformerEncoderLayer and TransformerEncoder
  - Configurable number of heads, model dimension, and feed-forward dimension
  - Layer normalization with customizable epsilon
  - Kaiming normal weight initialization

### 3. iTransformer
Implementation based on the paper: [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625). This model inverts the traditional Transformer architecture by treating features as tokens and timestamps as channels, which is particularly effective for time series forecasting tasks with high-dimensional feature spaces.

## 📈 Main Results

| Model | MSE | RMSE | MAE |
|-------|-----|------|-----|
| GRU | 0.0131 | 0.1142 | 0.0893 |
| vTransformer | 0.0004 | 0.0211 | 0.0211 |
| iTransformer | 0.0622 | 0.2495 | 0.1716 |


## 🚀 Usage

### Setup
1. Upload data to `./data/raw` folder
2. Install PyTorch and the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Training
```bash
# Train vTransformer
bash ./scripts/train_defaults/vTransformer.sh

# Train iTransformer
bash ./scripts/train_defaults/iTransformer.sh

# Train GRU
bash ./scripts/train_defaults/GRU.sh
```

### Testing
```bash
# Test vTransformer
bash ./scripts/test_defaults/vTransformer.sh

# Test iTransformer
bash ./scripts/test_defaults/iTransformer.sh

# Test GRU
bash ./scripts/test_defaults/GRU.sh
```

## 🔍 Results Analysis
1. **vTransformer** significantly outperforms other models
2. During training process, **iTransformer** and **GRU** showed better training loss decreasing, especially **iTransformer**, however, it suffers overfitting problem.
3. The training paradigm of **iTransformer** differs fundamentally from GRU and conventional transformers - it first captures dependencies among features before modeling temporal relationships. This architectural inversion may explain its underperformance, as temporal dependencies in high-frequency financial data appear to be more significant than cross-feature relationships. What's more, the original **iTransformer** is designed to fit the multi-variate time-series forecasting task, while in this task, we are trying to predict the label for each timestamp based on the features at that timestamp and before.

## 🔮 Future Direction
1. **Hybrid architectures**: 
   - With a MAE for dimensionality reduction of anonymized features
   - Feature engineering based on the known features, and train a GBDT model or other ML models with low feature dimensionality
   - Design a Fuser to integrate predictions from both models
2. NAS for best network
3. Change downsampling strides and looking-back window size


## 📬 Contact
Yunkai Gao (yunkai.gao@duke.edu)
