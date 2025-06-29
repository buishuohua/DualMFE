# DualMFE
This repo is implementaion of the qualification test from Professor Sun.
# Task Analysis

1. Long time series: ~530000 timestamps, 1 year minute level data resides in both train and test dataset
2. High dimensonal features ~900 features, but divided into 2 categories:
   - Known features: ~6 items
   - Anonymize features: ~890 items
3. Considerations, **ordered by emergency**:
   1. de-dimention of features;
   2. downsampling timesteps
   3. Known features seems to contain domain knowledge, might be more useful; considering dual feature predicter
   4. Feature engineering: construct features based on known features

### Model Architeture

#### Known features predicter


#### Anonymized features predicter

1. Patched Transformers
   - PatchTST
   - iTransformers

#### Fused Layer

# Usage
1. Upload data to `./data/raw` folder
2. Install Pytorch and the necessary dependencies.
```bash
pip install -r requirements.txt
```
3. Train and test the model. We provide all the above tasks under the folder ./scripts/train_defaults and ./scripts/test_defaults. You can reproduce the results as the following examples:
```bash
# Train vTransformer
bash ./scripts/train_defaults/vTransformer.sh

# Train iTransformer
bash ./scripts/train_defaults/iTransformer.sh

# Train GRU
bash ./scripts/train_defaults/GRU.sh

# Test vTransformer
bash ./scripts/test_defaults/vTransformer.sh

# Test iTransformer
bash ./scripts/test_defaults/iTransformer.sh

# Test GRU
bash ./scripts/test_defaults/GRU.sh
```
# Main Results
| Model | MSE | RMSE | MAE |
| --- | --- | --- | --- |
| GRU | 0.0131 | 0.1142 | 0.0893 |
| vTransformer | 0.0004 | 0.0211 | 0.0211 |
| iTransformer | 0.0622 | 0.2495 | 0.1716 |

# Results Analysis \& Issues

# Future Direction

# Contact
Yunkai Gao (yunkai.gao@duke.edu)
