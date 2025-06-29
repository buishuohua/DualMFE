# DualMFE
This repo is implementaion of the qualification test from Professor Sun.
# Task Analysis

# Usage
1. Install Pytorch and the necessary dependencies.
```bash
pip install -r requirements.txt
```
2. Train and test the model. We provide all the above tasks under the folder ./scripts/train_defaults and ./scripts/test_defaults. You can reproduce the results as the following examples:
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
