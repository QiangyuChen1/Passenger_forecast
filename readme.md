# Installation Guide

## Step 1: Install Dependencies

Run the following command to install the required dependencies:

```bash
conda create -n hk-passenger-forecast python=3.9.23 -y
pip install -r requirements.txt
```
## Step 2: Run Scripts

Navigate to the `scripts` directory and run the following scripts in order:

1. 数据预处理:
    ```bash
    python scripts/01_data_exploration.py
    ```

2. 模型训练:
    ```bash
    python scripts/02_model_training.py
    ```

3. 预测&分析:
    ```bash
    python scripts/03_forecast_analysis.py
    ```