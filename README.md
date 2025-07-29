# 🏡 House Price Prediction API with FastAPI

This project serves a trained regression model as an API using FastAPI. It predicts house prices based on engineered features from housing data.
aims to predict house prices using regression model and evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price
## 🚀 Features

- CSV upload endpoint to predict multiple house prices
- Feature engineering with `create_features()` function
- Scikit-learn pipeline integration

## 📁 Folder Structure

- `app/`: Contains the FastAPI app and feature engineering code
- `models/`: Contains the serialized model (`.pkl`)
- `test.csv`: Example input
- `requirements.txt`: Python dependencies

## ▶️ How to Run Locally

```bash
# 1. Create virtual env
conda create -n house_api python=3.11 -y
conda activate house_api

# 2. Clone this repo
git clone https://github.com/yourusername/house_price_demo.git
cd house_price_demo

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
uvicorn app.main:app --reload

```
## 🗂️ Dataset
This project uses the dataset from the House Prices - Advanced Regression Techniques competition on Kaggle.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data



# Result
The result is RMSE 0.11038 on X_test and 
RMSE 0.12280 on test data
