# Retail Demand Forecasting 📦📈

End-to-end demand forecasting pipeline on 76,000+ retail transactions across 5
product categories and 4 regions - from exploratory analysis of what drives demand
to a tuned XGBoost model that predicts it with high accuracy.

## 📌 Objective
- Understand what factors drive product demand (promotions, pricing, seasonality, external shocks)
- Engineer features that capture pricing dynamics and competitive position
- Build and tune a regression model to predict demand, and validate it against the EDA findings

## 🛠️ Tools & Libraries
- **Python** - Pandas, NumPy, Matplotlib, Seaborn
- **Scikit-learn** - `train_test_split`, `LabelEncoder`, `RandomizedSearchCV`, regression metrics
- **XGBoost** - `XGBRegressor`

## 🧹 Data Cleaning & Feature Engineering (EDA notebook)
- Verified no missing values or duplicates across 76,000 rows
- Parsed `Date` into `Year`, `month`, `week_day` for time-based analysis
- Engineered `Final Price` (list price adjusted for discount)
- Engineered `Sales value` (`Units Sold × Final Price`)
- Engineered `price to competitor` (High/Low flag vs. competitor pricing) to test price positioning effects
- Exported the enriched dataset to `preprocessed_data.csv` as input for the modeling notebook

## 📊 Key Findings (EDA)

| Question | Finding |
|---|---|
| Top category by volume | **Groceries** sells the most units; **Furniture** the least |
| Top category by revenue | **Furniture** generates the highest sales value despite lower volume |
| Effect of promotions | Promotions clearly **increase** both units sold and demand |
| Effect of epidemics | Epidemics clearly **decrease** both units sold and demand |
| Seasonal pattern | **Summer** has the highest demand; **Spring** the lowest |
| Weather pattern | **Sunny** weather sees the highest demand; **Snowy** the lowest |
| Monthly pattern | **August** peaks; **May** is the low point |
| Regional pattern | Demand is fairly even across regions, with **South** slightly ahead |
| Pricing vs. competitor | Products priced **below** competitors see consistently higher demand, across every category |
| Correlation check | `Price` and `Competitor Pricing` are highly collinear (0.98) - expected, since pricing tracks the market |

## 🤖 Demand Forecasting Model

Trained an `XGBRegressor` to predict `Demand` using engineered and raw features
(Store, Product, Category, Region, Price, Inventory, Discount, Weather, Promotion,
Competitor Pricing, Seasonality, Epidemic, and time features). Categorical variables
were label-encoded (a valid choice for tree-based models, which split on ordinal
category codes rather than requiring one-hot vectors).

Hyperparameters were tuned with `RandomizedSearchCV` (25 candidate combinations,
3-fold CV, optimizing for MAE) over n_estimators, max_depth, learning_rate,
subsample, colsample_bytree, and min_child_weight.

**Results on held-out test data:**

| Metric | Value |
|---|---|
| R² | **0.961** |
| RMSE | 9.22 |
| MAE | 6.04 |

**Feature importance confirms the EDA**: `Epidemic` and `Promotion` are the two
most influential features in the trained model — the same two factors flagged as
having the strongest visual effect on demand during exploratory analysis. This
agreement between EDA and model behavior is a strong internal consistency check:
the patterns identified through visualization are the same ones the model learned
to rely on.

## 🚀 How to Run
```bash
git clone <your-repo-url>
cd retail-demand-forecasting
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Step 1: run EDA and generate preprocessed_data.csv
jupyter notebook EDA.ipynb

# Step 2: train and evaluate the forecasting model
jupyter notebook Demand_Forecast.ipynb
```

## 📂 Repo Structure
```
├── demand_forecasting.csv
├── preprocessed_data.csv
├── EDA.ipynb
├── Demand_Forecast.ipynb
└── README.md
```
## 📄 License
MIT
