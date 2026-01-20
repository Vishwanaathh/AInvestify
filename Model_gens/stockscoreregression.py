import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
print("Welcome to fundamentals stock regression")
data=pd.read_csv("../nifty_500_stats.csv",sep=";")
data.head()
dicti={"Yes":1,"No":1}

data["fundamental_score"] = (
    0.20 * (data["price_earnings"].between(10, 20)).astype(int) +
    0.20 * (data["market_cap"] >= 20000).astype(int) +
    0.20 * (data["roce"] > data["roce"].mean()).astype(int) +
    0.20 * (data["roe"] > data["roe"].mean()).astype(int) +
    0.20 * (
        (data["dividend_yield"] > 3.5) |
        (data["sales_growth_3yr"] > data["sales_growth_3yr"].mean())
    ).astype(int)
)

X=data[[ 'market_cap','current_value', 'high_52week', 'low_52week', 'book_value',
       'price_earnings', 'dividend_yield', 'roce', 'roe', 'sales_growth_3yr']]
Y=data['fundamental_score']
modell=XGBRegressor(n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42)
modell.fit(X,Y)
joblib.dump(modell,'stock_score_regression.pkl')
print("model dumped")
