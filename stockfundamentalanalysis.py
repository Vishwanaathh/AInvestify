import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Welcome to fundamentals stock classification")

data = pd.read_csv("../nifty_500_stats.csv", sep=";")

data["correct_p_e"] = np.where((data["price_earnings"] < 20) & (data["price_earnings"] > 10), "Yes", "No")
data["Large_cap"] = np.where(data["market_cap"] >= 20000, "Yes", "No")
data["Dividend_Stock"] = np.where(data['dividend_yield'] > 3.5, "Yes", "No")
data['Good_Return_Potential'] = np.where((data['roce'] > np.average(data['roce'])) & (data['roe'] > np.average(data['roe'])), "Yes", "No")
data['Good_Sales_Potential'] = np.where(data['sales_growth_3yr'] > np.average(data['sales_growth_3yr']), "Yes", "No")
data["pb_ratio"] = data["current_value"] / data["book_value"]
data["good_book_value"] = np.where((data["pb_ratio"] < data["pb_ratio"].mean()) & (data["roe"] > 15), "Yes", "No")
data["selling_zone"] = np.where(((data["high_52week"] - data["current_value"]) / data["high_52week"]) <= 0.10, "Yes", "No")
data["good_fundamentals"] = np.where(
    (data["price_earnings"].between(10, 20)) &
    (data["market_cap"] >= 20000) &
    (data["roce"] > data["roce"].mean()) &
    (data["roe"] > data["roe"].mean()) &
    ((data["dividend_yield"] > 3.5) | (data["sales_growth_3yr"] > data["sales_growth_3yr"].mean())),
    1,
    0
)

X = data[['market_cap','current_value','high_52week','low_52week','book_value','price_earnings','dividend_yield','roce','roe','sales_growth_3yr']]
Y = data['good_fundamentals']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, Y)

joblib.dump(model, "fundamentals_stock_model.joblib")
print("Model saved as fundamentals_stock_model.joblib")
