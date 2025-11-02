import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# Example training data
data = pd.DataFrame([
    {"crop_name": "Tomato", "location": "Port Louis", "temperature": 28, "rainfall": 50, "season": "Summer", "historical_price": 2.5, "price": 3.0},
    {"crop_name": "Potato", "location": "Curepipe", "temperature": 22, "rainfall": 70, "season": "Winter", "historical_price": 1.8, "price": 2.2},
    # Add more rows of historical crop prices
])

# Features and target
X = data[["crop_name", "location", "temperature", "rainfall", "season", "historical_price"]]
y = data["price"]

# Preprocessing
categorical_cols = ["crop_name", "location", "season"]
numerical_cols = ["temperature", "rainfall", "historical_price"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, random_state=42))
])

# Train
model.fit(X, y)

# Save the trained model
joblib.dump(model, "crop_price_model.pkl")
print("Model saved as crop_price_model.pkl")
