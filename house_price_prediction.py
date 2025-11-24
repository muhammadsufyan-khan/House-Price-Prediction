
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

sns.set(style="whitegrid")

# 1. Load dataset
df = pd.read_csv("dataset/House Price Prediction Dataset.csv")

# 2. Select relevant features
df = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Location', 'Price']]
X = df.drop('Price', axis=1)
y = df['Price']

# 3. Encode categorical feature
categorical_features = ['Location']
preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder='passthrough'
)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = GradientBoostingRegressor()
model.fit(preprocess.fit_transform(X_train), y_train)

# 6. Predictions
y_pred = model.predict(preprocess.transform(X_test))

# 7. Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

# 8. Visualizations
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(y_test - y_pred, kde=True)
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.show()
