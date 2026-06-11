# ============================================
# House Price Prediction using Linear Regression
# California Housing Dataset
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("=" * 50)
print("LOADING CALIFORNIA HOUSING DATASET")
print("=" * 50)

housing = fetch_california_housing(as_frame=True)

df = pd.concat(
    [housing.data,
     housing.target.rename("HousePrice")],
    axis=1
)

print("\nFirst 5 Rows:")
print(df.head())

# ============================================
# STEP 2: DATASET INFORMATION
# ============================================

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

# ============================================
# STEP 3: CHECK MISSING VALUES
# ============================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())

# ============================================
# STEP 4: DESCRIPTIVE STATISTICS
# ============================================

print("\n" + "=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)

print(df.describe())

# ============================================
# STEP 5: HISTOGRAMS
# ============================================

df.hist(figsize=(15,10), bins=30)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# ============================================
# STEP 6: CORRELATION HEATMAP
# ============================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================
# STEP 7: MEDIAN INCOME VS HOUSE PRICE
# ============================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="HousePrice",
    alpha=0.4
)

plt.title("Median Income vs House Price")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.show()

# ============================================
# STEP 8: FEATURES & TARGET
# ============================================

X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# ============================================
# STEP 9: TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape :", X_test.shape)

# ============================================
# STEP 10: TRAIN MODEL
# ============================================

print("\nTraining Linear Regression Model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Training Completed!")

# ============================================
# STEP 11: MODEL COEFFICIENTS
# ============================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")
print(coefficients)

# ============================================
# STEP 12: PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

# ============================================
# STEP 13: EVALUATION METRICS
# ============================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R² Score                  : {r2:.4f}")

# ============================================
# STEP 14: ACTUAL VS PREDICTED
# ============================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted")
plt.show()

# ============================================
# STEP 15: RESIDUAL PLOT
# ============================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.5
)

plt.axhline(
    y=0,
    color='red',
    linestyle='--'
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ============================================
# STEP 16: SAVE MODEL
# ============================================

joblib.dump(
    model,
    "california_housing_model.pkl"
)

print("\nModel saved as california_housing_model.pkl")

# ============================================
# STEP 17: SAMPLE PREDICTION
# ============================================

sample_house = [[
    8.3252,      # MedInc
    41.0,        # HouseAge
    6.984127,    # AveRooms
    1.023810,    # AveBedrms
    322.0,       # Population
    2.555556,    # AveOccup
    37.88,       # Latitude
    -122.23      # Longitude
]]

prediction = model.predict(sample_house)

print("\nPredicted House Price:")
print(prediction[0])

print("\nProject Completed Successfully!")