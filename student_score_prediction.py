#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ==============================
# Student Score Prediction - Kaggle Notebook
# ==============================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[6]:


# Step 2: Load dataset
# On Kaggle, datasets are usually in /kaggle/input/.../
df = pd.read_csv("/kaggle/input/student-score-prediction/student_performance_dataset.csv")

# Quick look
df.head()


# In[7]:


# Step 3: Data Cleaning & Exploration
print("Missing values:\n", df.isnull().sum())
print("\nDataset Info:")
print(df.info())

# Descriptive stats
df.describe()


# In[13]:


# Step 4: Visualization - Explore relationships

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Scatter plot: Study hours vs Final Score
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Study_Hours_per_Week"], y=df["Final_Exam_Score"])
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()


# In[15]:


# Step 5: Feature Selection (study hours only)
X = df[["Study_Hours_per_Week"]]
y = df["Final_Exam_Score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[16]:


# Step 6: Train Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


# In[17]:


# Step 7: Visualization - Linear Fit
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Exam Score")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()


# In[18]:


# Step 8: Polynomial Regression (Degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train_poly)

y_pred_poly = poly_reg.predict(X_test_poly)

# Evaluation
mae_poly = mean_absolute_error(y_test_poly, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test_poly, y_pred_poly))
r2_poly = r2_score(y_test_poly, y_pred_poly)

print("Polynomial Regression (deg=2) Performance:")
print("MAE:", mae_poly)
print("RMSE:", rmse_poly)
print("R²:", r2_poly)


# In[19]:


# Step 9: Visualization - Polynomial Fit
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", label="Actual")
plt.scatter(X, poly_reg.predict(X_poly), color="red", label="Polynomial Fit")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Exam Score")
plt.title("Polynomial Regression (Degree=2)")
plt.legend()
plt.show()


# In[22]:


# Step 10 (Bonus): Multi-feature Regression
# Using extra predictors like attendance, sleep hours, etc. if available
features = ["Study_Hours_per_Week", "attendance" ]
available_features = [f for f in features if f in df.columns]

if available_features:
    X_multi = df[available_features]
    y_multi = df["Final_Exam_Score"]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )

    multi_reg = LinearRegression()
    multi_reg.fit(X_train_m, y_train_m)

    y_pred_m = multi_reg.predict(X_test_m)

    print("Multi-Feature Regression Performance:")
    print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
    print("RMSE:", np.sqrt(mean_squared_error(y_test_m, y_pred_m)))
    print("R²:", r2_score(y_test_m, y_pred_m))
else:
    print("No extra features available for multi-feature regression.")

