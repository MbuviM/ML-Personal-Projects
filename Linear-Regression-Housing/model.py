# %% [markdown]
# # Housing Price Project
# 
# This is a project that uses regression to determine the prices of houses. The sample dataset contains 20,000 rows and 21 columns. 

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd # Data Analysis and cleaning
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sns # Data Visualization
# import numpy as np 
import plotly.express as px # Data Visualization

# %% [markdown]
# ## Load Dataset

# %%
housing_prices = pd.read_csv("Part1_house_price.csv")

# %%
# Checking for a sample size of the dataset
housing_prices.sample(10)

# %%
# Checking for null values
housing_prices.isnull().sum()

# %% [markdown]
# There are no null values present.

# %%
# Drop unnecessary columns
housing_prices.drop(columns="id", axis=1, inplace=True)

# %%
# Convert date format to YYYY-MM-DD
housing_prices["date"] = housing_prices["date"].str[:8].astype("int64")

# %%
housing_prices.sample()

# %%
housing_prices.info()

# %% [markdown]
# ## Data Analysis

# %%
# Statistical Analysis
housing_prices.describe()

# %% [markdown]
# From the analysis, there are 20,000 rows in each of the 21 columns. The mean price is 535,567.9 USD while the minimum prices is 75,000 and maximum price is 7,700,000 USD. Most values lie between 317,000 and 640,000. In price, sqft_lot and sqft_lot15, the maximum values are way higher than the upper quartile range which makes the mean deviate slightly from the median. This means, they have many outliers.

# %%
# Boxplot for all columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=housing_prices, orient="h", palette="Set2")
plt.title("Boxplot for All Numerical Columns")
plt.show()

# %%
# Finding the location from where the dataset was collected
# Using Plotly
fig = px.scatter_geo(housing_prices, lat='lat', lon='long'
                    )
fig.update_layout(title="Houses Location", geo_scope="usa")

#fig.update_geos(projection_type="natural earth")
fig.show()

# %% [markdown]
# From the map, it is clear that the dataset was collected from houses in Washington, USA. In addition, the houses are located in the same area hence the assumption they are all in the same neighborhood.

# %% [markdown]
# ## Model Creation

# %%
#pip install scikit_learn

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# %%
# Splitting the dataset
X = housing_prices.drop(columns="price")
y=housing_prices["price"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Instantiate the model
model = LinearRegression()

# %%
# Fit the model
model.fit(X_train, y_train)

# %% [markdown]
# ## Model Prediction

# %%
y_pred = model.predict(X_test)

# %% [markdown]
# ## Performance Metrics

# %%
## R2_Score and MSE
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("The MSE score is: ", mse)
print("The R2 Score is: ", r2)
print("The MAE score is: ", mae)

# %% [markdown]
# The high MSE score is due to the presence of outliers in the columns of the dataset. Mean Squared Error is sensitive to outliers.

# %% [markdown]
# ## 2. Random Forest

# %%
# Instantiate the model
random_model = RandomForestRegressor()

# %%
# Fit the model
random_model.fit(X_train, y_train)

# %%
# Make Predictions
y_test_pred = random_model.predict(X_test)

# %%
# Perfomance metrics
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
print("The MSE score is: ", mse)
print("The R2 Score is: ", r2)
print("The MAE score is: ", mae)

# %% [markdown]
# ## 3. Gradient Boosting

# %%
boost_model = GradientBoostingRegressor()

# %%
boost_model.fit(X_train, y_train)

# %%
y_test_pred = boost_model.predict(X_test)

# %%
# Perfomance metrics
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
print("The MSE score is: ", mse)
print("The R2 Score is: ", r2)
print("The MAE score is: ", mae)

# %% [markdown]
# These are the scores for the three models without feature engineering. 

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# ### Standardization

# %%
from sklearn.preprocessing import StandardScaler

# Instantiate Scaler
scaler = StandardScaler()

housing_prices[['price', 'sqft_lot', 'sqft_lot15']] = scaler.fit_transform(housing_prices[['price', 'sqft_lot', 'sqft_lot15']])

# %%
housing_prices.sample()

# %%
# Splitting the dataset
X_scaled = housing_prices.drop(columns="price")
y_scaled =housing_prices["price"]

# %%
X_scaled_train, X_scaled_test, y_scaled_train, y_scaled_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### 1. Linear Regression

# %%
# Instantiate model
model = LinearRegression()

# Fit model
model.fit(X_scaled_train, y_scaled_train)

# Predict model
y_pred = model.predict(X_scaled_test)

# Performance metrics
mse = mean_squared_error(y_scaled_test, y_pred)
r2 = r2_score(y_scaled_test, y_pred)
mae = mean_absolute_error(y_scaled_test, y_pred)
print("The MSE score is: ", mse)
print("The R2 Score is: ", r2)
print("The MAE score is: ", mae)

# %%



