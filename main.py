import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

filename = "properties.csv"
df = pd.read_csv(filename)

# Identify columns with more than 10000 missing values and drop them
missing_values_count = df.isnull().sum()
columns_with_many_missing = missing_values_count[missing_values_count > 10000]
df.drop(columns=columns_with_many_missing.index, inplace=True)

df.drop(['id'], axis=1, inplace=True)

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

# Defining the target (price) and features by dropping irrelevant columns
target = df['price']
features = df.drop(columns=['price'])

# Using OneHotEncoder to transform categorical variables in features
categorical_features = features.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical_data = encoder.fit_transform(features[categorical_features])
encoded_categorical_df = pd.DataFrame(
    encoded_categorical_data, 
    columns=encoder.get_feature_names_out(categorical_features),
    index=features.index
)
# Removing original categorical columns and adding encoded ones
features_encoded = features.drop(columns=categorical_features).join(encoded_categorical_df)

# Feature Scaling for Numerical Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_encoded)
features_scaled = pd.DataFrame(scaled_features, columns=features_encoded.columns, index=features_encoded.index)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Displaying shapes to confirm preprocessing steps
(X_train.shape, X_test.shape), y_train.shape, y_test.shape

#The top 10 features most correlated with price are:
# Adding the target column back temporarily for correlation analysis
correlation_df = features_scaled.copy()
correlation_df['price'] = target

# Calculating the correlation matrix
correlation_matrix = correlation_df.corr()

# Sorting features by their absolute correlation with price, excluding 'price' itself
price_correlation = correlation_matrix['price'].drop('price').abs().sort_values(ascending=False)

# Displaying the top 10 features most correlated with price
top_10_correlated_features = price_correlation.head(10)
top_10_correlated_features

# Choosing the feature with the highest correlation with price for single linear regression visualization
feature_name = top_10_correlated_features.index[0]
X_train_feature = X_train[feature_name].values.reshape(-1, 1)

# Re-fitting the model with a single feature for visualization purposes
single_feature_model = LinearRegression()
single_feature_model.fit(X_train_feature, y_train)

# Predicting based on the single feature in the test set
X_test_feature = X_test[feature_name].values.reshape(-1, 1)
y_pred_single = single_feature_model.predict(X_test_feature)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[feature_name], y_test, color="blue", label="Actual Prices", alpha=0.5)
plt.plot(X_test[feature_name], y_pred_single, color="red", label="Predicted Prices (Linear Fit)")
plt.xlabel(feature_name)
plt.ylabel("Price")
plt.title(f"Linear Regression with {feature_name} vs. Price")
plt.legend()
plt.show()

# Initializing and training the Linear Regression model with all selected features
multiple_linear_model = LinearRegression()
multiple_linear_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred_multiple = multiple_linear_model.predict(X_test)

# Calculating metrics for evaluation
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

mse_multiple, r2_multiple

# Initialize the XGBoost Regressor
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model on the training set
xgboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgboost = xgboost_model.predict(X_test)


# Calculate evaluation metrics
mse_xgboost = mean_squared_error(y_test, y_pred_xgboost)
r2_xgboost = r2_score(y_test, y_pred_xgboost)

# Print the evaluation metrics
print(f"XGBoost Mean Squared Error: {mse_xgboost}")
print(f"XGBoost R-squared: {r2_xgboost}")

#SVM – Support vector Machine
model_SVR = svm.SVR()
model_SVR.fit(X_train, y_train)
y_pred = model_SVR.predict(X_test)

print(mean_absolute_percentage_error(y_test, y_pred))

#Random Forest Regression
X = features_encoded.iloc[:,1:2].values  #features
y = features_encoded.iloc[:,2].values  # Target variable

#Check for and handle categorical variables
label_encoder = LabelEncoder()
x_categorical = features_encoded.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = features_encoded.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

# Fit the regressor with x and y data
regressor.fit(x, y)

# Access the OOB Score
oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

# Making predictions on the same data or new data
predictions = regressor.predict(x)

# Evaluating the model
mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')