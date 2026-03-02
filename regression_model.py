import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("house_price_raw_data.csv")
print("Dataset loaded ✅")
print(data.head())

# Convert yes/no columns to categorical
yes_no_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea']
for col in yes_no_cols:
    data[col] = data[col].astype('category')

# Furnishing status as categorical
data['furnishingstatus'] = data['furnishingstatus'].astype('category')

# Ensure numeric columns are numeric
numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Separate features and target
X = data.drop('price', axis=1)
y = data['price']

# Automatically detect column types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['category', 'object']).columns

print("Numerical features:", list(numerical_features))
print("Categorical features:", list(categorical_features))

# Numeric pipeline: impute missing values + scale
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: impute missing values + one-hot encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine numeric and categorical pipelines
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])


# Full pipeline with Random Forest regressor
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model_pipeline.fit(X_train, y_train)
print("Model trained successfully ✅")


# Evaluate model
y_pred = model_pipeline.predict(X_test)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))

# Interactive input for new house prediction
print("\nEnter the details of your house:")

area = float(input("Enter area in sq ft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
stories = int(input("Enter number of stories: "))
mainroad = input("Main road access (yes/no): ")
guestroom = input("Guest room (yes/no): ")
basement = input("Basement (yes/no): ")
hotwaterheating = input("Hot water heating (yes/no): ")
airconditioning = input("Air conditioning (yes/no): ")
parking = int(input("Number of parking spots: "))
prefarea = input("Preferred area (yes/no): ")
furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ")

new_house = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

predicted_price = model_pipeline.predict(new_house)
print(f"\n💰 Predicted House Price: ₹{predicted_price[0]:,.2f}")

# Visualization: Actual vs Predicted Prices
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()