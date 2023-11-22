import pandas as pd
import pickle
import xgboost 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Identifying the independent (predictors) and dependent (target) variables.
# Indepdent (predictors): All other variables (year, race, sex, and average_life_expectancy_years)
# Dependent (target): Age_adjusted_death_rate

# Getting the processed dataset
df = pd.read_csv('model_dev1/data/processed/processed_death_rate_life_expectancy_data.csv')
print(df)

# Dropping rows with missing values
df.dropna(inplace=True)
print(df)

# Creating a dataframe for each of my variables
X = df.drop('age_adjusted_death_rate', axis=1)
print(X)
y = df['age_adjusted_death_rate']  
print(y) 

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler to the features and transform
scaler.fit(X) 
X_scaled = scaler.transform(X)
X_scaled

# Splitting the scaled data into training, validation, and testing (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Checking the size of each dataset
(X_train.shape, X_val.shape, X_test.shape)