import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Identifying the independent (predictors) and dependent (target) variables
# Indepdent (predictors): All other variables
# Dependent (target): Proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses

# Loading the processed dataset
df = pd.read_csv('model_dev2/data/processed/processed_HIV_AIDS_data.csv')
print(df)

# Creating a dataframe for my variables
X = df.drop('proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses', axis=1)
print(X)
y = df['proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses']  
print(y) 

# Initializing the StandardScaler
scaler = StandardScaler()

# Fitting the scaler and transforming
scaler.fit(X) 
X_scaled = scaler.transform(X)
X_scaled

# Saving the scaler
pickle.dump(scaler, open('model_dev2/models/scaler.sav', 'wb'))

# Splitting the scaled data into training, validation, and testing (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Saving the X_train
pickle.dump(X_train, open('model_dev2/models/X_train.sav', 'wb'))
pickle.dump(X.columns, open('model_dev2/models/X_columns.sav', 'wb'))