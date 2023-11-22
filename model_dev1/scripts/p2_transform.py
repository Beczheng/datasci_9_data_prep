import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

# Getting the raw dataset
df = pd.read_pickle('model_dev1/data/raw/death_rate_life_expectancy_data.pkl')
print(df)

# Getting the column names
df.columns

# Cleaning the column names
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('(', '').str.replace(')', '')

# Keeping columns 
to_keep = [
    'year',
    'race',
    'sex',
    'average_life_expectancy_years',
    'age_adjusted_death_rate'
]

# Dropping missing values
df = df[to_keep]
df.dropna(inplace=True)

# Getting the data type
df.dtypes 

# Performing ordinal encoding on year
enc = OrdinalEncoder()
enc.fit(df[['year']])
df['year'] = enc.transform(df[['year']])

# Creating a dataframe with mapping for year
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['year'])
df_mapping_date['year_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for year
df_mapping_date.to_csv('model_dev1/data/processed/mapping_year.csv', index=False)

# Performing ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

# Creating a dataframe with mapping for sex
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_date['sex_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for sex
df_mapping_date.to_csv('model_dev1/data/processed/mapping_sex.csv', index=False)

# Performing ordinal encoding on race
enc = OrdinalEncoder()
enc.fit(df[['race']])
df['race'] = enc.transform(df[['race']])

# Creating a dataframe with mapping for race
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['race'])
df_mapping_date['race_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for race
df_mapping_date.to_csv('model_dev1/data/processed/mapping_race.csv', index=False)

# Saving the entire mapping as a CSV
df.to_csv('model_dev1/data/processed/processed_death_rate_life_expectancy_data.csv', index=False)