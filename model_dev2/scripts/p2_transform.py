import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

# Getting the raw dataset
df = pd.read_pickle('model_dev2/data/raw/HIV_AIDS_data.pkl')
print(df)

# Getting the column names
df.columns

# Cleaning the column names
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('/', '_').str.replace('(', '').str.replace(')', '').str.replace('.', '').str.replace(',', '')
df.columns

# Keeping columns
to_keep = [
    'year',
    'borough',
    'neighborhood_uhf',
    'sex',
    'race_ethnicity',
    'total_number_of_hiv_diagnoses',
    'hiv_diagnoses_per_100000_population',
    'total_number_of_concurrent_hiv_aids_diagnoses',                      
    'proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses',
    'total_number_of_aids_diagnoses',                                       
    'aids_diagnoses_per_100000_population'       
]

# Dropping missing values
df = df[to_keep]
df.dropna(inplace=True)

# Getting the data type
df.dtypes 

# Changing data types
df = df[to_keep]
df[['total_number_of_hiv_diagnoses', 'hiv_diagnoses_per_100000_population', 'total_number_of_concurrent_hiv_aids_diagnoses', 'proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses', 'total_number_of_aids_diagnoses', 'aids_diagnoses_per_100000_population']] = df[['total_number_of_hiv_diagnoses', 'hiv_diagnoses_per_100000_population', 'total_number_of_concurrent_hiv_aids_diagnoses', 'proportion_of_concurrent_hiv_aids_diagnoses_among_all_hiv_diagnoses', 'total_number_of_aids_diagnoses', 'aids_diagnoses_per_100000_population']].apply(pd.to_numeric)
df[['year']] = df[['year']].astype('object')
df.dtypes

# Performing ordinal encoding on year
enc = OrdinalEncoder()
enc.fit(df[['year']])
df['year'] = enc.transform(df[['year']])

df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['year'])
df_mapping_date['year_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for year
df_mapping_date.to_csv('model_dev2/data/processed/mapping_year.csv', index=False)

# Performing ordinal encoding on borough
enc = OrdinalEncoder()
enc.fit(df[['borough']])
df['borough'] = enc.transform(df[['borough']])

df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['borough'])
df_mapping_date['borough_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for borough
df_mapping_date.to_csv('model_dev2/data/processed/mapping_borough.csv', index=False)

# Performing ordinal encoding on neighborhood_uhf
enc = OrdinalEncoder()
enc.fit(df[['neighborhood_uhf']])
df['neighborhood_uhf'] = enc.transform(df[['neighborhood_uhf']])

df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['neighborhood_uhf'])
df_mapping_date['neighborhood_uhf_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for neighborhood_uhf
df_mapping_date.to_csv('model_dev2/data/processed/mapping_neighborhood_uhf.csv', index=False)

# Performing ordinal encoding on sex
enc = OrdinalEncoder()
enc.fit(df[['sex']])
df['sex'] = enc.transform(df[['sex']])

df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['sex'])
df_mapping_date['sex_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for sex
df_mapping_date.to_csv('model_dev2/data/processed/mapping_sex.csv', index=False)

# Performing ordinal encoding on race_ethnicity
enc = OrdinalEncoder()
enc.fit(df[['race_ethnicity']])
df['race_ethnicity'] = enc.transform(df[['race_ethnicity']])

df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['race_ethnicity'])
df_mapping_date['race_ethnicity_ordinal'] = df_mapping_date.index
df_mapping_date

# Saving the mapping as a CSV for race_ethnicity
df_mapping_date.to_csv('model_dev2/data/processed/mapping_race_ethnicity.csv', index=False)

# Saving the entire mapping as a CSV
df.to_csv('model_dev2/data/processed/processed_HIV_AIDS_data.csv', index=False)