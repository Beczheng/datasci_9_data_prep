import pandas as pd

# Loading the dataset
# Original link: https://catalog.data.gov/dataset/hiv-aids-diagnoses-by-neighborhood-sex-and-race-ethnicity/resource/1da36ee7-5023-4e76-b4e2-e43be3a377f4
datalink = 'https://data.cityofnewyork.us/api/views/ykvb-493p/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(datalink)
df.size

# Saving as CSV file to model_dev2/data/raw
df.to_csv('model_dev2/data/raw/HIV_AIDS_data.csv', index=False)

# Saving as Pickle file to model_dev2/data/raw
df.to_pickle('model_dev2/data/raw/HIV_AIDS_data.pkl')