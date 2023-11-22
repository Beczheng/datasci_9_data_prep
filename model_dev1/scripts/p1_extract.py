import pandas as pd

# Loading the dataset
# Original link: https://catalog.data.gov/dataset/nchs-death-rates-and-life-expectancy-at-birth/resource/eb20e7ad-2a82-4dce-82df-503a0bdb27be
datalink = 'https://data.cdc.gov/api/views/w9j2-ggv5/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(datalink)
print(df)

# Saving as a CSV file to model_dev1/data/raw
df.to_csv('model_dev1/data/raw/death_rate_life_expectancy_data.csv', index=False)

# Saving as a Pickle file to model_dev1/data/raw
df.to_pickle('model_dev1/data/raw/death_rate_life_expectancy_data.pkl')