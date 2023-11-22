
# Dropping columns
to_drop = [
    'year'
]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

