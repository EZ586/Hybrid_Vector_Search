import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_json("dataset/yelp_academic_dataset_review.json", lines=True, nrows=200000)
print(df.head(20))

