import pandas as pd

df = pd.read_csv('/home/yuz/wsi-survival/clinical_survival_processed.csv')

print(len(df))
print(df["cases.submitter_id"].nunique())