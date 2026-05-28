import pandas as pd
import os

df = pd.read_csv("gdc_manifest_top800_part02.txt", sep="\t")

chunk_size = 50

os.makedirs("manifests_50", exist_ok=True)

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_csv(
        f"manifests_50/batch0_{i//chunk_size}.txt",
        sep="\t",
        index=False
    )