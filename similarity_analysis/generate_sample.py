import pandas as pd


df = pd.read_csv("integrated_dataset.csv")
sample = df.sample(n=30000, random_state=42)
sample.to_csv("30k_sample.csv", index=False)