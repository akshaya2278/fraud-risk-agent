import pandas as pd
from data_prep import prepare_and_engineer_data

print("Loading data...")
df = prepare_and_engineer_data("creditcard.csv")

# Grab 30 normal transactions and 15 known fraud transactions
normal_tx = df[df["Class"] == 0].sample(30, random_state=42)
fraud_tx = df[df["Class"] == 1].sample(15, random_state=42)

# Combine and shuffle them so the AI has to hunt for them
demo_df = pd.concat([normal_tx, fraud_tx]).sample(frac=1, random_state=42)
demo_df.to_csv("hackathon_demo.csv", index=False)

print("✅ hackathon_demo.csv created with guaranteed fraud cases!")