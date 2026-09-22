import pandas as pd

print("Loading the massive dataset...")
df = pd.read_csv("full_dataset_engineered.csv")

# Grab ALL the fraud cases (approx 492 rows)
fraud_tx = df[df["Class"] == 1]

# Grab a random sample of 9,500 normal transactions
normal_tx = df[df["Class"] == 0].sample(9500, random_state=42)

# Combine them into a 10,000 row dataset (approx 5MB) and shuffle
git_df = pd.concat([fraud_tx, normal_tx]).sample(frac=1, random_state=42)

git_df.to_csv("github_dataset.csv", index=False)
print("✅ github_dataset.csv created! Size is approximately 5MB.")