from data_prep import prepare_and_engineer_data

print("Engineering the full 285,000+ row dataset. This will take about 1-2 minutes...")
df = prepare_and_engineer_data("creditcard.csv")

print("Saving to CSV...")
df.to_csv("full_dataset_engineered.csv", index=False)

print("✅ Done! You can now upload 'full_dataset_engineered.csv' to your dashboard.")