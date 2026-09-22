import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def prepare_and_engineer_data(file_path="creditcard.csv"):
    print("1. Loading raw Kaggle dataset...")
    # Load the Kaggle dataset
    df = pd.read_csv(file_path)
    
    print("2. Synthesizing missing behavioral entities (user_id)...")
    # Since the Kaggle dataset lacks users, we randomly assign the ~284,800 
    # transactions to a pool of 5,000 synthetic users to create transaction histories.
    np.random.seed(42)
    synthetic_users = [f"USER_{i:04d}" for i in range(1, 5001)]
    df["user_id"] = np.random.choice(synthetic_users, size=len(df))
    
    # We also synthesize a merchant category for later modeling if needed
    merchants = ["grocery", "retail", "tech", "dining", "travel", "fuel"]
    df["merchant_category"] = np.random.choice(merchants, size=len(df))

    print("3. Handling missing values and applying RobustScaler...")
    # The Kaggle dataset doesn't typically have nulls, but this handles real-world scenarios
    df["Amount"] = df["Amount"].fillna(df["Amount"].median())
    df["Time"] = df["Time"].fillna(0)
    
    # RobustScaler is robust to extreme outliers (highly recommended for transaction amounts)
    scaler_amount = RobustScaler()
    scaler_time = RobustScaler()
    
    df["scaled_amount"] = scaler_amount.fit_transform(df[["Amount"]])
    df["scaled_time"] = scaler_time.fit_transform(df[["Time"]])

    print("4. Engineering Temporal Features (Time since last transaction)...")
    # To calculate user-specific metrics, we MUST sort by user_id and Time
    df = df.sort_values(by=["user_id", "Time"]).reset_index(drop=True)
    
    # Calculate seconds elapsed since this specific user's last transaction
    df["time_since_prev_tx"] = df.groupby("user_id")["Time"].diff().fillna(86400.0) # Default to 24 hours for their first transaction

    print("5. Engineering Behavioral Features (Rolling averages & frequency)...")
    # Calculate the rolling average transaction amount per user (excluding the current transaction)
    user_rolling_mean = df.groupby("user_id")["Amount"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    # If it's the user's first transaction, fill with the global median amount
    global_median = df["Amount"].median()
    df["user_avg_amount"] = user_rolling_mean.fillna(global_median)
    
    # Measure the deviation of the current transaction from their historical average
    df["amount_diff_from_avg"] = df["Amount"] - df["user_avg_amount"]
    # Ratio: e.g., 5.0 means they spent 5x their normal average
    df["amount_ratio_to_avg"] = df["Amount"] / (df["user_avg_amount"] + 1e-5) 
    
    # Calculate transaction frequency in the last hour (3600 seconds) to detect rapid velocity attacks
    def calc_hourly_frequency(group):
        times = group["Time"].values
        counts = np.zeros(len(times))
        for i, t in enumerate(times):
            # Count transactions within the prior 3600 seconds
            counts[i] = np.sum((times[max(0, i - 100):i] >= (t - 3600)) & (times[max(0, i - 100):i] < t))
        return pd.Series(counts, index=group.index)
        
    df["tx_freq_last_hour"] = df.groupby("user_id", group_keys=False).apply(calc_hourly_frequency)

    # Re-sort chronologically for standard model training
    df = df.sort_values(by="Time").reset_index(drop=True)
    
    print("Data Engineering Complete! Features generated successfully.")
    return df

# Execute the pipeline
if __name__ == "__main__":
    processed_df = prepare_and_engineer_data("creditcard.csv")
    
    # Display the newly engineered features for the first 5 rows
    display_cols = ["user_id", "Amount", "scaled_amount", "time_since_prev_tx", "user_avg_amount", "amount_ratio_to_avg", "tx_freq_last_hour", "Class"]
    print("\nPreview of Engineered Data:")
    print(processed_df[display_cols].head())