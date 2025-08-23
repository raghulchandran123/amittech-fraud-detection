# prepare_data.py
import pandas as pd

print("Reading the original dataset...")
df = pd.read_csv('transactions.csv')

# The 'Time' column is the number of seconds from the first transaction
# We can sort by time and split the dataframe in the middle
df = df.sort_values('Time')
split_point = len(df) // 2

df_v0 = df.iloc[:split_point]
df_v1 = df.iloc[split_point:]

# Save the two halves into the respective directories
df_v0.to_csv('data/v0/transactions_2022.csv', index=False)
df_v1.to_csv('data/v1/transactions_2023.csv', index=False)

print("Data successfully split into 'data/v0/transactions_2022.csv' and 'data/v1/transactions_2023.csv'")
