# poison_data.py
import pandas as pd
import numpy as np

def poison_dataset(df, target_col, percent_poisoned):
    """
    Takes a dataframe and flips a percentage of the majority class labels (0) to the minority class (1).

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target variable column.
        percent_poisoned (float): The percentage of the majority class to poison (e.g., 2.0 for 2%).

    Returns:
        pd.DataFrame: The poisoned dataframe.
    """
    # Separate the majority (non-fraud) and minority (fraud) classes
    df_majority = df[df[target_col] == 0].copy()
    df_minority = df[df[target_col] == 1].copy()

    # Determine how many majority samples to poison
    num_to_poison = int(len(df_majority) * (percent_poisoned / 100.0))

    if num_to_poison == 0:
        print(f"Warning: For {percent_poisoned}%, the number of samples to poison is 0. Returning original df.")
        return df

    print(f"Poisoning {num_to_poison} samples for the {percent_poisoned}% level...")

    # Randomly select indices to poison
    poison_indices = np.random.choice(df_majority.index, num_to_poison, replace=False)

    # Flip the labels for the selected indices
    df_majority.loc[poison_indices, target_col] = 1

    # Combine the now-poisoned data with the original minority class data
    df_poisoned = pd.concat([df_majority, df_minority])

    # Shuffle the dataset to mix the poisoned records
    return df_poisoned.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    print("Loading clean dataset (transactions_2022.csv)...")
    clean_df = pd.read_csv('data/v0/transactions_2022.csv')

    poison_levels = [2, 8, 20]

    for level in poison_levels:
        print(f"\n--- Generating dataset for {level}% poisoning ---")
        poisoned_df = poison_dataset(clean_df, 'Class', float(level))

        output_filename = f'data/v0/poisoned_{level}_percent.csv'
        poisoned_df.to_csv(output_filename, index=False)
        print(f"Successfully saved poisoned data to {output_filename}")

