import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
EF = pd.read_csv('../datasets/earthquake_alert_balanced_dataset.csv')


def augment_earthquake_data(df, n_augmented=200):
    # Numerical columns used for augmentation
    numerical_cols = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']

    # Check if required columns exist
    missing_cols = [col for col in numerical_cols + ['alert'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # Remove rows with missing values
    df = df.dropna(subset=numerical_cols + ['alert'])

    # Scale numerical data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_cols])

    augmented_data = []
    alert_labels = []

    for _ in range(n_augmented):
        # Random row selection
        idx = np.random.randint(0, len(df_scaled))

        # Add Gaussian noise
        noise = np.random.normal(loc=0, scale=0.1, size=df_scaled.shape[1])

        augmented_row = df_scaled[idx] + noise

        augmented_data.append(augmented_row)

        # Keep corresponding label
        alert_labels.append(df.iloc[idx]['alert'])

    # Convert back to original scale
    augmented_scaled = np.array(augmented_data)
    augmented_original = scaler.inverse_transform(augmented_scaled)

    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_original, columns=numerical_cols)

    # Prevent negative values
    augmented_df[numerical_cols] = augmented_df[numerical_cols].clip(lower=0)

    # Add labels
    augmented_df['alert'] = alert_labels

    return augmented_df


# Generate augmented data
augmented_df = augment_earthquake_data(EF, n_augmented=200)

# Display sample
print(augmented_df.head())

# Save to CSV
augmented_df.to_csv('../datasets/augmented_earthquake_data.csv', index=False)

print("Augmented dataset saved successfully.")