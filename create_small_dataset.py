import pandas as pd

# Load full dataset
df = pd.read_csv("creditcard.csv")

# Separate fraud and legit
fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

# Take sample of legit transactions
legit_sample = legit.sample(n=10000, random_state=42)

# Combine
small_df = pd.concat([fraud, legit_sample])

# Save new dataset
small_df.to_csv("creditcard_small.csv", index=False)

print("✅ Small dataset created: creditcard_small.csv")