# pip install ucimlrepo


# Get the
"""
from ucimlrepo import fetch_ucirepo

# fetch dataset
land_mines = fetch_ucirepo(id=763)

# data (as pandas dataframes)
X = land_mines.data.features
y = land_mines.data.targets

# metadata
print(land_mines.metadata)

# variable information
print(land_mines.variables)
"""

# Write data to CSV

# pip install ucimlrepo pandas

from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
land_mines = fetch_ucirepo(id=763)

# Extract features and targets
X = land_mines.data.features
y = land_mines.data.targets

# Combine features and targets into one DataFrame
df = pd.concat([X, y], axis=1)

# Save to CSV
df.to_csv("land_mines.csv", index=False)

print("Saved to land_mines.csv")
