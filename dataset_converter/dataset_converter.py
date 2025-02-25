import pandas as pd
from scipy.io import arff
import numpy as np

# Filename
filename = "diabetes"  # Change to your file

# Load the ARFF file
arff_file = f"dataset_converter/{filename}.arff"  # Change to your file
data, meta = arff.loadarff(arff_file)
print("Loaded", arff_file)

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Convert categorical attributes (if any)
for col in df.select_dtypes(['object']).columns:
    df[col] = df[col].astype(str)

# Separate features (X) and target (Y)
X = df.iloc[:, :-1].values  # All columns except last (features)
Y = df.iloc[:, -1].values   # Last column (target)

# Scale features to [0,1] range
X_scaled = np.zeros_like(X, dtype=float)
for j in range(X.shape[1]):
    column = X[:, j].astype(float)
    if column.max() > column.min():  # Avoid division by zero
        X_scaled[:, j] = (column - column.min()) / (column.max() - column.min())
    else:
        X_scaled[:, j] = column

# Convert numeric class labels if necessary
unique_classes = {v: i+1 for i, v in enumerate(sorted(set(Y)))}  # Map classes to integers
Y = [unique_classes[v] for v in Y]

# Format X as Julia matrix-style
X_formatted = "X = [" + "; ".join([" ".join(map(str, row)) for row in X_scaled]) + "]"

# Format Y as Julia vector-style
Y_formatted = "Y = Vector{Any}([" + ", ".join(map(str, Y)) + "])"

# Save to TXT file
with open(f"{filename}.txt", "w") as f:
    f.write(X_formatted + "\n" + Y_formatted)

print(f"Conversion complete! Saved as {filename}.txt")

# Print dataset information
print(f"\nDataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(unique_classes)}")
print(f"Feature range after scaling: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")