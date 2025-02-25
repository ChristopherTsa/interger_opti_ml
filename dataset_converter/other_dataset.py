import pandas as pd
import numpy as np

# Load the ecoli data
df = pd.read_csv("dataset_converter/ecoli/ecoli.data", 
                 delim_whitespace=True,  # Use whitespace as delimiter
                 header=None)  # No header in file

# Remove sequence name (first column) and separate features and target
X = df.iloc[:, 1:-1].values  # Columns 1 to -2 (features)
y_labels = df.iloc[:, -1].values  # Last column (target)

# Convert string labels to numeric (1-based)
unique_classes = sorted(set(y_labels))
class_to_num = {label: i+1 for i, label in enumerate(unique_classes)}
y = np.array([class_to_num[label] for label in y_labels])

# Scale features to [0,1] range
X_scaled = np.zeros_like(X, dtype=float)
for j in range(X.shape[1]):
    column = X[:, j].astype(float)
    if column.max() > column.min():  # Avoid division by zero
        X_scaled[:, j] = (column - column.min()) / (column.max() - column.min())
    else:
        X_scaled[:, j] = column

# Format X as Julia matrix-style
X_formatted = "X = [" + "; ".join([" ".join(map(str, row)) for row in X_scaled]) + "]"

# Format y as Julia vector-style
y_formatted = "Y = Vector{Any}([" + ", ".join(map(str, y)) + "])"

# Save to TXT file
with open("ecoli.txt", "w") as f:
    f.write(X_formatted + "\n" + y_formatted)

# Print dataset information
print(f"Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(unique_classes)}")
print(f"Classes: {dict(zip(unique_classes, range(1, len(unique_classes)+1)))}")
print(f"Feature range after scaling: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")