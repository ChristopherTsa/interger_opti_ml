from sklearn.datasets import load_breast_cancer
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Normalize the features to [0,1] range
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Format X as Julia matrix-style
X_formatted = "X=[" + "; ".join([" ".join(map(str, row)) for row in X_normalized]) + "]"

# Format y as Julia vector-style (add 1 to make classes 1-based instead of 0-based)
y_formatted = "Y=Vector{Any}([" + ", ".join(map(str, y + 1)) + "])"

# Save to TXT file
with open("breast_cancer.txt", "w") as f:
    f.write(X_formatted + "\n" + y_formatted)

print("Conversion complete! Saved as breast_cancer.txt")