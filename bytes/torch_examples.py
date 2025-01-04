# %% expand_as(), view(), example
import torch

x = torch.randn(2, 3)  # Shape: (2, 3)
y = torch.tensor([1.0, 2.0])  # Shape: (2,)

# Reshape y to add a new dimension
y_reshaped = y.view(2, 1)  # Shape: (2, 1)

# Expand y to match the shape of x
y_expanded = y_reshaped.expand_as(x)  # Shape: (2, 3)

# Perform element-wise multiplication
result = x * y_expanded

print("Original x shape:", x.shape)
print("Original y shape:", y.shape)
print("Expanded y shape:", y_expanded.shape)
print("Result shape:", result.shape)

# %%
