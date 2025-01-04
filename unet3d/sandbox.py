import torch

torch.manual_seed(1)

# Example tensors
x = torch.randint(0, 10, (2, 3, 4))  # Shape: (2, 3, 4)
print(f"x: {x}")
y = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Shape: (4,)
print(f"y: {y}")

# Expand y to match the shape of x
y_expanded = y.expand_as(x)
print(f"y_expanded: {y_expanded}")
# Perform element-wise multiplication
result = x * y_expanded
print(f"result: {result}")

print("Original x shape:", x.shape)
print("Original y shape:", y.shape)
print("Expanded y shape:", y_expanded.shape)
print("Result shape:", result.shape)
