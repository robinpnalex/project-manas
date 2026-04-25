import torch
from torch.utils.data import DataLoader, TensorDataset

# Wrap your data
X = torch.randn(1000, 20)  # 1000 samples
y = torch.randint(0, 2, (1000,))

dataset = TensorDataset(X, y)
loader  = DataLoader(dataset,
                      batch_size=32,
                      shuffle=True)

# Iterate in training loop
for X_batch, y_batch in loader:
    print(X_batch.shape)  # torch.Size([32, 20])