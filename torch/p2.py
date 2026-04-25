import torch

# Simple 2-layer example


W1 = torch.randn(4, 3, requires_grad=True)
W2 = torch.randn(1, 4, requires_grad=True)


x = torch.randn(3)  # input

# Forward pass
h = (W1 @ x).relu()  # hidden layer
out = W2 @ h           # output

loss = out.sum()
loss.backward()    # gradients flow back!

print(W1.grad)   # torch.Size([4, 3])
print(W2.grad) 