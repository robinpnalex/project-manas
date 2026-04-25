import torch

x = torch.tensor(3.0, requires_grad=True)
#basically requires grad tracks whats being done to tensor x


# Define some function
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1
#here since reqgrad is true torch now knows that x was squared
#if false then it would have just put value of x and gotten y value


# Compute gradient dy/dx
y.backward()

print(x.grad)   