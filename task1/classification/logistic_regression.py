import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(y, y_hat):
    """Binary cross-entropy loss"""
    epsilon = 1e-15  # avoid log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def gradient_descent(X, y, lr=0.1, epochs=1000):
    """
    Trains logistic regression using gradient descent.
    Returns: coefficients [intercept, w1, w2, ...], cost history
    """
    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack([ones, X])
    n = X_b.shape[0]

    coefficients = np.zeros(X_b.shape[1])
    cost_history = []

    for i in range(epochs):
        z = X_b @ coefficients
        y_hat = sigmoid(z)

        gradient = (1 / n) * (X_b.T @ (y_hat - y))
        coefficients = coefficients - lr * gradient

        cost = compute_cost(y, y_hat)
        cost_history.append(cost)

    return coefficients, cost_history


def predict(X, coefficients, threshold=0.5):
    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack([ones, X])
    probabilities = sigmoid(X_b @ coefficients)
    return (probabilities >= threshold).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred) * 100


# ── Load Candy Dataset ──

df = pd.read_csv("candy-data.csv")
df = df.dropna()

feature_cols = ["fruity", "caramel", "peanutyalmondy", "nougat", "crispedricewafer",
                "hard", "bar", "pluribus", "sugarpercent", "pricepercent"]
target_col = "chocolate"

X = df[feature_cols].values
y = df[target_col].values

# Train
coefficients, cost_history = gradient_descent(X, y, lr=0.1, epochs=2000)
y_pred = predict(X, coefficients)

print(f"Target: {target_col}")
print(f"Features: {feature_cols}")
print(f"\nCoefficients:")
print(f"  Intercept: {coefficients[0]:.4f}")
for name, coef in zip(feature_cols, coefficients[1:]):
    print(f"  {name:20s} {coef:.4f}")
print(f"\nFinal Cost: {cost_history[-1]:.4f}")
print(f"Accuracy:   {accuracy(y, y_pred):.2f}%")

# ── Plots ──

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: coefficient importance
ax = axes[0]
ax.barh(feature_cols, coefficients[1:], color='steelblue')
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Importance for Predicting Chocolate')

# Right: cost over epochs
ax = axes[1]
ax.plot(cost_history, color='purple')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cost (Binary Cross-Entropy)')
ax.set_title('Cost over Training')

plt.tight_layout()
plt.show()
