import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None

    def train(self, X, y):
        """Gradient Descent for single or multiple features."""
        # If X is 1D, reshape to 2D column vector
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        X_b = np.hstack([np.ones((n, 1)), X])
        self.weights = np.zeros(X_b.shape[1])

        for _ in range(self.iterations):
            y_pred = X_b @ self.weights
            gradient = (-2 / n) * (X_b.T @ (y - y_pred))
            self.weights -= self.lr * gradient

    def predict(self, X):
        if isinstance(X, (int, float)):
            X = np.array([[X]])
        elif X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        X_b = np.hstack([np.ones((n, 1)), X])
        return X_b @ self.weights


# --- Testing: single feature ---
if __name__ == "__main__":
    print("=== Single Feature: y = 2x + 1 ===")
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 7, 9, 11])

    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.train(X, y)

    print(f"Weights: intercept={model.weights[0]:.2f}, w1={model.weights[1]:.2f}")
    print(f"Prediction for x=10: {model.predict(10)[0]:.2f}")

    # --- Testing: multiple features ---
    print("\n=== Multiple Features: y = 3*x1 + 2*x2 + 5 ===")
    np.random.seed(42)
    n = 100
    x1 = np.random.uniform(1, 50, n)
    x2 = np.random.uniform(1, 30, n)
    X_multi = np.column_stack([x1, x2])
    y_multi = 3 * x1 + 2 * x2 + 5

    model2 = LinearRegression(learning_rate=0.0001, iterations=5000)
    model2.train(X_multi, y_multi)

    print(f"Weights: intercept={model2.weights[0]:.2f}, w1={model2.weights[1]:.2f}, w2={model2.weights[2]:.2f}")
    print(f"Prediction for x1=10, x2=5: {model2.predict(np.array([[10, 5]]))[0]:.2f}")
    print(f"Expected: {3*10 + 2*5 + 5}")
