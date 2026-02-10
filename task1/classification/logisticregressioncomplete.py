import numpy as np
import matplotlib.pyplot as plt


def read_csv(filepath):
    raw = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=range(1, 13))
    labels = raw[:, 0]
    features = raw[:, 1:]
    return features, labels


def standardize(X):
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - avg) / std, avg, std


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def log_loss(actual, predicted):
    predicted = np.clip(predicted, 1e-10, 1 - 1e-10)
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))


class LogisticClassifier:
    def __init__(self, lr=0.1, n_iter=2000):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = 0.0
        self.loss_log = []

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.loss_log = []

        for i in range(self.n_iter):
            prob = sigmoid(X @ self.w + self.b)
            cost = log_loss(y, prob)
            self.loss_log.append(cost)

            residual = prob - y
            self.w -= self.lr * (X.T @ residual) / len(y)
            self.b -= self.lr * np.sum(residual) / len(y)

            if i % 250 == 0:
                print(f"Step {i}: loss = {cost:.5f}")

    def predict_prob(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, thresh=0.5):
        return (self.predict_prob(X) >= thresh).astype(int)


def show_training_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()


def show_decision_curve(model, X, y):
    scores = X @ model.w + model.b
    idx = np.argsort(scores)

    plt.figure(figsize=(9, 5))
    plt.scatter(scores[idx], y[idx], alpha=0.35, s=20, label="Data points")
    plt.plot(scores[idx], sigmoid(scores[idx]), color='red', linewidth=2, label="Sigmoid curve")
    plt.xlabel("Score (Xw + b)")
    plt.ylabel("P(chocolate)")
    plt.title("Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, y = read_csv("candy-data.csv")
    X, avg, std = standardize(X)

    model = LogisticClassifier(lr=0.1, n_iter=2000)
    model.fit(X, y)

    predictions = model.predict(X)
    acc = np.mean(predictions == y)
    print(f"\nAccuracy: {acc:.4f}")

    show_training_loss(model.loss_log)
    show_decision_curve(model, X, y)
