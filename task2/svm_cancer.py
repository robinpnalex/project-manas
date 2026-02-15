import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# load data
df = pd.read_csv("gene_expression.csv")

print("Shape:", df.shape)
print(df["Cancer Present"].value_counts())

X = df.drop("Cancer Present", axis=1).values
y = df["Cancer Present"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# linear SVM using hinge loss and stochastic gradient descent
class LinearSVM:
    def __init__(self, lr=0.001, lam=0.01, n_iters=500):
        self.lr = lr
        self.lam = lam
        self.n_iters = n_iters

    def fit(self, X, y):
        n, d = X.shape
        y_ = np.where(y == 0, -1, 1)

        self.w = np.zeros(d)
        self.b = 0.0

        for ep in range(self.n_iters):
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y_[idx]

            for i in range(n):
                margin = y_s[i] * (X_s[i] @ self.w + self.b)

                if margin >= 1:
                    self.w -= self.lr * (self.lam * self.w)
                else:
                    self.w -= self.lr * (self.lam * self.w - y_s[i] * X_s[i])
                    self.b += self.lr * y_s[i]

            if (ep+1) % 100 == 0:
                loss = 0.5 * self.lam * (self.w @ self.w) + np.mean(np.maximum(0, 1 - y_ * (X @ self.w + self.b)))
                print(f"  epoch {ep+1}/{self.n_iters}, loss: {loss:.4f}")

    def predict(self, X):
        return (X @ self.w + self.b >= 0).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


svm = LinearSVM(lr=0.001, lam=0.01, n_iters=500)
svm.fit(X_train, y_train)

preds = svm.predict(X_val)
print(f"\nAccuracy: {svm.score(X_val, y_val):.4f}")
print(classification_report(y_val, preds, target_names=["No Cancer", "Cancer"]))
print("Confusion matrix:")
print(confusion_matrix(y_val, preds))
