# Task A: SVM Kernel Support Audit - Solution Guide

## The Training Snippet Explained

```python
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load iris dataset (binary classification: classes 0 and 1)
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

svm = SVC(kernel='rbf', C=1.0)
svm.fit(X, y)
```

### What each line does:

| Code | Meaning |
|------|---------|
| `load_iris()` | Load the famous iris flower dataset |
| `X = iris.data[iris.target != 2]` | Get only 2 classes (setosa vs versicolor), drop virginica |
| `y = iris.target[iris.target != 2]` | Get labels: 0 or 1 |
| `SVC(kernel='rbf')` | Create SVM with RBF kernel |
| `svm.fit(X, y)` | "Train" the model - find the decision boundary |

---

## The Actual Math

### What the RBF Kernel Does

The RBF kernel is a **similarity function**:

$$K(x, x') = \exp\left(-\gamma \cdot ||x - x'||^2\right)$$

- Output is between 0 and 1
- If x = x', output = 1 (identical)
- If x is far from x', output → 0 (completely different)

### The Infinite-Dimensional Problem

The "kernel trick" implicitly maps data to infinite dimensions:

$$\phi(x) \cdot \phi(x') = K(x, x')$$

We never compute φ(x) directly - we just compute similarity!

---

## Representer Theorem

### The Key Insight

> Any solution to an SVM can be written as a **weighted sum of training points**.

### The Decision Function

$$f(x) = \sum_{i=1}^{n} \alpha_i \cdot y_i \cdot K(x_i, x) + b$$

| Symbol | Meaning |
|--------|---------|
| αᵢ | **Dual coefficient** - importance of training point i |
| yᵢ | The correct label for training point i (+1 or -1) |
| K(xᵢ, x) | Similarity between training point and query point |
| b | Bias term - shifts the decision boundary |

---

## Interpreting αᵢ for Decision Attribution

### What αᵢ Tells Us

- **αᵢ > 0**: This point pushes toward class +1
- **αᵢ < 0**: This point pushes toward class -1
- **αᵢ = 0**: This point is not a support vector (irrelevant)

### Which Points Matter?

Only points with **αᵢ > 0** are **support vectors** - they define the boundary.

### Computing the Decision

For a new point x, we compute:

```
decision = Σ (αᵢ × yᵢ × K(xᵢ, x)) + b
```

Each support vector contributes:
- Its α value (importance)
- Its label yᵢ (which class it supports)
- Its similarity K to the query point

---

## Example: The Iris Dataset

The iris dataset has 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

For binary classification (setosa = 0, versicolor = 1):

```
Training: 100 points (50 each class)
Support vectors: ~10-20 points (not all points matter)
```

### Decision Process

1. For query point x, compute similarity to each support vector
2. Multiply by αᵢ and yᵢ
3. Sum all contributions
4. Add bias b
5. If result > 0 → class 1, else class 0

---

## The Attribution Output

When we run `svm_attribution.py`, we get:

```json
{
  "query_point": [5.1, 3.5, 1.4, 0.2],
  "decision_value": 2.34,
  "predicted_class": 0,
  "intercept": -0.5,
  "n_support_vectors": 12,
  "support_vector_contributions": [
    {
      "sv_index": 0,
      "alpha": 0.5,
      "y_i": 0,
      "contribution": 0.42,
      "x": [5.1, 3.5, 1.4, 0.2]
    },
    ...
  ]
}
```

### Each Field:

| Field | Meaning |
|-------|---------|
| `query_point` | The point we're classifying |
| `decision_value` | The raw f(x) output |
| `predicted_class` | Final prediction (0 or 1) |
| `intercept` | The bias term b |
| `n_support_vectors` | How many points matter |
| `contribution` | How much that point influenced the decision |

---

## Key Takeaways

1. **αᵢ is the importance weight** - higher = more influential
2. **Only support vectors matter** - points with αᵢ > 0
3. **The RBF kernel** measures similarity in infinite dimensions
4. **No linear approximation needed** - we use kernel directly
5. **Each αᵢ contributes** αᵢ × yᵢ × K(xᵢ, x) to the final decision

This is mathematically exact - no approximation!