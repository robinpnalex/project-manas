"""
SVM Kernel Support Audit
=========================
Uses Representer Theorem to attribute outlier classification to dual coefficients α_i
and support vectors for RBF kernel SVM.

Decision function: f(x) = Σ α_i y_i K(x_i, x) + b
where K(x_i, x) = exp(-γ ||x_i - x||²)
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import json


def rbf_kernel(x1, x2, gamma=1.0):
    """RBF kernel: K(x1, x2) = exp(-γ ||x1 - x2||²)"""
    diff = np.atleast_2d(x1) - np.atleast_2d(x2)
    return np.exp(-gamma * np.sum(diff ** 2, axis=-1))


def extract_svm_attribution(X_query):
    """Main function to extract SVM attribution using Representer Theorem."""
    iris = load_iris()
    X = iris.data[iris.target != 2]
    y = iris.target[iris.target != 2]
    
    svm = SVC(kernel='rbf', C=1.0)
    svm.fit(X, y)
    
    alpha = svm.dual_coef_[0]
    support_vectors = svm.support_vectors_
    intercept = svm.intercept_[0]
    gamma_str = svm.gamma
    gamma = 1.0 / (X.shape[1] * np.var(X)) if gamma_str == 'scale' or gamma_str == 'auto' else gamma_str
    
    sv_indices = svm.support_
    y_train = y[sv_indices]
    
    n_support = len(support_vectors)
    contributions = np.zeros(n_support)
    
    for i in range(n_support):
        k_val = rbf_kernel(support_vectors[i], X_query, gamma)
        contributions[i] = alpha[i] * y_train[i] * k_val
    
    total_decision = np.sum(contributions) + intercept
    predicted_class = 1 if total_decision >= 0 else 0
    
    attribution_results = {
        "query_point": X_query.tolist(),
        "decision_value": float(total_decision),
        "predicted_class": int(predicted_class),
        "intercept": float(intercept),
        "n_support_vectors": n_support,
        "support_vector_contributions": [
            {
                "sv_index": int(i),
                "alpha": float(alpha[i]),
                "y_i": int(y_train[i]),
                "contribution": float(contributions[i]),
                "x": support_vectors[i].tolist()
            }
            for i in range(n_support)
        ]
    }
    
    print(f"SVM Attribution Results:")
    print(f"  Query: {X_query}")
    print(f"  Decision value: {total_decision:.4f}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Support vectors: {n_support}")
    print(f"  α values: {alpha}")
    
    return attribution_results


if __name__ == "__main__":
    X_sample = load_iris().data[load_iris().target != 2][0]
    result = extract_svm_attribution(X_sample)
    print("\n=== Detailed Attribution ===")
    print(json.dumps(result, indent=2))