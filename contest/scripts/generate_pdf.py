"""
PDF Report Generator for Persistence Analysis
===============================================
HDBSCAN Stability vs DBSCAN Epsilon Analysis Report
"""

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import json
import warnings
warnings.filterwarnings('ignore')


def generate_mixed_density_data():
    """Generate mixed-density dataset."""
    np.random.seed(42)
    
    dense_cluster = np.random.normal(loc=[0, 0], scale=0.3, size=(50, 2))
    sparse_cluster = np.random.normal(loc=[8, 8], scale=0.8, size=(30, 2))
    noise = np.random.uniform(low=-5, high=15, size=(40, 2))
    
    return np.vstack([dense_cluster, sparse_cluster, noise])


def run_analysis():
    """Run full DBSCAN vs HDBSCAN analysis."""
    X = generate_mixed_density_data()
    
    eps_values = np.linspace(0.1, 2.0, 20)
    dbscan_clusters = []
    dbscan_noise = []
    
    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        dbscan_clusters.append(n_clusters)
        dbscan_noise.append(n_noise)
    
    hdbscan = HDBSCAN(min_cluster_size=5)
    hdbscan_labels = hdbscan.fit_predict(X)
    hdbscan_n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    hdbscan_n_noise = list(hdbscan_labels).count(-1)
    
    try:
        persistence_scores = hdbscan.probabilities_
    except:
        persistence_scores = np.ones(len(X))
    
    return X, eps_values, dbscan_clusters, dbscan_noise, hdbscan_n_clusters, hdbscan_n_noise, persistence_scores, hdbscan_labels


def create_persistence_pdf(output_path):
    """Generate PDF report."""
    X, eps_values, dbscan_clusters, dbscan_noise, hdbscan_n_clusters, hdbscan_n_noise, persistence_scores, hdbscan_labels = run_analysis()
    
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    
    fig1 = plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=hdbscan_labels, cmap='viridis', alpha=0.7)
    plt.title('HDBSCAN Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    pdf.savefig(fig1)
    plt.close()
    
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(eps_values, dbscan_clusters, 'b-o', label='Clusters')
    plt.plot(eps_values, dbscan_noise, 'r-s', label='Noise')
    plt.axhline(y=hdbscan_n_clusters, color='g', linestyle='--', label=f'HDBSCAN clusters={hdbscan_n_clusters}')
    plt.axhline(y=hdbscan_n_noise, color='m', linestyle='--', label=f'HDBSCAN noise={hdbscan_n_noise}')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Count')
    plt.title('DBSCAN: Clusters & Noise vs Epsilon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    pdf.savefig(fig2)
    plt.close()
    
    fig3 = plt.figure(figsize=(10, 6))
    eps_critical = None
    for i, eps in enumerate(eps_values):
        if dbscan_clusters[i] == hdbscan_n_clusters:
            eps_critical = eps
            break
    
    if eps_critical:
        plt.axvline(x=eps_critical, color='r', linestyle='--', label=f'Critical ε = {eps_critical:.2f}')
    plt.hist(persistence_scores, bins=50)
    plt.xlabel('Persistence Score (probability)')
    plt.ylabel('Count')
    plt.title('HDBSCAN Cluster Stability Scores')
    plt.legend()
    pdf.savefig(fig3)
    plt.close()
    
    pdf.close()
    
    return {
        'critical_epsilon': float(eps_critical) if eps_critical else None,
        'critical_lambda': float(1.0/eps_critical) if eps_critical else None,
        'hdbscan_n_clusters': hdbscan_n_clusters,
        'hdbscan_n_noise': hdbscan_n_noise
    }


if __name__ == "__main__":
    results = create_persistence_pdf('/home/phoenix/robin/persistence_report.pdf')
    print(f"PDF created: {results}")