# Task B: DBSCAN vs HDBSCAN Persistence Paradox - Solution Guide

## Problem Statement

### The Scenario
We have a dataset with **mixed densities** - some clusters are dense (points close together), others are sparse (points spread apart).

### Example
Imagine finding cities on a map:
- **Dense cluster**: A big city (many people in small area)
- **Sparse cluster**: Rural towns (few people spread over large area)

### The Problem with DBSCAN

```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X)
```

**What is DBSCAN?**
- **D**ensity-**B**ased **S**patial **C**lustering of **A**pplications with **N**oise
- Finds clusters based on density

**What's the issue?**
- DBSCAN uses a **fixed radius (ε)**
- It can only find clusters where points are within ε of each other
- If ε = 0.5: too small for sparse clusters (they become noise)
- If ε = 2.0: too big (separate clusters merge)

### Why HDBSCAN Wins

```python
hdbscan = HDBSCAN(min_cluster_size=5)
labels_hdb = hdbscan.fit_predict(X)
```

**What is HDBSCAN?**
- **H**ierarchical **DBSCAN**
- Looks at **multiple radii at once** (like checking all zoom levels)
- Finds clusters that persist across scales

---

## Objective

### What We Need to Do

1. **Map HDBSCAN stability** to DBSCAN ε range
2. **Find λ = 1/ε** where cluster becomes noise
3. **Explain the hierarchy** - why some clusters persist, others don't

### Key Formula

$$\lambda = \frac{1}{\epsilon}$$

- **ε** = radius (how far to look for neighbors)
- **λ** = persistence (how "strong" the cluster is)
- Higher λ = more persistent = more real cluster

---

## Focus: Interpreting Stability and Hierarchy

### What Does "Stability" Mean?

Think of stability as **how real is this cluster?**

| Stability Score | Meaning |
|----------------|---------|
| High (close to 1) | Cluster is very real, exists at many scales |
| Low (close to 0) | Cluster is borderline, might be noise |

### Example: Density Mountain

Imagine density as height of a mountain:
```
          /\
         /  \    ← High density (peak)
        /    \
_______/______\___ ← Low density (base)
```

DBSCAN cuts at ONE height → might miss the peak or base
HDBSCAN looks at ALL heights → sees the whole mountain

### What Does "Hierarchy" Mean?

**Hierarchy** = clusters within clusters

```
Level 1 (low λ):  [   big cluster   ]
Level 2 (high λ): [sub-cluster A][sub-cluster B]
```

- DBSCAN: sees one flat layer
- HDBSCAN: sees nested layers (ierarchy)

### How HDBSCAN Works

1. **Build a density tree** - like a family tree of clusters
2. **Compute persistence** - how long each branch lives
3. **Cut at min_cluster_size** - keep stable clusters

### The Math of Persistence

For each cluster, HDBSCAN computes:

```
persistence = λ = 1/ε_min
```

Where ε_min is the smallest radius where the cluster still exists.

**Example:**
| Cluster | Exists at ε | λ = 1/ε | Stability |
|---------|------------|----------|-----------|
| A | 0.3 to 2.0 | 0.5 to 3.3 | HIGH |
| B | 0.8 to 1.2 | 0.83 to 1.25 | LOW |

Cluster A is more stable because it persists across more scales.

---

## How to Analyze This

### Step 1: Run DBSCAN at Multiple ε Values

```python
for eps in [0.1, 0.5, 1.0, 1.5, 2.0]:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    print(f"eps={eps}: {n_clusters} clusters, {n_noise} noise")
```

### Step 2: Run HDBSCAN

```python
hdbscan = HDBSCAN(min_cluster_size=5)
labels = hdbscan.fit_predict(X)
stability = hdbscan.probabilities_
```

### Step 3: Compare

| ε (DBSCAN) | Clusters Found | Cluster A? | Cluster B? |
|-----------|---------------|-----------|------------|
| 0.1 | 0 | No | No |
| 0.5 | 1 | Yes | **No** ← Missed! |
| 1.0 | 2 | Yes | Yes |
| 1.5 | 2 | Yes | Yes |
| 2.0 | 1 | Yes (merged) | Yes (merged) |

At ε = 0.5, DBSCAN misses Cluster B (sparse cluster)

### Step 4: Find Critical λ

The **critical λ** is where Cluster B becomes noise:

```
λ_critical = 1/0.5 = 2.0
```

Clusters with λ < 2.0 are considered noise by DBSCAN at ε = 0.5.

---

## The Output

When you run `persistence_analysis.py`, you get:

```json
{
  "dbscan_analysis": {
    "eps_range": [0.1, 2.0],
    "results": [
      {"eps": 0.5, "n_clusters": 1, "n_noise": 30},
      {"eps": 1.0, "n_clusters": 2, "n_noise": 10}
    ]
  },
  "hdbscan_analysis": {
    "n_clusters": 2,
    "n_noise": 10,
    "stability_scores": {0: 0.95, 1: 0.88}
  },
  "persistence_mapping": {
    "critical_eps": 0.8,
    "critical_lambda": 1.25
  }
}
```

This tells us:
- At ε < 0.8 (λ > 1.25), DBSCAN misses clusters
- HDBSCAN finds them by checking multiple scales

---

## Key Takeaways

1. **DBSCAN limitation**: Uses fixed ε, misses varying-density clusters
2. **HDBSCAN advantage**: Checks all ε values, finds stability
3. **Stability = persistence**: How long a cluster exists across scales
4. **Hierarchy**: Some clusters contain sub-clusters
5. **λ = 1/ε**: Higher λ = more stable cluster

**The paradox solved:** HDBSCAN finds what DBSCAN misses because it doesn't rely on a single ε value.