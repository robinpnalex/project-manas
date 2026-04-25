# Multi-Paradigm Attribution Audit

## What is this?

A toolkit to understand why machine learning models make their decisions. We focus on three different models and explain their predictions using math.

---

## Task A: SVM Kernel Support Audit

### What is an SVM?
- **SVM** = Support Vector Machine = a分类器 (classifier)
- It draws a boundary to separate different classes

### The Problem
- This SVM uses an **RBF kernel** - a special math trick
- The trick maps data to infinite dimensions (like a mirror universe)
- We can't see the actual boundary because it's in infinite dimensions

### The Solution
- Use **Representer Theorem** - a math rule that says:
  > Any solution can be written as a sum of training points
  
- Formula: `f(x) = Σ αᵢ × yᵢ × K(xᵢ, x) + b`
  - αᵢ = importance of each training point
  - yᵢ = correct answer
  - K() = RBF kernel (similarity measure)

### Why this matters
- We can see which training points (support vectors) influenced the decision
- This is like knowing which witnesses influenced a judge's verdict

---

## Task B: DBSCAN vs HDBSCAN Persistence Paradox

### What is clustering?
- **Clustering** = grouping similar points together automatically
- No one tells the computer what groups to find

### The Problem
- **DBSCAN** = Density-Based Clustering
  - Uses a fixed radius (ε) to find density
  - Problem: if clusters have different densities, it misses some
  
- **HDBSCAN** = Hierarchical DBSCAN
  - Looks at multiple radii at once
  - Finds clusters that density-based methods miss

### The Key Concept: Persistence
- Think of persistence as "how stable is this cluster?"
- Calculate: λ = 1/ε where ε = radius
- Higher λ = cluster persists through more scales = more real

### Why this matters
- In mixed-density data (dense + sparse clusters), DBSCAN fails while HDBSCAN succeeds
- Example: Finding both a dense city and sparse rural areas

---

## Task C: AlexNet Spatial-Semantic Bridge

### What is AlexNet?
- A neural network for image recognition
- Has layers: Conv1→Conv2→...→Conv5→FC6→FC7→output

### The Problem
- **FC6** = Fully Connected layer with 4096 neurons
- It lost all spatial information (where things are in the image)
- We see "dead neurons" (always 0) but don't know what image part caused it

### The Solution
- Trace backwards through the network:
  1. FC6 neuron → MaxPool index (which 2x2 region)
  2. MaxPool → Conv5 receptive field (the original image patch)

### The Architecture Path
```
Conv5 (13×13×256) → MaxPool (6×6×256) → FC6 (4096)
```

- Each FC6 neuron connects to one 2x2 patch in Conv5
- Back-project to find the image region that "woke up" the neuron

---

## Summary Table

| Task | Model | Problem | Solution |
|------|-------|---------|----------|
| A | SVM (RBF) | Can't see infinite-dim boundary | Representer Theorem |
| B | DBSCAN | Misses varying-density clusters | HDBSCAN persistence scores |
| C | AlexNet | FC6 lost spatial info | Reverse max-pool mapping |

---

## Files Included

- `svm_attribution.py` - Extract α coefficients from SVM
- `persistence_analysis.py` - Compare DBSCAN vs HDBSCAN
- `alexnet_spatial_mapping.py` - Map FC6 neurons back to image
- `generate_pdf.py` - Create visual report
- `persistence_report.pdf` - Visual comparison
- `counterfactual.json` - How to flip a decision (math)