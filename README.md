# 🛍️ Mall Customer Segmentation

A Streamlit web app that segments mall customers into groups using **K-Means Clustering** — an unsupervised ML algorithm.

## What This App Does

| Section | What it shows |
|---|---|
| Data Overview | Gender distribution, feature histograms, pairplot, correlation heatmap |
| Find Optimal K | Elbow Method (WCSS) + Silhouette scores to determine best number of clusters |
| Train & Visualize Clusters | Train KMeans, visualize clusters, profile segments, show centroids |
| Predict Customer Segment | Enter new customer details → find which cluster they belong to |

## Goal

Segment 200 mall customers into meaningful groups based on:
- **Annual Income**
- **Spending Score**
- **Age** (optional 3rd feature)

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/mall-customer-segmentation.git
cd mall-customer-segmentation
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
clustering_app/
├── app.py                ← Streamlit web app (4 pages)
├── model.py              ← KMeans training, Elbow, Silhouette, pickle
├── utils.py              ← All charts and visualizations
├── mall_customers.csv    ← Dataset (200 customers, 5 features)
├── requirements.txt      ← Dependencies
└── README.md             ← This file
```

## Key Concepts

- **K-Means Clustering** — groups customers by minimizing within-cluster distance
- **WCSS (Inertia)** — sum of squared distances from each point to its cluster center
- **Elbow Method** — plot WCSS vs k, look for the "elbow" bend
- **Silhouette Score** — measures how well-separated clusters are (-1 to +1, higher is better)
- **k-means++** — smarter initialization that speeds up convergence
- **Cluster Profiling** — understand each segment's average characteristics

## Typical Segments Found (k=5, Income vs Spending)

| Segment | Income | Spending | Description |
|---|---|---|---|
| High Value | High | High | Prime marketing targets |
| Cautious Rich | High | Low | Hard to convert |
| Average | Medium | Medium | Typical shoppers |
| Low Budget | Low | Low | Budget-conscious |
| Impulse Buyers | Low | High | Young, spend beyond means |

## Course

CST2216 — Machine Learning 2: Advanced Models and Emerging Topics
Algonquin College
