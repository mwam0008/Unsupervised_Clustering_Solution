"""
utils.py - Visualization helpers for Mall Customer Segmentation App
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

PALETTE = 'colorblind'


def plot_pairplot(df: pd.DataFrame):
    """Pairplot of Age, Annual_Income, Spending_Score."""
    try:
        fig = sns.pairplot(df[['Age', 'Annual_Income', 'Spending_Score']]).fig
        fig.suptitle('Pairplot of Customer Features', y=1.02, fontweight='bold')
        return fig
    except Exception as e:
        logging.error(f"Pairplot error: {e}")
        raise


def plot_gender_distribution(df: pd.DataFrame):
    """Pie chart of gender split."""
    try:
        counts = df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=['#2196F3', '#E91E63'], startangle=90)
        ax.set_title('Gender Distribution', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Gender dist error: {e}")
        raise


def plot_feature_distributions(df: pd.DataFrame):
    """Histograms for Age, Income, Spending Score."""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, col, color in zip(axes,
            ['Age', 'Annual_Income', 'Spending_Score'],
            ['#2196F3', '#4CAF50', '#FF9800']):
            axes_i = ax
            axes_i.hist(df[col], bins=20, color=color, edgecolor='white', alpha=0.85)
            axes_i.set_title(f'{col} Distribution', fontweight='bold')
            axes_i.set_xlabel(col)
            axes_i.set_ylabel('Count')
            axes_i.axvline(df[col].mean(), color='red', linestyle='--',
                           label=f'Mean: {df[col].mean():.1f}')
            axes_i.legend(fontsize=8)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Feature dist error: {e}")
        raise


def plot_clusters_2d(df: pd.DataFrame, x: str, y: str, centers=None, title=''):
    """Scatter plot of clusters with optional cluster centers."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = sns.color_palette(PALETTE, n_colors=df['Cluster'].nunique())
        for cluster_id in sorted(df['Cluster'].unique()):
            subset = df[df['Cluster'] == cluster_id]
            ax.scatter(subset[x], subset[y], label=f'Cluster {cluster_id}',
                       color=palette[cluster_id], alpha=0.7, edgecolors='white', s=60)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c='black',
                       s=200, marker='X', zorder=5, label='Centroids')
        ax.set_xlabel(x, fontweight='bold')
        ax.set_ylabel(y, fontweight='bold')
        ax.set_title(title or f'K-Means Clusters: {x} vs {y}', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"2D cluster plot error: {e}")
        raise


def plot_elbow(wss_df: pd.DataFrame, optimal_k: int = None):
    """Elbow curve — WCSS vs number of clusters."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(wss_df['cluster'], wss_df['WCSS_Score'],
                marker='o', color='#2196F3', linewidth=2, markersize=7)
        if optimal_k:
            best_wcss = wss_df[wss_df['cluster'] == optimal_k]['WCSS_Score'].values[0]
            ax.axvline(optimal_k, color='red', linestyle='--',
                       label=f'Optimal k = {optimal_k}')
            ax.scatter([optimal_k], [best_wcss], color='red', s=150, zorder=5)
        ax.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax.set_ylabel('WCSS Score (Inertia)', fontweight='bold')
        ax.set_title('Elbow Method — Find Optimal k', fontweight='bold')
        ax.grid(True, alpha=0.3)
        if optimal_k:
            ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Elbow plot error: {e}")
        raise


def plot_silhouette(sil_df: pd.DataFrame, optimal_k: int = None):
    """Silhouette score vs number of clusters."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#4CAF50' if s == sil_df['Silhouette_Score'].max()
                  else '#2196F3' for s in sil_df['Silhouette_Score']]
        bars = ax.bar(sil_df['cluster'], sil_df['Silhouette_Score'],
                      color=colors, edgecolor='white')
        for bar, val in zip(bars, sil_df['Silhouette_Score']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
        if optimal_k:
            ax.axvline(optimal_k, color='red', linestyle='--',
                       label=f'Best k = {optimal_k}', alpha=0.7)
            ax.legend()
        ax.set_xlabel('Number of Clusters (k)', fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontweight='bold')
        ax.set_title('Silhouette Method — Higher is Better', fontweight='bold')
        ax.set_ylim(0, sil_df['Silhouette_Score'].max() + 0.05)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Silhouette plot error: {e}")
        raise


def plot_cluster_profiles(df: pd.DataFrame, features: list):
    """Bar charts showing mean feature values per cluster."""
    try:
        cluster_means = df.groupby('Cluster')[features].mean().reset_index()
        fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
        if len(features) == 1:
            axes = [axes]
        palette = sns.color_palette(PALETTE, n_colors=len(cluster_means))
        for ax, feat in zip(axes, features):
            bars = ax.bar(cluster_means['Cluster'].astype(str),
                          cluster_means[feat], color=palette, edgecolor='white')
            for bar, val in zip(bars, cluster_means[feat]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
            ax.set_title(f'Avg {feat} per Cluster', fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(feat)
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Cluster profile error: {e}")
        raise


def plot_cluster_sizes(df: pd.DataFrame):
    """Bar chart of how many customers per cluster."""
    try:
        counts = df['Cluster'].value_counts().sort_index()
        palette = sns.color_palette(PALETTE, n_colors=len(counts))
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(counts.index.astype(str), counts.values,
                      color=palette, edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontweight='bold')
        ax.set_title('Customers Per Cluster', fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Cluster size error: {e}")
        raise


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of numeric features."""
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[['Age', 'Annual_Income', 'Spending_Score']].corr(),
                    annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Heatmap error: {e}")
        raise
