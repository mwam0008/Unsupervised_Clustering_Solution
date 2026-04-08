"""
model.py - ML logic for Mall Customer Segmentation (K-Means Clustering)
Covers: KMeans training, Elbow method (WCSS), Silhouette scores, cluster profiling
"""

import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FEATURE_OPTIONS = {
    "Income + Spending (2D - visualizable)": ['Annual_Income', 'Spending_Score'],
    "Age + Income + Spending (3D)": ['Age', 'Annual_Income', 'Spending_Score'],
    "Age + Spending (2D - visualizable)": ['Age', 'Spending_Score'],
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the mall customers dataset."""
    try:
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Load failed: {e}")
        raise


def train_kmeans(df: pd.DataFrame, features: list, n_clusters: int,
                 init='k-means++', n_init='auto', max_iter=300) -> tuple:
    """
    Train a KMeans model on the given features.
    Returns: model, dataframe with Cluster column added
    """
    try:
        logging.info(f"Training KMeans: k={n_clusters}, features={features}")
        kmodel = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )
        kmodel.fit(df[features])
        df = df.copy()
        df['Cluster'] = kmodel.labels_
        logging.info("KMeans trained.")
        return kmodel, df
    except Exception as e:
        logging.error(f"KMeans training failed: {e}")
        raise


def elbow_method(df: pd.DataFrame, features: list, k_range=range(2, 11)) -> pd.DataFrame:
    """
    Run WCSS (inertia) for a range of k values — used to plot the Elbow curve.
    Returns a DataFrame with columns: cluster, WCSS_Score
    """
    try:
        logging.info("Running Elbow Method...")
        K, WCSS = [], []
        for i in k_range:
            km = KMeans(n_clusters=i, init='k-means++', n_init='auto',
                        random_state=42).fit(df[features])
            WCSS.append(km.inertia_)
            K.append(i)
        result = pd.DataFrame({'cluster': K, 'WCSS_Score': WCSS})
        logging.info("Elbow method complete.")
        return result
    except Exception as e:
        logging.error(f"Elbow method failed: {e}")
        raise


def silhouette_method(df: pd.DataFrame, features: list, k_range=range(2, 11)) -> pd.DataFrame:
    """
    Run Silhouette scores for a range of k values.
    Returns a DataFrame with columns: cluster, Silhouette_Score
    """
    try:
        logging.info("Running Silhouette Method...")
        K, SS = [], []
        for i in k_range:
            km = KMeans(n_clusters=i, init='k-means++', n_init='auto',
                        random_state=42).fit(df[features])
            sil = silhouette_score(df[features], km.labels_)
            K.append(i)
            SS.append(round(sil, 4))
        result = pd.DataFrame({'cluster': K, 'Silhouette_Score': SS})
        logging.info("Silhouette method complete.")
        return result
    except Exception as e:
        logging.error(f"Silhouette method failed: {e}")
        raise


def find_optimal_k(sil_df: pd.DataFrame) -> int:
    """Return the k with the highest silhouette score."""
    return int(sil_df.loc[sil_df['Silhouette_Score'].idxmax(), 'cluster'])


def profile_clusters(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Return mean stats per cluster for interpretation."""
    try:
        cols = features + ['Cluster']
        if 'Gender' in df.columns:
            df_temp = df.copy()
            df_temp['Gender_Male'] = (df_temp['Gender'] == 'Male').astype(int)
            profile = df_temp[cols + ['Gender_Male']].groupby('Cluster').agg(
                ['mean', 'count']
            )
        else:
            profile = df[cols].groupby('Cluster').agg(['mean', 'count'])
        return profile.round(2)
    except Exception as e:
        logging.error(f"Profiling failed: {e}")
        raise


def save_model(model, path='Cluster_Model.pkl'):
    """Save KMeans model with pickle."""
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Save failed: {e}")
        raise


def load_model(path='Cluster_Model.pkl'):
    """Load KMeans model from pickle."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Load failed: {e}")
        raise


def predict_cluster(model, income: float, spending: float,
                    age: float = None, features: list = None) -> int:
    """Predict which cluster a new customer belongs to."""
    try:
        if features and len(features) == 3:
            input_data = [[age, income, spending]]
        else:
            input_data = [[income, spending]]
        cluster = model.predict(input_data)[0]
        logging.info(f"Predicted cluster: {cluster}")
        return int(cluster)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise
