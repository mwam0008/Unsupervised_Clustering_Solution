"""
app.py - Streamlit Web App for Mall Customer Segmentation
Using K-Means Clustering with Elbow + Silhouette evaluation
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from model import (
    load_data, train_kmeans, elbow_method, silhouette_method,
    find_optimal_k, profile_clusters, save_model, load_model,
    predict_cluster, FEATURE_OPTIONS,
)
from utils import (
    plot_pairplot, plot_gender_distribution, plot_feature_distributions,
    plot_clusters_2d, plot_elbow, plot_silhouette,
    plot_cluster_profiles, plot_cluster_sizes, plot_correlation_heatmap,
)
from logger import (
    log_app_start, log_page_visit, log_data_loaded,
    log_elbow_method, log_silhouette_method,
    log_model_training, log_model_results, log_model_saved,
    log_model_loaded, log_prediction, log_error, log_warning,
    get_log_contents, get_log_line_count,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# Log app startup once per session
if "app_started" not in st.session_state:
    log_app_start()
    st.session_state["app_started"] = True

st.title("Mall Customer Segmentation")
st.markdown("Group mall customers into segments using **K-Means Clustering** - an unsupervised ML algorithm.")

# ── Load Data ─────────────────────────────────────────────────
DATA_PATH = "mall_customers.csv"

@st.cache_data
def get_data():
    df = load_data(DATA_PATH)
    log_data_loaded(DATA_PATH, df.shape[0], df.shape[1])
    return df

try:
    df = get_data()
except Exception as e:
    log_error("Data loading", e)
    st.error(f"Could not load mall_customers.csv. Error: {e}")
    st.stop()

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choose a section:", [
    "Data Overview",
    "Find Optimal K",
    "Train & Visualize Clusters",
    "Predict Customer Segment",
    "Activity Log",
])

# Log every page visit (only when it changes)
if st.session_state.get("current_section") != section:
    log_page_visit(section)
    st.session_state["current_section"] = section


# ════════════════════════════════════════════════════════════
# SECTION 1 - Data Overview
# ════════════════════════════════════════════════════════════
if section == "Data Overview":
    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers",    df.shape[0])
    col2.metric("Avg Age",            f"{df['Age'].mean():.1f}")
    col3.metric("Avg Income",         f"${df['Annual_Income'].mean():.1f}k")
    col4.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.1f}")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        try:
            st.pyplot(plot_gender_distribution(df))
        except Exception as e:
            log_error("plot_gender_distribution", e)
            st.error("Could not render gender distribution chart.")

    with col2:
        st.subheader("Correlation Heatmap")
        try:
            st.pyplot(plot_correlation_heatmap(df))
        except Exception as e:
            log_error("plot_correlation_heatmap", e)
            st.error("Could not render correlation heatmap.")

    st.subheader("Feature Distributions")
    try:
        st.pyplot(plot_feature_distributions(df))
    except Exception as e:
        log_error("plot_feature_distributions", e)
        st.error("Could not render feature distributions.")

    st.subheader("Pairplot — Relationships Between Features")
    st.markdown("""
    Key observations from the notebook:
    - **Spending Score is high for ages 20–40**, and drops after 40
    - **Income and Spending Score** show clear natural groupings
    """)
    with st.spinner("Generating pairplot..."):
        try:
            st.pyplot(plot_pairplot(df))
        except Exception as e:
            log_error("plot_pairplot", e)
            st.error("Could not render pairplot.")

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())


# ════════════════════════════════════════════════════════════
# SECTION 2 - Find Optimal K
# ════════════════════════════════════════════════════════════
elif section == "Find Optimal K":
    st.header("Find the Optimal Number of Clusters (k)")
    st.markdown("""
    Two methods help us find the best k:
    - **Elbow Method** — look for the "elbow" where WCSS stops dropping sharply
    - **Silhouette Method** — higher score = better separated clusters (range: −1 to +1)
    """)

    feature_label = st.selectbox("Select Features", list(FEATURE_OPTIONS.keys()))
    features      = FEATURE_OPTIONS[feature_label]

    k_min = st.slider("Min clusters to test", 2, 4,  2)
    k_max = st.slider("Max clusters to test", 5, 15, 10)

    if st.button("Run Elbow + Silhouette Analysis"):
        with st.spinner("Running analysis across all k values..."):
            try:
                k_range = range(k_min, k_max + 1)

                log_elbow_method(features, k_min, k_max)
                wss_df = elbow_method(df, features, k_range)

                sil_df    = silhouette_method(df, features, k_range)
                optimal_k = find_optimal_k(sil_df)
                best_sil  = sil_df["Silhouette_Score"].max()
                log_silhouette_method(features, optimal_k, best_sil)

                st.success(f"Analysis complete! Recommended k = **{optimal_k}**")

                col1, col2 = st.columns(2)
                col1.metric("Optimal k (by Silhouette)", optimal_k)
                col2.metric("Best Silhouette Score", f"{best_sil:.4f}")

                st.subheader("Elbow Plot (WCSS)")
                st.markdown("Look for the 'elbow' — the point where the curve bends and flattens.")
                st.pyplot(plot_elbow(wss_df, optimal_k))

                st.subheader("Silhouette Scores")
                st.markdown("The tallest green bar = best k.")
                st.pyplot(plot_silhouette(sil_df, optimal_k))

                st.subheader("Raw Scores Table")
                st.dataframe(wss_df.merge(sil_df, on="cluster").round(4))

                st.session_state["optimal_k"] = optimal_k
                st.session_state["features"]  = features
                st.info(f"Use **k = {optimal_k}** in the Train & Visualize section!")

            except Exception as e:
                log_error("Find Optimal K", e)
                st.error(f"Analysis failed: {e}")


# ════════════════════════════════════════════════════════════
# SECTION 3 - Train & Visualize Clusters
# ════════════════════════════════════════════════════════════
elif section == "Train & Visualize Clusters":
    st.header("Train K-Means & Visualize Clusters")

    st.sidebar.subheader("Model Settings")
    feature_label = st.sidebar.selectbox("Features", list(FEATURE_OPTIONS.keys()))
    features      = FEATURE_OPTIONS[feature_label]

    suggested_k = st.session_state.get("optimal_k", 5)
    n_clusters  = st.sidebar.slider("Number of Clusters (k)", 2, 10, suggested_k)
    init_method = st.sidebar.selectbox("Init Method", ["k-means++", "random"],
        help="k-means++ initializes centroids based on data patterns")
    max_iter    = st.sidebar.slider("Max Iterations", 50, 500, 300, step=50)

    if st.button("Train K-Means"):
        with st.spinner(f"Training KMeans with k={n_clusters}..."):
            try:
                log_model_training(n_clusters, features, init_method, max_iter)
                kmodel, clustered_df = train_kmeans(
                    df, features, n_clusters,
                    init=init_method, max_iter=max_iter,
                )

                save_model(kmodel, "Cluster_Model.pkl")
                log_model_saved("Cluster_Model.pkl")

                cluster_sizes = clustered_df["Cluster"].value_counts().to_dict()
                log_model_results(n_clusters, kmodel.inertia_, cluster_sizes)

                st.session_state["clustered_df"]      = clustered_df
                st.session_state["kmodel"]            = kmodel
                st.session_state["trained_features"]  = features

                st.success(f"KMeans trained with k={n_clusters}! Model saved.")

                # ── Cluster sizes ────────────────────────────
                st.subheader("Customers Per Cluster")
                col1, col2 = st.columns([1, 2])
                with col1:
                    counts = clustered_df["Cluster"].value_counts().sort_index()
                    for cluster_id, count in counts.items():
                        st.metric(f"Cluster {cluster_id}", f"{count} customers")
                with col2:
                    st.pyplot(plot_cluster_sizes(clustered_df))

                # ── 2D visualization ─────────────────────────
                if len(features) == 2:
                    st.subheader("Cluster Visualization")
                    centers = kmodel.cluster_centers_
                    st.pyplot(plot_clusters_2d(
                        clustered_df, features[0], features[1],
                        centers=centers,
                        title=f"K-Means Clusters (k={n_clusters})",
                    ))
                else:
                    st.subheader("2D Projections (3 feature mode)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.pyplot(plot_clusters_2d(clustered_df, "Annual_Income",
                            "Spending_Score", title="Income vs Spending"))
                    with c2:
                        st.pyplot(plot_clusters_2d(clustered_df, "Age",
                            "Spending_Score", title="Age vs Spending"))

                # ── Cluster profiles ─────────────────────────
                st.subheader("Cluster Profiles — Average Features")
                st.markdown("What does each cluster look like on average?")
                st.pyplot(plot_cluster_profiles(clustered_df, features))

                # ── Centroids table ──────────────────────────
                st.subheader("Cluster Centroids")
                centers_df = pd.DataFrame(kmodel.cluster_centers_, columns=features)
                centers_df.index.name = "Cluster"
                st.dataframe(centers_df.round(2))

                # ── Segment interpretation ───────────────────
                st.subheader("Segment Interpretation")
                st.markdown("""
                Based on **Income vs Spending Score** (k=5), typical segments are:

                | Segment | Income | Spending | Who are they? |
                |---|---|---|---|
                | High Value | High | High | Young spenders — prime marketing targets |
                | Cautious Rich | High | Low | Earn a lot but save — hard to convert |
                | Average | Medium | Medium | Typical shoppers |
                | Low Budget | Low | Low | Budget-conscious, older customers |
                | Impulse | Low | High | Spend beyond means — young impulsive buyers |
                """)

                # ── Sample customers ─────────────────────────
                st.subheader("Sample Customers per Cluster")
                for cid in sorted(clustered_df["Cluster"].unique()):
                    with st.expander(f"Cluster {cid} - sample customers"):
                        st.dataframe(clustered_df[clustered_df["Cluster"] == cid].head(5))

            except Exception as e:
                log_error("Train & Visualize Clusters", e)
                st.error(f"Training failed: {e}")
                import traceback
                st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════
# SECTION 4 - Predict Customer Segment
# ════════════════════════════════════════════════════════════
elif section == "Predict Customer Segment":
    st.header("Predict Customer Segment")
    st.markdown("Enter a new customer's details to find which segment they belong to.")

    try:
        kmodel = load_model("Cluster_Model.pkl")
        n_feat = kmodel.cluster_centers_.shape[1]
        log_model_loaded("Cluster_Model.pkl", kmodel.n_clusters)
        model_available = True
        st.success(f"Trained model loaded! ({kmodel.n_clusters} clusters, {n_feat} features)")
    except Exception:
        model_available = False
        log_warning("No trained model found — user directed to train first.")
        st.warning("No trained model found. Go to **Train & Visualize Clusters** first!")

    if model_available:
        col1, col2 = st.columns(2)
        with col1:
            income   = st.slider("Annual Income ($k)",    10, 150, 60)
            spending = st.slider("Spending Score (1–100)", 1, 100, 50)
        with col2:
            n_feat = kmodel.cluster_centers_.shape[1]
            if n_feat == 3:
                age = st.slider("Age", 18, 70, 35)
            else:
                age = None
                st.info("Current model uses Income + Spending only (2 features). "
                        "Retrain with 3 features to include Age.")

        if st.button("Find My Segment"):
            try:
                features_used = list(FEATURE_OPTIONS.values())[0 if n_feat == 2 else 1]
                cluster = predict_cluster(kmodel, income, spending, age, features_used)
                log_prediction(income, spending, age, cluster)

                st.divider()
                st.success(f"### This customer belongs to **Cluster {cluster}**")

                # Show on scatter plot
                if "clustered_df" in st.session_state:
                    clustered_df = st.session_state["clustered_df"]
                    import matplotlib.pyplot as plt
                    fig = plot_clusters_2d(
                        clustered_df, "Annual_Income", "Spending_Score",
                        title="Where Does This Customer Fall?",
                    )
                    ax = fig.axes[0]
                    ax.scatter([income], [spending], c="red", s=300,
                               marker="*", zorder=10, label="New Customer")
                    ax.legend()
                    st.pyplot(fig)

                # Cluster stats
                if "clustered_df" in st.session_state:
                    cdf  = st.session_state["clustered_df"]
                    cdat = cdf[cdf["Cluster"] == cluster]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cluster Size",        f"{len(cdat)} customers")
                    c2.metric("Avg Income",          f"${cdat['Annual_Income'].mean():.1f}k")
                    c3.metric("Avg Spending Score",  f"{cdat['Spending_Score'].mean():.1f}")

            except Exception as e:
                log_error("Predict Customer Segment", e)
                st.error(f"Prediction failed: {e}")


# ════════════════════════════════════════════════════════════
# SECTION 5 - Activity Log
# ════════════════════════════════════════════════════════════
elif section == "Activity Log":
    st.header("Activity Log")
    st.markdown(
        "All app events — data loads, elbow/silhouette analysis, model training, "
        "predictions, and errors - are recorded in `app_activity.txt`."
    )

    log_text   = get_log_contents()
    line_count = get_log_line_count()

    c1, c2 = st.columns(2)
    c1.metric("Total Log Lines", line_count)
    c2.metric("Log File", "app_activity.txt")

    st.subheader("Log Contents")
    st.text_area("app_activity.txt", value=log_text, height=500)

    st.download_button(
        label="Download app_activity.txt",
        data=log_text,
        file_name="app_activity.txt",
        mime="text/plain",
    )

    if st.button("Clear Log"):
        try:
            with open("app_activity.txt", "w") as f:
                f.write("")
            log_app_start()
            st.success("Log cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not clear log: {e}")


# ── Footer ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Project**")
st.sidebar.markdown("Mall Customer Segmentation — K-Means")
st.sidebar.markdown(f"📋 Log lines: {get_log_line_count()}")
