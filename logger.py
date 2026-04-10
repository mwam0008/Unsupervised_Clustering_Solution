"""
logger.py - File-based activity logger for Mall Customer Segmentation App
Writes all app events to app_activity.txt
"""

import logging

LOG_FILE = "app_activity.txt"


# ── Configure file logger ─────────────────────────────────────
def _get_logger() -> logging.Logger:
    """Return a configured logger that writes to app_activity.txt."""
    logger = logging.getLogger("mall_segmentation_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


_logger = _get_logger()


# ── Public logging functions ──────────────────────────────────

def log_app_start() -> None:
    """Log when the Streamlit app starts up."""
    _logger.info("=" * 60)
    _logger.info("APP STARTED - Mall Customer Segmentation")
    _logger.info("=" * 60)


def log_page_visit(page: str) -> None:
    """Log which section the user navigated to.

    Args:
        page: Name of the sidebar section visited.
    """
    _logger.info(f"PAGE VISIT        | section='{page}'")


def log_data_loaded(filepath: str, rows: int, cols: int) -> None:
    """Log successful dataset load.

    Args:
        filepath: Path to the CSV file.
        rows: Number of rows loaded.
        cols: Number of columns loaded.
    """
    _logger.info(f"DATA LOADED       | file='{filepath}' rows={rows} cols={cols}")


def log_elbow_method(features: list, k_min: int, k_max: int) -> None:
    """Log when the Elbow method analysis runs.

    Args:
        features: Feature columns used for clustering.
        k_min: Minimum k value tested.
        k_max: Maximum k value tested.
    """
    _logger.info(
        f"ELBOW METHOD      | features={features} "
        f"k_range={k_min}→{k_max}"
    )


def log_silhouette_method(features: list, optimal_k: int,
                           best_score: float) -> None:
    """Log Silhouette analysis result and recommended k.

    Args:
        features: Feature columns analysed.
        optimal_k: Best k value identified by highest silhouette score.
        best_score: The best silhouette score achieved.
    """
    _logger.info(
        f"SILHOUETTE METHOD | features={features} "
        f"optimal_k={optimal_k} best_score={best_score:.4f}"
    )


def log_model_training(n_clusters: int, features: list,
                        init: str, max_iter: int) -> None:
    """Log when KMeans model training starts.

    Args:
        n_clusters: Number of clusters (k).
        features: Feature columns used for training.
        init: Initialization method (k-means++ or random).
        max_iter: Maximum number of iterations.
    """
    _logger.info(
        f"TRAINING STARTED  | model='KMeans' k={n_clusters} "
        f"features={features} init='{init}' max_iter={max_iter}"
    )


def log_model_results(n_clusters: int, inertia: float,
                       cluster_sizes: dict) -> None:
    """Log KMeans training results.

    Args:
        n_clusters: Number of clusters trained.
        inertia: Final WCSS inertia value.
        cluster_sizes: Dict of {cluster_id: customer_count}.
    """
    sizes_str = " ".join([f"C{k}={v}" for k, v in sorted(cluster_sizes.items())])
    _logger.info(
        f"MODEL RESULTS     | model='KMeans' k={n_clusters} "
        f"inertia={inertia:.2f} sizes=[{sizes_str}]"
    )


def log_model_saved(path: str) -> None:
    """Log when the model is saved to disk.

    Args:
        path: File path where the pickle was saved.
    """
    _logger.info(f"MODEL SAVED       | path='{path}'")


def log_model_loaded(path: str, n_clusters: int) -> None:
    """Log when a model is loaded from disk.

    Args:
        path: File path from which the model was loaded.
        n_clusters: Number of clusters in the loaded model.
    """
    _logger.info(f"MODEL LOADED      | path='{path}' k={n_clusters}")


def log_prediction(income: float, spending: float, age,
                   predicted_cluster: int) -> None:
    """Log a customer segment prediction.

    Args:
        income: Customer's annual income ($k).
        spending: Customer's spending score (1–100).
        age: Customer's age, or None if not used.
        predicted_cluster: Predicted cluster ID.
    """
    age_str = str(age) if age is not None else "N/A"
    _logger.info(
        f"PREDICTION        | income=${income}k spending={spending} "
        f"age={age_str} → cluster={predicted_cluster}"
    )


def log_error(context: str, error: Exception) -> None:
    """Log an application error.

    Args:
        context: Description of where the error occurred.
        error: The exception that was raised.
    """
    _logger.error(
        f"ERROR             | context='{context}' "
        f"error={type(error).__name__}: {error}"
    )


def log_warning(message: str) -> None:
    """Log a non-critical warning.

    Args:
        message: Warning message to record.
    """
    _logger.warning(f"WARNING           | {message}")


def get_log_contents() -> str:
    """Read and return the full contents of the log file.

    Returns:
        Log file contents as a string, or a placeholder if not found.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No log file found yet. Activity will appear here after the app is used."
    except Exception as e:
        return f"Could not read log file: {e}"


def get_log_line_count() -> int:
    """Return the number of lines in the log file.

    Returns:
        Line count, or 0 if the file does not exist.
    """
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0
