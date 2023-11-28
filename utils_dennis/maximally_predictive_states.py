import numbers
import numpy as np
import matplotlib.pyplot as plt
import scipy
import deeptime as dtime

from sklearn.cluster import MiniBatchKMeans, KMeans

def time_delay_embedding(data: np.ndarray, delay: int) -> np.ndarray:
    """Create a time-delay embedding of a N times D numpy data array.

    Args:
        data (np.ndarray): The input data array of shape (N, D).
        delay (int): The time delay (lag) for creating the embedding.

    Returns:
        np.ndarray: The time-delay embedded data array of shape (N - delay, D * (delay + 1)).
    """

    # Check if the input data is a numpy array
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")

    # Check if delay is a non-negative integer
    if not isinstance(delay, numbers.Integral) or delay < 0:
        raise ValueError("Delay must be a non-negative integer")

    # Create the time-delay embedding
    embedded_data = np.hstack([np.roll(data, i, axis=0) for i in range(delay + 1)])

    # Remove the rows with NaN values introduced by rolling
    embedded_data = embedded_data[delay:]

    return embedded_data


def cluster_data(data: np.ndarray, n_clusters: int, algorithm: str = 'kmeans',
                 random_state: int or None = None,
                 batch_size: int or None = None) -> tuple:
    """
    Perform clustering on the input data using either KMeans or MiniBatchKMeans.

    Args:
        data (np.ndarray): The input data array of shape (N, D).
        n_clusters (int): The number of clusters to form.
        algorithm (str): The clustering algorithm to use, either 'kmeans' or 'minibatchkmeans' (default: 'kmeans').
        random_state (int or None): The random seed for reproducibility (default: None).
        batch_size (int or None): The number of samples to use for each mini-batch (only applicable for 'minibatchkmeans').

    Returns:
        tuple: A tuple containing:
        np.ndarray: The transformed discrete time series.
        np.ndarray: The cluster centers.
        model: The kmeans model.
    """

    # Perform clustering based on the selected algorithm
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=3)
    elif algorithm == 'minibatchkmeans':
        if batch_size is None:
            batch_size = 3 * n_clusters
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state,
                                batch_size=batch_size, n_init=3)
    else:
        raise ValueError(
            "Invalid algorithm. Choose either 'kmeans' or 'minibatchkmeans'")

    # Fit the model and get cluster assignments and centers
    cluster_assignments = model.fit_predict(data)
    cluster_centers = model.cluster_centers_

    return cluster_assignments, cluster_centers, model


def compare_time_series(data1: np.ndarray, data2: np.ndarray, k: int,
                        label1: str = 'Data 1', label2: str = 'Data 2'):
    """
    Compare the first K dimensions of two multi-dimensional time series on the same subplot.

    Parameters:
    - data1: 2D array-like, shape (N, D)
      The first multi-dimensional time series data.
    - data2: 2D array-like, shape (N, D)
      The second multi-dimensional time series data.
    - k: int
      The number of dimensions to compare.
    - label1: str, optional (default='Data 1')
      Label for the first data source.
    - label2: str, optional (default='Data 2')
      Label for the second data source.
    """

    N, D = data1.shape
    time_steps = np.arange(N)

    # Create subplots for each selected dimension
    fig, axes = plt.subplots(k, 1, figsize=(10, 2 * k), sharex=True)

    for i in range(min(k, D)):
        axes[i].plot(time_steps, data1[:, i], label=label1)
        axes[i].plot(time_steps, data2[:, i], label=label2)
        axes[i].set_ylabel(f'Dimension {i + 1}')

    plt.xlabel('Time Steps')
    plt.suptitle(f'Comparison of First {k} Dimensions of Time Series')

    # Show legend only once
    axes[0].legend()

    plt.show()


def get_entropy(transition_matrix: scipy.sparse.csr_matrix) -> float:
    """
    Calculate the entropy of a Markov chain represented by a transition matrix.

    Args:
        transition_matrix (scipy.sparse.csr_matrix): The transition matrix of the Markov chain.

    Returns:
        float: The entropy of the Markov chain.
    """
    # Calculate the stationary distribution
    mu = dtime.markov.tools.analysis.stationary_distribution(transition_matrix)

    # Convert stationary distribution to a sparse diagonal matrix
    mu_as_sparse_matrix = scipy.sparse.diags(mu)

    # Create a copy of the transition matrix and replace non-zero entries with their logarithms
    logTM = transition_matrix.copy()
    logTM.data = np.log(logTM.data)

    # Compute the entropy using the stationary distribution and logarithmic transition matrix
    entropy = -mu_as_sparse_matrix.dot(transition_matrix.multiply(logTM)).sum()
    return entropy
