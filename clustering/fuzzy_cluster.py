from sklearn.mixture import GaussianMixture


def perform_fuzzy_clustering(embeddings, n_clusters=10):

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=42
    )

    gmm.fit(embeddings)

    # probability distribution for each document
    probabilities = gmm.predict_proba(embeddings)

    return gmm, probabilities
