import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def lsh_query(words, N, filtered_results, genre_index):
    """
    Perform LSH-based search for reviews containing the specified keywords.
    :param words: List of words to search for in reviews.
    :param N: Number of nearest neighbors to return.
    :param filtered_results: List of rows from the dataset (filtered by Octree and categorical conditions).
    :param genre_index: Index of the 'genre_names' column in the dataset.
    :return: List of tuples containing the matching rows and their cosine similarities.
    """
    if not words or N <= 0 or not filtered_results:
        return []

    # Extract genres from the filtered results
    genres_to_hash = []
    for res in filtered_results:
        raw = res[genre_index]
        try:
            genres = ast.literal_eval(raw)
        except:
            # fallback: remove brackets/quotes and split by comma
            genres = str(raw).replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
        
        # clean, lowercase, join into space-separated string
        genres_clean = " ".join([g.strip().lower() for g in genres if g.strip()])
        genres_to_hash.append(genres_clean if genres_clean else "unknown")

    # Vectorize the genres
    vectorizer = CountVectorizer(stop_words='english', binary=True)
    try:
        genre_vectors = vectorizer.fit_transform(genres_to_hash)
    except:
        # If all genres are empty, return empty results
        return []

    # Fit the NearestNeighbors model
    n_neighbors = min(N, len(genres_to_hash))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    nn_model.fit(genre_vectors)

    # Transform the query words into a vector
    query_vector = vectorizer.transform([" ".join([w.lower() for w in words])])

    # Perform the nearest neighbors search
    distances, indices = nn_model.kneighbors(query_vector, n_neighbors=n_neighbors)

    # Collect results with cosine similarity (1 - distance)
    results = [
        (filtered_results[idx], 1 - distances[0][i])
        for i, idx in enumerate(indices[0])
    ]

    return results