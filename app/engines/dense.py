import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

class DenseEngine:
    def __init__(self, products, n_components: int = 128):
        self.products = products
        self.index = {p.id: i for i, p in enumerate(products)}
        self.vectorizer = None
        self.svd = None
        self.matrix = None

        texts = []
        for p in products:
            attrs = ' '.join(f'{k} {v}' for k, v in (p.attributes or {}).items())
            texts.append(' '.join([
                p.title or '',
                p.description or '',
                ' '.join(p.category or []),
                p.brand or '',
                attrs
            ]).strip())

        if not texts or all(not t for t in texts):
            return

        try:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            n_samples, n_features = tfidf_matrix.shape

            # TruncatedSVD requires n_components < min(n_samples, n_features)
            max_components = min(n_samples - 1, n_features - 1)
            if max_components >= 1:
                import warnings
                components = min(n_components, max_components)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.svd = TruncatedSVD(n_components=components, random_state=42)
                    dense_matrix = self.svd.fit_transform(tfidf_matrix)
            else:
                # Fallback to dense array of TF-IDF directly if dataset is too small for SVD
                dense_matrix = tfidf_matrix.toarray()

            self.matrix = normalize(dense_matrix, norm='l2', axis=1)
        except Exception:
            self.vectorizer = None
            self.svd = None
            self.matrix = None

    def similar(self, product_id: str, limit: int = 100) -> list[tuple[str, float]]:
        if self.matrix is None or product_id not in self.index:
            return []
        i = self.index[product_id]
        target_vec = self.matrix[i]
        sims = np.dot(self.matrix, target_vec).ravel()
        order = sims.argsort()[::-1]
        return [(self.products[j].id, float(sims[j])) for j in order if j != i][:limit]

    def query(self, query: str, limit: int = 100) -> list[tuple[str, float]]:
        if self.matrix is None or self.vectorizer is None or not query or not query.strip():
            return []
        try:
            q_tfidf = self.vectorizer.transform([query])
            if q_tfidf.nnz == 0:
                return []
            if self.svd is not None:
                q_dense = self.svd.transform(q_tfidf)
            else:
                q_dense = q_tfidf.toarray()

            q_norm = normalize(q_dense, norm='l2', axis=1)
            sims = np.dot(self.matrix, q_norm.T).ravel()
            order = sims.argsort()[::-1]
            return [(self.products[j].id, float(sims[j])) for j in order[:limit] if sims[j] > 0]
        except Exception:
            return []
