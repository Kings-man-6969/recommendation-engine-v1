from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentEngine:
    def __init__(self, products):
        self.products = products
        self.index = {p.id: i for i, p in enumerate(products)}
        texts = []
        for p in products:
            attrs = ' '.join(f'{k} {v}' for k, v in p.attributes.items())
            texts.append(' '.join([p.title or '', p.description or '', ' '.join(p.category or []), p.brand or '', attrs]))
        if texts:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.vectorizer = None
            self.matrix = None

    def similar(self, product_id, limit=100):
        if self.matrix is None or product_id not in self.index:
            return []
        i = self.index[product_id]
        sims = cosine_similarity(self.matrix[i], self.matrix).ravel()
        order = sims.argsort()[::-1]
        return [(self.products[j].id, float(sims[j])) for j in order if j != i][:limit]

    def query(self, query, limit=100):
        if self.matrix is None or self.vectorizer is None or not query:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        return [(self.products[j].id, float(sims[j])) for j in sims.argsort()[::-1][:limit] if sims[j] > 0]
