"""TF-IDF vector search over a plain text corpus (one document per line)."""
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Index:
    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)

    def search(self, query, k=3):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(int(i), float(scores[i]), self.docs[i]) for i in order if scores[i] > 0]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--corpus", default="corpus.txt")
    a = p.parse_args()
    for i, s, d in Index(load(a.corpus)).search(a.query, a.k):
        print(f"{s:.3f}  [{i}] {d}")
