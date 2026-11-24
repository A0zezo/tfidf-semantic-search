# tfidf-semantic-search

Small vector-search example: TF-IDF (unigrams and bigrams, English stop words) with cosine similarity over a 15-line text corpus, using scikit-learn. It is lexical search, so it matches shared words, not meaning.

## Run

```bash
pip install -r requirements.txt
python search.py "containers for applications" -k 2
pytest
```

## Output (reproduced)

```
$ python search.py "containers for applications" -k 2
0.471  [2] Docker packages applications and their dependencies into containers.

$ python search.py "how do I find similar embeddings"
0.263  [3] A vector database stores embeddings and supports nearest neighbour search.

$ pytest -q
4 passed in 0.46s
```

## Limits

- Only word overlap counts: a query with no shared vocabulary returns nothing.
- The corpus is 15 hand-written sentences, so this shows the mechanism and is not a benchmark.
