import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.query import load_meta, META_PATH, TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH

print("Loading metadata...")
# Use your existing metadata loader
metadata = load_meta(META_PATH) 

# We need a stable order, so we build from the dict's items
# (Assuming metadata is {0: {...}, 1: {...}})
ids = sorted(metadata.keys())
corpus = [metadata[i].get('text', '') for i in ids]

print("Building TF-IDF Vectorizer...")
# You can tune these parameters
vectorizer = TfidfVectorizer(
    max_df=0.8,         # Ignore terms in > 80% of docs
    min_df=5,           # Ignore terms in < 5 docs
    stop_words="english",
    ngram_range=(1, 2)  # Use unigrams and bigrams
)

tfidf_matrix = vectorizer.fit_transform(corpus)

print(f"Matrix shape: {tfidf_matrix.shape}")

print(f"Saving models to {TFIDF_VECTORIZER_PATH}...")
joblib.dump(vectorizer, TFIDF_VECTORIZER_PATH)
joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)

print("Done.")