# src/generate_gold.py
import csv
from src.query import recommend

# Sample test queries (you can expand this list)
queries = [
    "Neural networks for image classification",
    "Deep learning for medical image segmentation",
    "Optimization algorithms for wireless sensor networks",
    "Fuzzy logic in control systems",
    "Image compression using deep learning"
]

# Generate ground truth style CSV automatically from top authors per query
rows = []
for q in queries:
    print(f"Running query: {q}")
    result = recommend(q, top_k=50, rerank_k=10, device="cpu")
    authors = [a for a, _ in result["author_rank"][:5]]  # top 5 authors
    rows.append({"query": q, "relevant_authors": ";".join(authors)})

# Write CSV
with open("test_gold.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "relevant_authors"])
    writer.writeheader()
    writer.writerows(rows)

print("✅ test_gold.csv created successfully.")
