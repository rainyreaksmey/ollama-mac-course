"""RAG: answer from MY documents. Embed every chunk, find the closest ones, put them in the prompt."""
import sys
from pathlib import Path
import numpy as np
import ollama

EMBED, LLM, TOP_K = "nomic-embed-text", "llama3.2:3b", 3

# 1. split the docs into small chunks (one paragraph each)
chunks = [p.strip() for f in sorted(Path("docs").glob("*.md")) for p in f.read_text().split("\n\n") if p.strip()]

# 2. embed every chunk once
vectors = np.array(ollama.embed(model=EMBED, input=chunks).embeddings)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

# 3. embed the question and take the most similar chunks (cosine similarity)
question = sys.argv[1] if len(sys.argv) > 1 else "What is Ollama and which port does it use?"
q = np.array(ollama.embed(model=EMBED, input=question).embeddings[0])
scores = vectors @ (q / np.linalg.norm(q))
best = np.argsort(scores)[::-1][:TOP_K]
for i in best:
    print(f"  {scores[i]:.3f}  {chunks[i][:70]}…")

# 4. answer using ONLY those chunks
context = "\n\n".join(chunks[i] for i in best)
answer = ollama.generate(model=LLM, prompt=f"Answer using only this context:\n\n{context}\n\nQuestion: {question}")
print("\n" + answer.response)
