from pathlib import Path

import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from tqdm.auto import tqdm

from embedder import Embedder


QUERY_Q1 = "How does approximate nearest neighbor search work?"
QUERY_Q4 = "What metric do we use to evaluate a search engine?"
QUERY_Q5 = "How do I store vectors in PostgreSQL?"
QUERY_Q6 = "How do I give the model access to tools?"


def load_documents():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    return [file.parse() for file in reader.read()]


def build_vector_search(chunks, vectors):
    search = VectorSearch(keyword_fields={"filename"})
    search.fit(vectors, chunks)
    return search


def build_text_search(chunks):
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(chunks)
    return index


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def filenames(results):
    return [result["filename"] for result in results]


def main():
    base_dir = Path(__file__).resolve().parent
    embedder = Embedder(base_dir / "models" / "Xenova" / "all-MiniLM-L6-v2")

    query_vector = embedder.encode(QUERY_Q1)
    q1 = float(query_vector[0])

    documents = load_documents()
    target = next(
        doc
        for doc in documents
        if doc["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
    )
    target_vector = embedder.encode(target["content"])
    q2 = float(target_vector.dot(query_vector))

    chunks = chunk_documents(documents, size=2000, step=1000)
    vectors = embedder.encode_batch(
        [chunk["content"] for chunk in tqdm(chunks, desc="Embedding chunks")]
    )
    matrix = np.vstack(vectors)

    scores = matrix.dot(query_vector)
    q3_chunk = chunks[int(scores.argmax())]

    vector_search = build_vector_search(chunks, matrix)
    text_search = build_text_search(chunks)

    q4_vector = embedder.encode(QUERY_Q4)
    q4_results = vector_search.search(q4_vector, num_results=5)

    q5_vector = embedder.encode(QUERY_Q5)
    q5_vector_results = vector_search.search(q5_vector, num_results=5)
    q5_text_results = text_search.search(QUERY_Q5, num_results=5)
    q5_vector_files = set(filenames(q5_vector_results))
    q5_text_files = set(filenames(q5_text_results))
    q5_only_vector = sorted(q5_vector_files - q5_text_files)

    q6_vector = embedder.encode(QUERY_Q6)
    q6_vector_results = vector_search.search(q6_vector, num_results=5)
    q6_text_results = text_search.search(QUERY_Q6, num_results=5)
    q6_results = rrf([q6_vector_results, q6_text_results])

    print("Q1 v[0]:", q1)
    print("Q2 cosine:", q2)
    print("Q3 best chunk:", q3_chunk["filename"], "start:", q3_chunk["start"])
    print("Q4 vector top files:", filenames(q4_results))
    print("Q5 vector top files:", filenames(q5_vector_results))
    print("Q5 text top files:", filenames(q5_text_results))
    print("Q5 only vector:", q5_only_vector)
    print("Q6 vector top files:", filenames(q6_vector_results))
    print("Q6 text top files:", filenames(q6_text_results))
    print("Q6 RRF top files:", filenames(q6_results))

    answers = {
        "Q1": "-0.02",
        "Q2": "0.37",
        "Q3": q3_chunk["filename"],
        "Q4": q4_results[0]["filename"],
        "Q5": q5_only_vector,
        "Q6": q6_results[0]["filename"],
    }
    print("Answers:", answers)


if __name__ == "__main__":
    main()
