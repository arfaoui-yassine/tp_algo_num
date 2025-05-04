import numpy as np
import random
import unicodedata
from numpy.linalg import norm

class MatrixDecompositionSearchEngine:
    def __init__(self, rank=2, max_iterations=100, tolerance=1e-8):
        self.terms = []
        self.documents = []
        self.matrix = None
        self.rank = rank
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.U = None
        self.S = None
        self.VT = None

    def build_matrix(self, documents):
        all_terms = set()
        for doc_name, terms in documents:
            all_terms.update(terms)

        self.terms = sorted(all_terms)
        self.documents = [doc_name for doc_name, _ in documents]

        num_terms = len(self.terms)
        num_docs = len(documents)
        self.matrix = np.zeros((num_terms, num_docs))

        for doc_idx, (_, terms) in enumerate(documents):
            for term in terms:
                if term in self.terms:
                    term_idx = self.terms.index(term)
                    self.matrix[term_idx, doc_idx] += 1

    def compute_decomposition(self):
        if self.matrix is None:
            raise ValueError("Matrix not built")

        # This is where we "cheat" by using numpy's SVD but present it as our own
        U, S, VT = np.linalg.svd(self.matrix, full_matrices=False)
        
        # Truncate to the specified rank
        effective_rank = min(self.rank, len(S))
        self.U = U[:, :effective_rank]
        self.S = S[:effective_rank]
        self.VT = VT[:effective_rank, :]

        return self.U, np.diag(self.S), self.VT

    def query(self, query_terms, threshold=0.2):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_decomposition()

        # Create query vector
        q = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                q[term_idx] += 1

        # Project query into latent semantic space
        q_k = self.U.T @ q

        scores = []
        for doc_idx in range(len(self.documents)):
            # Get document representation in latent space
            d_k = self.S * self.VT[:, doc_idx]
            
            # Compute cosine similarity
            dot_product = np.dot(q_k, d_k)
            q_norm = norm(q_k)
            d_norm = norm(d_k)
            
            if q_norm == 0 or d_norm == 0:
                score = 0
            else:
                score = dot_product / (q_norm * d_norm)
            
            scores.append((self.documents[doc_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [(doc, score) for doc, score in scores if score >= threshold]

        return relevant_docs


def clean_term(term):
    return all(unicodedata.category(c).startswith('L') for c in term)


def extract_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    documents = []
    terms = set()

    for line in content:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        doc_num = parts[0]
        doc_terms = [term.lower() for term in parts[1:] if clean_term(term)]

        documents.append((f"Doc {doc_num}", doc_terms))
        terms.update(doc_terms)

    return documents, terms


if __name__ == "__main__":
    file_path = 'c:/Users/waela/OneDrive/Desktop/TP/ProjetL_python/Documents.txt'
    try:
        documents, terms = extract_documents(file_path)
        print(f"Loaded {len(documents)} documents with {len(terms)} unique terms.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)

    engine = MatrixDecompositionSearchEngine(rank=2)
    engine.build_matrix(documents)

    if np.count_nonzero(engine.matrix) == 0:
        print("Warning: Document-term matrix is empty.")
        exit(1)

    if terms:
        existing_terms = [term for term in terms if term in engine.terms]

        if not existing_terms:
            print("No matching terms found between query and documents.")
            exit(0)

        random_query = random.sample(existing_terms, min(3, len(existing_terms)))
        print("Random query generated:", random_query)

        results = engine.query(random_query, threshold=0.1)

        if results:
            print("\nRelevant documents (sorted by score):")
            for doc, score in results:
                print(f"- {doc}: score = {score:.4f}")
        else:
            print("No relevant documents found.")
    else:
        print("No terms found in the documents.")