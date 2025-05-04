import numpy as np
import random
from numpy.linalg import norm

class DecompositionEngine:
    def __init__(self, rank=2):
        self.terms = []
        self.documents = []
        self.matrix = None
        self.rank = rank
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
                    self.matrix[term_idx, doc_idx] = 1
    
    def power_iteration(self, A, num_iterations=100):
        n = A.shape[1]
        v = np.random.rand(n)
        v = v / norm(v)
        
        for _ in range(num_iterations):
            Av = A.dot(v)
            v_new = Av / norm(Av)
            if np.allclose(v, v_new, atol=1e-10):
                break
            v = v_new
        
        eigenvalue = norm(A.dot(v))
        return eigenvalue, v
    
    def compute_decomposition(self):
        if self.matrix is None:
            raise ValueError("Matrix not built")
        
        D = self.matrix
        m, n = D.shape
        
        A = D.T @ D
        eigenvalues_V = []
        eigenvectors_V = []
        
        for _ in range(self.rank):
            if len(eigenvalues_V) > 0:
                B = A - sum(s * np.outer(v, v) for s, v in zip(eigenvalues_V, eigenvectors_V))
                sigma, v = self.power_iteration(B)
            else:
                sigma, v = self.power_iteration(A)
            
            eigenvalues_V.append(sigma)
            eigenvectors_V.append(v)
  
        V = np.column_stack(eigenvectors_V)
        S_V = np.sqrt(np.array(eigenvalues_V))
        U = D @ V @ np.diag(1 / S_V)
        
        order = np.argsort(S_V)[::-1]
        self.S = S_V[order]
        self.VT = V[:, order].T
        self.U = U[:, order]
        
        return self.U, np.diag(self.S), self.VT
    
    def query(self, query_terms, threshold=0.8):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_decomposition()
      
        q = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                q[term_idx] = 1
       
        q_k = self.U[:, :self.rank].T @ q
        
        scores = []
        for doc_idx in range(len(self.documents)):
            d_k = self.S[:self.rank] * self.VT[:self.rank, doc_idx]
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


# Extraction des documents depuis un fichier
def extract_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    n = len(content)
    doc = []
    terms = set()
    index1 = 0
    nb = 2
    while index1 < n - 1:
        index2 = find(content, index1, str(nb))
        doc.append(content[index1:index2][len(str(nb - 1)) + 1:-1])
        for i in doc[-1].split():
            terms.add(i)
        index1 = index2
        nb += 1
    return doc, terms


def find(content, i, j):
    while i < len(content) - 3 and content[i:i + len(j)] != j:
        i += 1
    if i >= len(content) - 3:
        return len(content)
    return i


# Charger les documents depuis le fichier
file_path = 'c:/Users/waela/OneDrive/Desktop/TP/ProjetL_python/Documents.txt'
doc, terms = extract_documents(file_path)

# Construire les documents sous forme de tuples (nom, termes)
documents = [("Doc " + str(i + 1), doc[i].split()) for i in range(len(doc))]

# Initialiser le modèle et construire la matrice termes-documents
engine = DecompositionEngine(rank=2)
engine.build_matrix(documents)

# Générer une requête aléatoire à partir des termes
random_query = random.sample(list(terms), 3)  # Générer une requête avec 5 termes aléatoires
print("Requête aléatoire générée :", random_query)

# Traiter la requête
results = engine.query(random_query, threshold=0.2)

# Afficher les résultats
if results:
    print("\nDocuments pertinents triés par score :")
    for doc, score in results[:min(6, len(results))]:
        print(f"- {doc}: score = {score:.2f}")
else:
    print("Aucun document pertinent trouvé.")