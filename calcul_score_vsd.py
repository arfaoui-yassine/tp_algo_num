import numpy as np
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
        #vsd_function
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
    
    def reduced_approximation(self):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_decomposition()
        
        Sigma_k = np.diag(self.S[:self.rank])
        U_k = self.U[:, :self.rank]
        VT_k = self.VT[:self.rank, :]
        
        return U_k @ Sigma_k @ VT_k
    
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
